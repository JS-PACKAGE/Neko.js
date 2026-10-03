import { env } from '@huggingface/transformers';
import { type ModelProfileId } from './manifest.js';
import { getRegisteredModelProfile, type ModelId, type RegisteredModelProfile } from './registry.js';
import { exportModelBundle, importModelBundle, type BundleBacking, type ModelBundleSource } from './bundle.js';
import type { ModelFileName } from './manifest.js';
import { NekoError } from '../errors.js';
import { authorizeNetwork, type ResourcePolicy } from '../web/policy.js';
import type * as NodeFs from 'node:fs/promises';
import type * as NodePath from 'node:path';
import type { constants as NodeFileConstants, Stats } from 'node:fs';
import type * as NodeOs from 'node:os';
import type * as NodeStream from 'node:stream';
import { downloadResumable } from './download.js';
import type { DownloadProgress, DownloadStage } from './download.js';
import { nodeDownloadStage } from './node-download.js';
import { browserDownloadStage } from './browser-download.js';
import { withBrowserInstallLock, withNodeInstallLock } from './lock.js';
import type { AsyncLocalStorage } from 'node:async_hooks';

const isNode = typeof process !== 'undefined' && process.release?.name === 'node';

export class ModelIntegrityError extends Error { override name = 'ModelIntegrityError'; }
export interface ModelSource {
  /** Absolute HTTP(S) directory containing only the selected profile's pinned model files. */
  baseUrl: string;
}
/** Own the URL without losing prototype/private-field getters during worker serialization. */
export function captureModelSource(source?: ModelSource): ModelSource | undefined {
  if (source === undefined) return undefined;
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    throw new TypeError('modelSource must be an object with a string baseUrl');
  }
  const baseUrl = source.baseUrl;
  if (typeof baseUrl !== 'string') throw new TypeError('modelSource must be an object with a string baseUrl');
  const base = new URL(baseUrl);
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash) {
    throw new TypeError('modelSource.baseUrl must be an absolute HTTP(S) directory without credentials, query, or fragment');
  }
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return Object.freeze({ baseUrl: base.href });
}
export interface CacheProgress extends DownloadProgress { file: ModelFileName; }
export interface VerifiedCacheOptions {
  /** Transformers.js native filesystem cache directory; browsers use its native Cache API cache. */
  cacheDir?: string;
  localFilesOnly?: boolean;
  onProgress?: (event: CacheProgress) => void;
  profile?: ModelProfileId;
  model?: ModelId;
  policy?: ResourcePolicy;
  modelSource?: ModelSource;
  /** Maximum simultaneous pinned-file downloads, from 1 through 16. Defaults to 3. */
  downloadConcurrency?: number;
  /** Preserve interrupted, validator-bound staging for the next explicit installation. Defaults to true. */
  resumeDownloads?: boolean;
}
export interface ModelCacheStatus { downloaded: boolean; verified: boolean; bytes: number; totalBytes: number; path: string; files: { name: ModelFileName; present: boolean; verified: boolean; bytes: number }[]; }
export interface ModelCacheDiagnostics {
  model: ModelId;
  revision: string;
  profile: ModelProfileId;
  backend: 'filesystem' | 'cache-storage';
  path: string;
  requiredBytes: number;
  storage: { usage: number | null; quota: number | null; available: number | null; persisted: boolean | null; source: 'browser-storage-estimate' | 'unavailable' };
}
export interface VerifiedCacheInstallation {
  restore(): void;
  prefetch(signal?: AbortSignal, files?: readonly ModelFileName[]): Promise<void>;
  status(signal?: AbortSignal): Promise<ModelCacheStatus>;
  clear(signal?: AbortSignal): Promise<void>;
  exportBundle(signal?: AbortSignal): ReadableStream<Uint8Array>;
  importBundle(source: ModelBundleSource, signal?: AbortSignal): Promise<void>;
  diagnostics(signal?: AbortSignal): Promise<ModelCacheDiagnostics>;
  withSignal<T>(signal: AbortSignal | undefined, operation: () => Promise<T>): Promise<T>;
}

type ResourceSpec = { readonly size: number; readonly sha256: string };
type CachedResponse = Pick<Response, 'body' | 'headers' | 'status'> & { filePath?: string };
type NativeCache = {
  match(request: string): Promise<CachedResponse | undefined>;
  put(request: string, response: Response): Promise<void>;
  delete(request: string): Promise<void>;
  stage(request: string): Promise<DownloadStage>;
  withLock<T>(request: string, operation: () => Promise<T>, signal?: AbortSignal): Promise<T>;
  path: string;
};
type NodeHash = { update(bytes: Uint8Array): NodeHash; digest(encoding: 'hex'): string };
let nodeCrypto: Promise<{ createHash(algorithm: string): NodeHash }> | undefined;
let installed = false;
const verifiedResponses = new WeakMap<Response, ResourceSpec>();

function fileName(request: string, model: RegisteredModelProfile): ModelFileName | undefined {
  if (!request.startsWith(model.baseUrl)) return undefined;
  const name = request.slice(model.baseUrl.length);
  if (!Object.hasOwn(model.files, name)) throw new ModelIntegrityError(`Unpinned model resource for selected profile: ${request}`);
  return name as ModelFileName;
}
function nativeKey(request: string, model: RegisteredModelProfile, name?: ModelFileName): string {
  return isNode && name ? `${model.id}/${model.revision}/${name}` : request;
}
function isRuntimeAsset(request: string): boolean {
  if (isNode || typeof location === 'undefined') return false;
  try {
    const decoded = decodeURIComponent(request);
    if (decoded.includes('\\') || decoded.split('/').some((part) => part === '.' || part === '..')) return false;
    const url = new URL(request, location.href);
    return url.origin === location.origin && /\.(wasm|mjs)$/.test(url.pathname);
  } catch { return false; }
}
async function createNodeHash(): Promise<NodeHash> {
  // A computed import keeps Node's crypto implementation out of the browser bundle.
  const protocol = 'node:';
  nodeCrypto ??= import(`${protocol}crypto`);
  return (await nodeCrypto).createHash('sha256');
}
function integrity(name: ModelFileName, detail: string): ModelIntegrityError {
  return new ModelIntegrityError(`Model integrity mismatch (${detail}): ${name}`);
}
function checkLength(response: CachedResponse, name: ModelFileName, spec: ResourceSpec): void {
  const length = response.headers.get('content-length');
  if (length !== null && Number(length) !== spec.size) {
    void response.body?.cancel().catch(() => undefined);
    throw integrity(name, 'declared size');
  }
  if (!response.body) throw integrity(name, 'missing body');
}
function verifiedResponse(bytes: ArrayBuffer, response: CachedResponse, spec: ResourceSpec): Response {
  const headers = new Headers(response.headers);
  headers.set('content-length', String(spec.size));
  const body = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(new Uint8Array(bytes)); controller.close(); } });
  return new Response(body, { status: response.status, headers });
}
async function readVerified(response: CachedResponse, name: ModelFileName, spec: ResourceSpec, options: VerifiedCacheOptions, phase: CacheProgress['phase'], signal?: AbortSignal): Promise<Response> {
  checkLength(response, name, spec);
  signal?.throwIfAborted();
  const bytes = new Uint8Array(spec.size);
  const hash = isNode ? await createNodeHash() : undefined;
  const reader = response.body!.getReader();
  const cancel = () => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener('abort', cancel, { once: true });
  let loaded = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) break;
      if (value.byteLength > spec.size - loaded) throw integrity(name, 'exceeds pinned size');
      bytes.set(value, loaded);
      hash?.update(value);
      loaded += value.byteLength;
      options.onProgress?.({ file: name, loaded, total: spec.size, phase });
    }
    signal?.throwIfAborted();
    if (loaded !== spec.size) throw integrity(name, 'size');
    let digest: string;
    if (hash) digest = hash.digest('hex');
    else {
      if (!globalThis.crypto?.subtle) throw new Error('Verified model loading requires WebCrypto in a secure context');
      const sha = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
      digest = '';
      for (const byte of sha) digest += byte.toString(16).padStart(2, '0');
    }
    signal?.throwIfAborted();
    if (digest !== spec.sha256) throw integrity(name, 'SHA-256');
    const verified = verifiedResponse(bytes.buffer, response, spec);
    verifiedResponses.set(verified, spec);
    return verified;
  } catch (error) {
    await reader.cancel(error).catch(() => undefined);
    throw error;
  } finally { signal?.removeEventListener('abort', cancel); reader.releaseLock(); }
}
async function hashCachedFile(response: CachedResponse, name: ModelFileName, spec: ResourceSpec, options: VerifiedCacheOptions, signal?: AbortSignal): Promise<void> {
  checkLength(response, name, spec);
  const hash = await createNodeHash();
  const reader = response.body!.getReader();
  const cancel = () => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener('abort', cancel, { once: true });
  let loaded = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) break;
      if (value.byteLength > spec.size - loaded) throw integrity(name, 'exceeds pinned size');
      hash.update(value);
      loaded += value.byteLength;
      options.onProgress?.({ file: name, loaded, total: spec.size, phase: 'verify' });
    }
    signal?.throwIfAborted();
    if (loaded !== spec.size || hash.digest('hex') !== spec.sha256) throw integrity(name, 'size or SHA-256');
    signal?.throwIfAborted();
  } catch (error) {
    await reader.cancel(error).catch(() => undefined);
    throw error;
  } finally { signal?.removeEventListener('abort', cancel); reader.releaseLock(); }
}
function rejectedResponse(error: ModelIntegrityError): Response {
  // Transformers.js swallows rejected cache matches. An errored body makes corruption a load failure, not a cache miss.
  const response = new Response(new ReadableStream({ start(controller) { controller.error(error); } }));
  // Chromium's native body reader replaces stream errors with TypeError('Failed to fetch').
  response.arrayBuffer = async () => { throw error; };
  return response;
}
async function nativeCache(options: VerifiedCacheOptions): Promise<NativeCache> {
  const model = getRegisteredModelProfile(options.profile, options.model);
  if (!isNode) {
    if (typeof caches === 'undefined') throw new Error('Verified model caching requires the browser Cache API in a secure context');
    const cache = await caches.open(env.cacheKey);
    const namespace = `CacheStorage:${env.cacheKey}`;
    return {
      path: namespace, match: (request) => cache.match(request), put: (request, response) => cache.put(request, response),
      async delete(request) { await cache.delete(request); },
      stage(request) { return browserDownloadStage(`${namespace}:${request}`, async (body) => { await cache.put(request, new Response(body)); }); },
      withLock(request, operation, signal) { return withBrowserInstallLock(`${namespace}:${request}`, operation, signal); },
    };
  }
  // Platform-only imports cannot be static: browsers have no Node filesystem or package source files.
  const protocol = 'node:';
  const fs: typeof NodeFs = await import(`${protocol}fs/promises`);
  const path: typeof NodePath = await import(`${protocol}path`);
  const os: typeof NodeOs = await import(`${protocol}os`);
  const directory = options.cacheDir ?? (process.platform === 'darwin'
    ? path.join(os.homedir(), 'Library', 'Caches', 'neko.js')
    : process.platform === 'win32' ? path.join(process.env.LOCALAPPDATA ?? path.join(os.homedir(), 'AppData', 'Local'), 'neko.js')
    : path.join(process.env.XDG_CACHE_HOME ?? path.join(os.homedir(), '.cache'), 'neko.js'));
  if (!directory) throw new Error('Transformers.js filesystem cache directory is unavailable');
  const root = path.resolve(directory);
  if (root === path.parse(root).root) throw new ModelIntegrityError('Cache directory cannot be a filesystem root');
  const uid = process.getuid?.();
  const missing = (error: unknown) => typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT';
  const owned = (info: Stats) => uid === undefined || info.uid === uid;
  async function secureDirectory(): Promise<void> {
    const ancestors: string[] = [];
    for (let current = root; current !== path.dirname(current); current = path.dirname(current)) ancestors.push(current);
    ancestors.push(path.parse(root).root);
    for (const current of ancestors.reverse()) {
      let info;
      try { info = await fs.lstat(current); }
      catch (error) {
        if (!missing(error)) throw error;
        await fs.mkdir(current, { mode: 0o700 }).catch((failure: unknown) => {
          if (!failure || typeof failure !== 'object' || !('code' in failure) || failure.code !== 'EEXIST') throw failure;
        });
        info = await fs.lstat(current);
      }
      if (info.isSymbolicLink() || !info.isDirectory()) throw new ModelIntegrityError(`Unsafe cache directory: ${current}`);
      // A private cache root is still renameable through an untrusted or non-sticky writable ancestor.
      if (uid !== undefined && current !== root && (info.uid !== uid && info.uid !== 0 || (info.mode & 0o022) !== 0 && (info.mode & 0o1000) === 0)) {
        throw new ModelIntegrityError(`Cache ancestor permits untrusted namespace replacement: ${current}`);
      }
    }
    const info = await fs.lstat(root);
    if (!owned(info)) throw new ModelIntegrityError('Cache directory must belong to the current user');
    if (await fs.realpath(root) !== root) throw new ModelIntegrityError('Cache directory changed during validation');
    await fs.chmod(root, 0o700);
  }
  async function secureEntry(request: string, createParents: boolean): Promise<string | undefined> {
    const prefix = `${model.id}/${model.revision}/`;
    if (!request.startsWith(prefix) || !Object.hasOwn(model.allFiles, request.slice(prefix.length))) return undefined;
    await secureDirectory();
    const parts = request.split('/');
    let current = root;
    for (const part of parts.slice(0, -1)) {
      current = path.join(current, part);
      let info;
      try { info = await fs.lstat(current); }
      catch (error) {
        if (!missing(error)) throw error;
        if (!createParents) return undefined;
        await fs.mkdir(current, { mode: 0o700 }).catch((failure: unknown) => {
          if (!failure || typeof failure !== 'object' || !('code' in failure) || failure.code !== 'EEXIST') throw failure;
        });
        info = await fs.lstat(current);
      }
      if (info.isSymbolicLink() || !info.isDirectory() || !owned(info)) throw new ModelIntegrityError(`Unsafe model cache namespace: ${current}`);
      await fs.chmod(current, 0o700);
    }
    const entry = path.join(root, request);
    try {
      const info = await fs.lstat(entry);
      if (info.isSymbolicLink() || !info.isFile() || !owned(info)) throw new ModelIntegrityError(`Unsafe cached model file: ${entry}`);
      await fs.chmod(entry, 0o600);
    } catch (error) { if (!missing(error)) throw error; if (!createParents) return undefined; }
    return entry;
  }
  async function secureAuxiliary(entry: string): Promise<void> {
    await secureDirectory();
    try {
      const info = await fs.lstat(entry);
      if (info.isSymbolicLink() || !info.isFile() || !owned(info)) throw new ModelIntegrityError(`Unsafe model installation file: ${entry}`);
      await fs.chmod(entry, 0o600);
    } catch (error) { if (!missing(error)) throw error; }
  }
  const coordinationKey = async (request: string) => {
    const hash = await createNodeHash();
    hash.update(new TextEncoder().encode(request));
    return hash.digest('hex');
  };
  await secureDirectory();
  return {
    path: root,
    async stage(request) {
      return nodeDownloadStage({
        directory: root, key: await coordinationKey(request), validate: secureDirectory, validateFile: secureAuxiliary,
        async destination() {
          const entry = await secureEntry(request, true);
          if (!entry) throw new ModelIntegrityError(`Unpinned model staging destination: ${request}`);
          return entry;
        },
      });
    },
    async withLock(request, operation, signal) {
      return withNodeInstallLock({ directory: root, key: await coordinationKey(request), validate: secureDirectory, validateFile: secureAuxiliary }, operation, signal);
    },
    async match(request) {
      const entry = await secureEntry(request, false);
      if (!entry) return undefined;
      const { constants }: { constants: typeof NodeFileConstants } = await import(`${protocol}fs`);
      const handle = await fs.open(entry, constants.O_RDONLY | constants.O_NOFOLLOW);
      const info = await handle.stat();
      const { Readable }: typeof NodeStream = await import(`${protocol}stream`);
      const stream = Readable.toWeb(handle.createReadStream()) as ReadableStream<Uint8Array<ArrayBuffer>>;
      return { body: stream, headers: new Headers({ 'content-length': String(info.size) }), status: 200, filePath: entry };
    },
    async put(request, response) {
      const entry = await secureEntry(request, true);
      if (!entry) throw new ModelIntegrityError(`Unpinned filesystem cache entry: ${request}`);
      const temporary = `${entry}.${globalThis.crypto.randomUUID()}.tmp`;
      const handle = await fs.open(temporary, 'wx', 0o600);
      const reader = response.body?.getReader();
      try {
        if (!reader) throw new ModelIntegrityError('Cannot cache an empty resource body');
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          let offset = 0;
          while (offset < value.length) { const { bytesWritten } = await handle.write(value, offset); offset += bytesWritten; }
        }
        await handle.sync();
        await handle.close();
        await secureEntry(request, true);
        await fs.rename(temporary, entry);
        await secureEntry(request, false);
      } finally {
        await reader?.cancel().catch(() => undefined);
        reader?.releaseLock();
        await handle.close().catch(() => undefined);
        await fs.unlink(temporary).catch((error: unknown) => { if (!missing(error)) throw error; });
      }
    },
    async delete(request) {
      const entry = await secureEntry(request, false);
      if (entry) await fs.unlink(entry);
    },
  };
}

/** Installs one pinned-model integrity policy into Transformers.js's global native cache/fetch hooks. */
export async function installVerifiedCache(options: VerifiedCacheOptions = {}): Promise<VerifiedCacheInstallation> {
  if (installed) throw new NekoError('Only one Neko runtime owner may be active; await its disposal before creating another', 'create', 'RUNTIME_BUSY');
  if (env.version !== '4.2.0') throw new Error('The verified model cache requires Transformers.js 4.2.0');
  const model = getRegisteredModelProfile(options.profile, options.model);
  const selectedFiles = model.files;
  const modelFileUrl = (name: ModelFileName) => `${model.baseUrl}${name}`;
  const modelSource = captureModelSource(options.modelSource);
  const downloadConcurrency = options.downloadConcurrency ?? 3;
  if (!Number.isSafeInteger(downloadConcurrency) || downloadConcurrency < 1 || downloadConcurrency > 16) throw new RangeError('downloadConcurrency must be an integer from 1 through 16');
  if (options.resumeDownloads !== undefined && typeof options.resumeDownloads !== 'boolean') throw new TypeError('resumeDownloads must be a boolean');
  installed = true;
  let backing: NativeCache;
  let signalContext: AsyncLocalStorage<{ signal: AbortSignal | undefined }> | undefined;
  try {
    backing = await nativeCache(options);
    if (isNode) {
      // Async context is Node-only; browser operations pass their signals explicitly.
      const protocol = 'node:';
      const { AsyncLocalStorage } = await import(`${protocol}async_hooks`);
      signalContext = new AsyncLocalStorage();
    }
  } catch (error) { installed = false; throw error; }
  const previous = {
    fetch: env.fetch, customCache: env.customCache, useCustomCache: env.useCustomCache,
    useBrowserCache: env.useBrowserCache, useFSCache: env.useFSCache, useFS: env.useFS,
    allowLocalModels: env.allowLocalModels, allowRemoteModels: env.allowRemoteModels,
    remoteHost: env.remoteHost, remotePathTemplate: env.remotePathTemplate, cacheDir: env.cacheDir,
  };
  let activeSignal: AbortSignal | undefined;
  let browserSignalDepth = 0;
  const currentSignal = () => signalContext?.getStore()?.signal ?? activeSignal;
  const verifiedCache = {
    async match(request: string, signal = currentSignal()): Promise<Response | string | undefined> {
      signal?.throwIfAborted();
      const name = fileName(request, model);
      if (!name && !isRuntimeAsset(request)) return undefined;
      if (!name) await authorizeNetwork(options.policy, new URL(request), 'runtime', !!options.localFilesOnly, undefined, signal);
      let response: CachedResponse | undefined;
      try {
        response = await backing.match(nativeKey(request, model, name));
        if (!response || !name) return response as Response | undefined;
        if (isNode && name.startsWith('onnx/') && response.filePath) {
          await hashCachedFile(response, name, model.allFiles[name], options, signal);
          return response.filePath;
        }
        return await readVerified(response, name, model.allFiles[name], options, 'verify', signal);
      } catch (error) {
        if (error instanceof ModelIntegrityError) {
          await response?.body?.cancel(error).catch(() => undefined);
          return rejectedResponse(error);
        }
        throw error;
      }
    },
    async put(request: string, response: Response, signal = currentSignal()): Promise<void> {
      const name = fileName(request, model);
      if (!name && !isRuntimeAsset(request)) throw new ModelIntegrityError(`Unpinned cache resource: ${request}`);
      if (!name) await authorizeNetwork(options.policy, new URL(request), 'runtime', !!options.localFilesOnly, undefined, signal);
      // Validate before the native writer opens a temporary file: 4.2.0 cannot reliably clean up an early stream error.
      const verified = name && (verifiedResponses.get(response) !== model.allFiles[name] || response.bodyUsed) ? await readVerified(response, name, model.allFiles[name], options, 'verify', signal) : response;
      const key = nativeKey(request, model, name);
      if (name) await backing.withLock(key, () => backing.put(key, verified), signal);
      else await backing.put(key, verified);
    },
  };
  const fetchRemote = async (request: string, name: ModelFileName | undefined, init: RequestInit | undefined, headers: Headers | undefined, signal: AbortSignal): Promise<{ response: Response; destination: string }> => {
    const modelMirror = name && modelSource ? new URL(name, modelSource.baseUrl) : undefined;
    const canonical = new URL(request);
    let destination = modelMirror ?? canonical;
    let previousOrigin = canonical.origin;
    let response: Response;
    for (let redirects = 0; ; redirects++) {
      await authorizeNetwork(options.policy, destination, name ? 'model' : 'runtime', !!options.localFilesOnly, name ? { modelFiles: selectedFiles, modelSource: new URL(modelFileUrl(name)), modelMirror } : undefined, signal);
      if (headers && destination.origin !== previousOrigin) {
        headers.delete('authorization');
        headers.delete('cookie');
        headers.delete('proxy-authorization');
      }
      previousOrigin = destination.origin;
      response = await previous.fetch(destination.href, { ...init, headers, signal, redirect: 'manual' });
      if (response.type === 'opaqueredirect' || response.status === 0) {
        await response.body?.cancel();
        throw new NekoError('Browser concealed a model redirect destination; use an explicitly configured modelSource mirror or verified offline cache. Automatic redirect following is denied.', 'cache', 'POLICY_DENIED');
      }
      if (![301, 302, 303, 307, 308].includes(response.status)) break;
      const location = response.headers.get('location');
      await response.body?.cancel();
      if (!location || redirects >= 10) throw new ModelIntegrityError('Model redirect limit or missing destination');
      destination = new URL(location, destination);
      if (!['http:', 'https:'].includes(destination.protocol) || destination.username || destination.password) throw new ModelIntegrityError('Unsafe model redirect');
    }
    return { response, destination: destination.href };
  };
  const downloadFile = async (name: ModelFileName, signal: AbortSignal, init?: RequestInit): Promise<void> => {
    const request = modelFileUrl(name);
    const key = nativeKey(request, model, name);
    const spec = model.allFiles[name];
    const stage = await backing.stage(key);
    await downloadResumable({
      source: modelSource ? new URL(name, modelSource.baseUrl).href : request,
      size: spec.size, sha256: spec.sha256, stage, signal, resume: options.resumeDownloads ?? true,
      async fetch(resumeHeaders) {
        const headers = new Headers(init?.headers);
        for (const [header, value] of resumeHeaders) headers.set(header, value);
        return fetchRemote(request, name, { ...init, method: 'GET' }, headers, signal);
      },
      onProgress(event) { options.onProgress?.({ file: name, ...event }); },
      integrityError(detail) { return integrity(name, detail); },
    });
  };
  const fetchVerified = async (input: string | URL, init?: RequestInit): Promise<Response> => {
    const ambientSignal = currentSignal();
    ambientSignal?.throwIfAborted();
    const request = String(input);
    const name = fileName(request, model);
    if (!name && !isRuntimeAsset(request)) throw new ModelIntegrityError(`Unpinned resource URL: ${request}`);
    const method = init?.method?.toUpperCase() ?? 'GET';
    const requestSignal = ambientSignal && init?.signal && ambientSignal !== init.signal
      ? AbortSignal.any([ambientSignal, init.signal]) : init?.signal ?? ambientSignal;
    if (!name && (method === 'GET' || method === 'HEAD')) {
      init?.signal?.throwIfAborted();
      const cached = await verifiedCache.match(request, requestSignal);
      if (cached instanceof Response) {
        init?.signal?.throwIfAborted();
        if (method === 'HEAD') { await cached.body?.cancel(); return new Response(null, { status: cached.status, headers: cached.headers }); }
        return cached;
      }
    }
    const headers = init?.headers === undefined ? undefined : new Headers(init.headers);
    const oneByteRange = method === 'GET' && headers?.get('range') === 'bytes=0-0';
    const metadata = name !== undefined && (method === 'HEAD' || oneByteRange);
    if (metadata) {
      init?.signal?.throwIfAborted();
      const cached = await verifiedCache.match(request, requestSignal);
      if (cached !== undefined) {
        // A corrupt cache match is an errored Response, not a cache miss; never turn it into successful metadata.
        if (typeof cached !== 'string' && verifiedResponses.get(cached) !== model.allFiles[name]) {
          await cached.arrayBuffer();
          throw integrity(name, 'unverified metadata cache entry');
        }
        init?.signal?.throwIfAborted();
        const spec = model.allFiles[name];
        const headers = typeof cached === 'string' ? new Headers({ 'content-type': 'application/octet-stream' }) : new Headers(cached.headers);
        if (method === 'HEAD') {
          if (typeof cached !== 'string') await cached.body?.cancel();
          headers.set('content-length', String(spec.size));
          return new Response(null, { status: 200, headers });
        }
        const firstByte = new Uint8Array(1);
        if (typeof cached === 'string') {
          // Browser builds cannot statically import Node-only modules; this path is a fully hashed native cache file.
          const protocol = 'node:';
          const fs: typeof NodeFs = await import(`${protocol}fs/promises`);
          const { constants }: { constants: typeof NodeFileConstants } = await import(`${protocol}fs`);
          const handle = await fs.open(cached, constants.O_RDONLY | constants.O_NOFOLLOW);
          try {
            const { bytesRead } = await handle.read(firstByte, 0, 1, 0);
            if (bytesRead !== 1) throw integrity(name, 'cached file changed during metadata read');
          } finally { await handle.close(); }
        } else {
          const reader = cached.body!.getReader();
          try {
            const { value, done } = await reader.read();
            if (done || !value?.byteLength) throw integrity(name, 'cached file changed during metadata read');
            firstByte[0] = value[0]!;
          } finally { await reader.cancel(); reader.releaseLock(); }
        }
        headers.set('content-length', '1');
        headers.set('content-range', `bytes 0-0/${spec.size}`);
        return new Response(firstByte, { status: 206, headers });
      }
    }
    if (options.localFilesOnly) throw new Error(`Offline cache miss: ${request}`);
    const timeout = AbortSignal.timeout(15 * 60_000);
    const signal = requestSignal ? AbortSignal.any([timeout, requestSignal]) : timeout;
    if (name && method === 'GET' && !metadata) {
      const key = nativeKey(request, model, name);
      return backing.withLock(key, async () => {
        await downloadFile(name, signal, init);
        const cached = await backing.match(key);
        if (!cached?.body) throw integrity(name, 'installed body unavailable');
        const headers = new Headers(cached.headers);
        headers.set('content-length', String(model.allFiles[name].size));
        const verified = new Response(cached.body, { status: 200, headers });
        verifiedResponses.set(verified, model.allFiles[name]);
        return verified;
      }, signal);
    }
    const { response } = await fetchRemote(request, name, init, headers, signal);
    if (!name || !response.ok) return response;
    // Metadata requests have no full body to hash. Remote metadata is not a verified cache entry.
    if (metadata) {
      if (oneByteRange && response.status === 206 && response.headers.get('content-range') !== `bytes 0-0/${model.allFiles[name].size}`) {
        await response.body?.cancel();
        throw integrity(name, 'metadata total size');
      }
      return response;
    }
    return readVerified(response, name, model.allFiles[name], options, 'download', signal);
  };
  Object.assign(env, {
    customCache: verifiedCache, useCustomCache: true, useBrowserCache: false, useFSCache: false,
    // 4.2.0 requires local-model access for local_files_only cache hits. Disable its raw filesystem bypass.
    allowLocalModels: true, useFS: false, allowRemoteModels: !options.localFilesOnly,
    remoteHost: 'https://huggingface.co/', remotePathTemplate: '{model}/resolve/{revision}/',
    fetch: fetchVerified,
  });
  if (isNode) env.cacheDir = backing.path;
  let restored = false;
  const bundleBacking: BundleBacking = {
    path: backing.path,
    async match(name) {
      if (restored) throw new Error('Verified cache installation has been restored');
      return (await backing.match(nativeKey(modelFileUrl(name), model, name)))?.body ?? undefined;
    },
    async put(name, body) {
      if (restored) throw new Error('Verified cache installation has been restored');
      const key = nativeKey(modelFileUrl(name), model, name);
      const response = new Response(body, { headers: { 'content-length': String(model.allFiles[name].size) } });
      if (isNode || typeof navigator !== 'undefined' && navigator.locks) await backing.withLock(key, () => backing.put(key, response));
      else await backing.put(key, response);
    },
    async delete(name) {
      const key = nativeKey(modelFileUrl(name), model, name);
      if (isNode || typeof navigator !== 'undefined' && navigator.locks) await backing.withLock(key, () => backing.delete(key));
      else await backing.delete(key);
    },
  };
  return {
    restore() {
      if (restored) return;
      restored = true;
      Object.assign(env, previous);
      installed = false;
    },
    async withSignal(signal, operation) {
      if (restored) throw new Error('Verified cache installation has been restored');
      signal?.throwIfAborted();
      if (signalContext) return signalContext.run({ signal }, operation);
      // Browsers lack async-local context. Nested loader scopes share the same signal;
      // independently overlapping scopes must not inherit each other's cancellation.
      if (browserSignalDepth && activeSignal !== signal) throw new NekoError('Overlapping browser cache signal scopes are not supported', 'cache', 'RUNTIME_BUSY');
      const previousSignal = activeSignal;
      activeSignal = signal;
      browserSignalDepth++;
      try { return await operation(); }
      finally { browserSignalDepth--; activeSignal = previousSignal; }
    },
    exportBundle(signal) {
      if (restored) throw new Error('Verified cache installation has been restored');
      return exportModelBundle(model, bundleBacking, signal);
    },
    async importBundle(source, signal) {
      if (restored) throw new Error('Verified cache installation has been restored');
      await importModelBundle(source, model, bundleBacking, signal);
    },
    async diagnostics(signal) {
      if (restored) throw new Error('Verified cache installation has been restored');
      signal?.throwIfAborted();
      const storage: ModelCacheDiagnostics['storage'] = { usage: null, quota: null, available: null, persisted: null, source: 'unavailable' };
      if (!isNode && typeof navigator !== 'undefined' && navigator.storage) {
        const estimate = await navigator.storage.estimate?.();
        signal?.throwIfAborted();
        if (typeof estimate?.usage === 'number' && Number.isFinite(estimate.usage) && estimate.usage >= 0) storage.usage = estimate.usage;
        if (typeof estimate?.quota === 'number' && Number.isFinite(estimate.quota) && estimate.quota >= 0) storage.quota = estimate.quota;
        if (storage.usage !== null && storage.quota !== null) storage.available = Math.max(0, storage.quota - storage.usage);
        if (estimate) storage.source = 'browser-storage-estimate';
        if (navigator.storage.persisted) storage.persisted = await navigator.storage.persisted();
        signal?.throwIfAborted();
      }
      return { model: model.id, revision: model.revision, profile: model.profile, backend: isNode ? 'filesystem' : 'cache-storage', path: backing.path,
        requiredBytes: Object.values(selectedFiles).reduce((total, file) => total + file.size, 0), storage };
    },
    async status(signal) {
      if (restored) throw new Error('Verified cache installation has been restored');
      const files: ModelCacheStatus['files'] = [];
      for (const name of Object.keys(selectedFiles) as ModelFileName[]) {
        signal?.throwIfAborted();
        const response = await verifiedCache.match(modelFileUrl(name), signal);
        if (response instanceof Response) {
          if (!response.headers.has('content-length')) await response.arrayBuffer();
          await response.body?.cancel();
        }
        files.push({ name, present: response !== undefined, verified: response !== undefined, bytes: response === undefined ? 0 : model.allFiles[name].size });
      }
      return { downloaded: files.every((file) => file.present), verified: files.every((file) => file.verified), bytes: files.reduce((total, file) => total + file.bytes, 0), totalBytes: Object.values(selectedFiles).reduce((total, file) => total + file.size, 0), path: backing.path, files };
    },
    async clear(signal) {
      if (restored) throw new Error('Verified cache installation has been restored');
      for (const name of Object.keys(model.allFiles) as ModelFileName[]) {
        signal?.throwIfAborted();
        const key = nativeKey(modelFileUrl(name), model, name);
        const remove = async () => {
          await backing.delete(key);
          if (isNode || typeof indexedDB !== 'undefined') await (await backing.stage(key)).reset();
        };
        if (isNode || typeof navigator !== 'undefined' && navigator.locks) await backing.withLock(key, remove, signal);
        else await remove();
      }
    },
    async prefetch(signal?: AbortSignal, files: readonly ModelFileName[] = Object.keys(selectedFiles) as ModelFileName[]) {
      if (restored) throw new Error('Verified cache installation has been restored');
      if (files.some((name) => !Object.hasOwn(selectedFiles, name))) throw new ModelIntegrityError('Cannot prefetch an unpinned selected-model file');
      const controller = new AbortController();
      const operationSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
      let next = 0;
      let failed = false;
      let failure: unknown;
      const worker = async () => {
        try {
          while (next < files.length) {
            operationSignal.throwIfAborted();
            const name = files[next++]!;
            const request = modelFileUrl(name);
            const key = nativeKey(request, model, name);
            const cached = await verifiedCache.match(request, operationSignal);
            if (cached !== undefined) {
              if (cached instanceof Response && !cached.headers.has('content-length')) await cached.arrayBuffer();
              else if (typeof cached !== 'string') await cached.body?.cancel();
              continue;
            }
            if (options.localFilesOnly) throw new Error(`Offline cache miss: ${request}`);
            const downloadSignal = AbortSignal.any([operationSignal, AbortSignal.timeout(15 * 60_000)]);
            await backing.withLock(key, async () => {
              // Another realm/process may have completed this file while we waited.
              const installed = await verifiedCache.match(request, downloadSignal);
              if (installed !== undefined) {
                if (installed instanceof Response && !installed.headers.has('content-length')) await installed.arrayBuffer();
                else if (typeof installed !== 'string') await installed.body?.cancel();
                return;
              }
              await downloadFile(name, downloadSignal);
            }, downloadSignal);
          }
        } catch (error) {
          if (!failed) { failed = true; failure = error; controller.abort(error); }
        }
      };
      await Promise.all(Array.from({ length: Math.min(downloadConcurrency, files.length) }, worker));
      if (failed) throw failure;
    },
  };
}
