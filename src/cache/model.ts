import { env } from '@huggingface/transformers';
import { MODEL_BASE_URL, MODEL_FILES, MODEL_ID, MODEL_REVISION, modelFileUrl } from './manifest.js';
import type { ModelFileName } from './manifest.js';
import type * as NodeFs from 'node:fs/promises';
import type * as NodePath from 'node:path';
import type { constants as NodeFileConstants, Stats } from 'node:fs';

const isNode = typeof process !== 'undefined' && process.release?.name === 'node';

export class ModelIntegrityError extends Error { override name = 'ModelIntegrityError'; }
export interface CacheProgress { file: ModelFileName; loaded: number; total: number; phase: 'download' | 'verify'; }
export interface VerifiedCacheOptions {
  /** Transformers.js native filesystem cache directory; browsers use its native Cache API cache. */
  cacheDir?: string;
  localFilesOnly?: boolean;
  onProgress?: (event: CacheProgress) => void;
}
export interface VerifiedCacheInstallation { restore(): void; prefetch(signal?: AbortSignal): Promise<void>; }

type ResourceSpec = { readonly size: number; readonly sha256: string };
type CachedResponse = Pick<Response, 'body' | 'headers' | 'status'> & { filePath?: string };
type NativeCache = {
  match(request: string): Promise<CachedResponse | undefined>;
  put(request: string, response: Response): Promise<void>;
};
type NodeHash = { update(bytes: Uint8Array): NodeHash; digest(encoding: 'hex'): string };
let nodeCrypto: Promise<{ createHash(algorithm: string): NodeHash }> | undefined;
let installed = false;
const verifiedResponses = new WeakMap<Response, ModelFileName>();

function fileName(request: string): ModelFileName | undefined {
  if (!request.startsWith(MODEL_BASE_URL)) return undefined;
  const name = request.slice(MODEL_BASE_URL.length);
  if (!Object.hasOwn(MODEL_FILES, name)) throw new ModelIntegrityError(`Unpinned model resource: ${request}`);
  return name as ModelFileName;
}
function nativeKey(request: string, name?: ModelFileName): string {
  return isNode && name ? `${MODEL_ID}/${MODEL_REVISION}/${name}` : request;
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
async function readVerified(response: CachedResponse, name: ModelFileName, options: VerifiedCacheOptions, phase: CacheProgress['phase'], signal?: AbortSignal): Promise<Response> {
  const spec = MODEL_FILES[name];
  checkLength(response, name, spec);
  signal?.throwIfAborted();
  const bytes = new Uint8Array(spec.size);
  const hash = isNode ? await createNodeHash() : undefined;
  const reader = response.body!.getReader();
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
    verifiedResponses.set(verified, name);
    return verified;
  } catch (error) {
    await reader.cancel(error).catch(() => undefined);
    throw error;
  } finally { reader.releaseLock(); }
}
async function hashCachedFile(response: CachedResponse, name: ModelFileName, options: VerifiedCacheOptions): Promise<void> {
  const spec = MODEL_FILES[name];
  checkLength(response, name, spec);
  const hash = await createNodeHash();
  const reader = response.body!.getReader();
  let loaded = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value.byteLength > spec.size - loaded) throw integrity(name, 'exceeds pinned size');
      hash.update(value);
      loaded += value.byteLength;
      options.onProgress?.({ file: name, loaded, total: spec.size, phase: 'verify' });
    }
    if (loaded !== spec.size || hash.digest('hex') !== spec.sha256) throw integrity(name, 'size or SHA-256');
  } catch (error) {
    await reader.cancel(error).catch(() => undefined);
    throw error;
  } finally { reader.releaseLock(); }
}
function rejectedResponse(error: ModelIntegrityError): Response {
  // Transformers.js swallows rejected cache matches. An errored body makes corruption a load failure, not a cache miss.
  const response = new Response(new ReadableStream({ start(controller) { controller.error(error); } }));
  // Chromium's native body reader replaces stream errors with TypeError('Failed to fetch').
  response.arrayBuffer = async () => { throw error; };
  return response;
}
async function nativeCache(options: VerifiedCacheOptions): Promise<NativeCache> {
  if (!isNode) {
    if (typeof caches === 'undefined') throw new Error('Verified model caching requires the browser Cache API in a secure context');
    return caches.open(env.cacheKey);
  }
  // Platform-only imports cannot be static: browsers have no Node filesystem or package source files.
  const protocol = 'node:';
  const fs: typeof NodeFs = await import(`${protocol}fs/promises`);
  const path: typeof NodePath = await import(`${protocol}path`);
  // The exact 4.2.0 package has no public FileCache export; reuse its native atomic writer rather than implement another one.
  const source = new URL('../src/utils/cache/FileCache.js', import.meta.resolve('@huggingface/transformers'));
  const { FileCache } = await import(source.href) as { FileCache: new (directory: string) => NativeCache };
  const directory = options.cacheDir ?? env.cacheDir;
  if (!directory) throw new Error('Transformers.js filesystem cache directory is unavailable');
  const root = path.resolve(directory);
  if (root === path.parse(root).root) throw new ModelIntegrityError('Cache directory cannot be a filesystem root');
  const uid = process.getuid?.();
  const missing = (error: unknown) => typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT';
  const owned = (info: Stats) => uid === undefined || info.uid === uid;
  async function secureDirectory(): Promise<void> {
    const ancestors: string[] = [];
    for (let current = root; current !== path.dirname(current); current = path.dirname(current)) ancestors.push(current);
    for (const current of ancestors.reverse()) {
      let info;
      try { info = await fs.lstat(current); }
      catch (error) { if (!missing(error)) throw error; await fs.mkdir(current, { mode: 0o700 }); info = await fs.lstat(current); }
      if (info.isSymbolicLink() || !info.isDirectory()) throw new ModelIntegrityError(`Unsafe cache directory: ${current}`);
    }
    const info = await fs.lstat(root);
    if (!owned(info)) throw new ModelIntegrityError('Cache directory must belong to the current user');
    if (await fs.realpath(root) !== root) throw new ModelIntegrityError('Cache directory changed during validation');
    await fs.chmod(root, 0o700);
  }
  async function secureEntry(request: string, createParents: boolean): Promise<string | undefined> {
    const prefix = `${MODEL_ID}/${MODEL_REVISION}/`;
    if (!request.startsWith(prefix) || !Object.hasOwn(MODEL_FILES, request.slice(prefix.length))) return undefined;
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
        await fs.mkdir(current, { mode: 0o700 });
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
  await secureDirectory();
  const cache = new FileCache(root);
  return {
    async match(request) { return await secureEntry(request, false) ? cache.match(request) : undefined; },
    async put(request, response) {
      const entry = await secureEntry(request, true);
      if (!entry) throw new ModelIntegrityError(`Unpinned filesystem cache entry: ${request}`);
      await cache.put(request, response);
      await secureEntry(request, false);
    },
  };
}

/** Installs one pinned-model integrity policy into Transformers.js's global native cache/fetch hooks. */
export async function installVerifiedCache(options: VerifiedCacheOptions = {}): Promise<VerifiedCacheInstallation> {
  if (installed) throw new Error('A verified Transformers.js cache installation is already active');
  if (env.version !== '4.2.0') throw new Error('The verified model cache requires Transformers.js 4.2.0');
  installed = true;
  let backing: NativeCache;
  try { backing = await nativeCache(options); } catch (error) { installed = false; throw error; }
  const previous = {
    fetch: env.fetch, customCache: env.customCache, useCustomCache: env.useCustomCache,
    useBrowserCache: env.useBrowserCache, useFSCache: env.useFSCache, useFS: env.useFS,
    allowLocalModels: env.allowLocalModels, allowRemoteModels: env.allowRemoteModels,
    remoteHost: env.remoteHost, remotePathTemplate: env.remotePathTemplate, cacheDir: env.cacheDir,
  };
  const verifiedCache = {
    async match(request: string): Promise<Response | string | undefined> {
      const name = fileName(request);
      if (!name && !isRuntimeAsset(request)) return undefined;
      let response: CachedResponse | undefined;
      try {
        response = await backing.match(nativeKey(request, name));
        if (!response || !name) return response as Response | undefined;
        if (isNode && name.startsWith('onnx/') && response.filePath) {
          await hashCachedFile(response, name, options);
          return response.filePath;
        }
        return await readVerified(response, name, options, 'verify');
      } catch (error) {
        if (error instanceof ModelIntegrityError) {
          await response?.body?.cancel(error).catch(() => undefined);
          return rejectedResponse(error);
        }
        throw error;
      }
    },
    async put(request: string, response: Response): Promise<void> {
      const name = fileName(request);
      if (!name && !isRuntimeAsset(request)) throw new ModelIntegrityError(`Unpinned cache resource: ${request}`);
      // Validate before the native writer opens a temporary file: 4.2.0 cannot reliably clean up an early stream error.
      const verified = name && (verifiedResponses.get(response) !== name || response.bodyUsed) ? await readVerified(response, name, options, 'verify') : response;
      await backing.put(nativeKey(request, name), verified);
    },
  };
  const fetchVerified = async (input: string | URL, init?: RequestInit): Promise<Response> => {
    const request = String(input);
    const name = fileName(request);
    if (!name && !isRuntimeAsset(request)) throw new ModelIntegrityError(`Unpinned resource URL: ${request}`);
    const method = init?.method?.toUpperCase() ?? 'GET';
    const oneByteRange = method === 'GET' && new Headers(init?.headers).get('range') === 'bytes=0-0';
    const metadata = name !== undefined && (method === 'HEAD' || oneByteRange);
    if (metadata) {
      init?.signal?.throwIfAborted();
      const cached = await verifiedCache.match(request);
      if (cached !== undefined) {
        // A corrupt cache match is an errored Response, not a cache miss; never turn it into successful metadata.
        if (typeof cached !== 'string' && verifiedResponses.get(cached) !== name) {
          await cached.arrayBuffer();
          throw integrity(name, 'unverified metadata cache entry');
        }
        init?.signal?.throwIfAborted();
        const spec = MODEL_FILES[name];
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
    const signal = init?.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
    const response = await previous.fetch(input, { ...init, signal });
    if (!name || !response.ok) return response;
    // Metadata requests have no full body to hash. Remote metadata is not a verified cache entry.
    if (metadata) {
      if (oneByteRange && response.status === 206 && response.headers.get('content-range') !== `bytes 0-0/${MODEL_FILES[name].size}`) {
        await response.body?.cancel();
        throw integrity(name, 'metadata total size');
      }
      return response;
    }
    return readVerified(response, name, options, 'download', signal);
  };
  Object.assign(env, {
    customCache: verifiedCache, useCustomCache: true, useBrowserCache: false, useFSCache: false,
    // 4.2.0 requires local-model access for local_files_only cache hits. Disable its raw filesystem bypass.
    allowLocalModels: true, useFS: false, allowRemoteModels: !options.localFilesOnly,
    remoteHost: 'https://huggingface.co/', remotePathTemplate: '{model}/resolve/{revision}/',
    fetch: fetchVerified,
  });
  if (options.cacheDir && isNode) env.cacheDir = options.cacheDir;
  let restored = false;
  return {
    restore() {
      if (restored) return;
      restored = true;
      Object.assign(env, previous);
      installed = false;
    },
    async prefetch(signal?: AbortSignal) {
      if (restored) throw new Error('Verified cache installation has been restored');
      for (const name of Object.keys(MODEL_FILES) as ModelFileName[]) {
        signal?.throwIfAborted();
        const request = modelFileUrl(name);
        const cached = await verifiedCache.match(request);
        if (cached !== undefined) {
          // Reading an errored cached response must expose corruption even during prefetch.
          if (cached instanceof Response && !cached.headers.has('content-length')) await cached.arrayBuffer();
          else if (typeof cached !== 'string') await cached.body?.cancel();
          continue;
        }
        const response = await fetchVerified(request, signal ? { signal } : undefined);
        if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status} (${name})`);
        await verifiedCache.put(request, response);
      }
    },
  };
}
