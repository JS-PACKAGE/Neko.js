import type { RegisteredModelProfile } from './registry.js';
import type { ModelFileName } from './manifest.js';
import { StreamingSha256 } from './sha256.js';
import type * as Fs from 'node:fs/promises';
import type * as Stream from 'node:stream';
import type * as NodeCrypto from 'node:crypto';

export type ModelBundleSource = Blob | ReadableStream<Uint8Array>;
export interface BundleBacking {
  path: string;
  match(name: ModelFileName): Promise<ReadableStream<Uint8Array> | undefined>;
  put(name: ModelFileName, body: ReadableStream<Uint8Array>): Promise<void>;
  delete(name: ModelFileName): Promise<void>;
}
interface BundleStage extends BundleBacking { cleanup(): Promise<void>; }
const node = typeof process !== 'undefined' && process.release?.name === 'node';
const chunkSize = 64 * 1024;
const headerLimit = 64 * 1024;
type Hasher = { update(bytes: Uint8Array): void; digest(): string };
async function createHash(): Promise<Hasher> {
  if (!node) return new StreamingSha256();
  // Node crypto is absent from browser runtimes; never include it in the browser bundle.
  const protocol = 'node:';
  const crypto: typeof NodeCrypto = await import(`${protocol}crypto`);
  const hash = crypto.createHash('sha256');
  return { update(bytes) { hash.update(bytes); }, digest() { return hash.digest('hex'); } };
}
function header(model: RegisteredModelProfile) {
  return { format: 'neko-model-bundle', version: 1, model: model.id, revision: model.revision, profile: model.profile,
    files: (Object.keys(model.files) as ModelFileName[]).map((name) => ({ name, ...model.files[name]! })) };
}
async function* verifiedChunks(body: ReadableStream<Uint8Array>, name: ModelFileName, model: RegisteredModelProfile, signal?: AbortSignal): AsyncGenerator<Uint8Array> {
  const reader = body.getReader();
  const cancel = () => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener('abort', cancel, { once: true });
  const hash = await createHash();
  const spec = model.files[name]!;
  let loaded = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      signal?.throwIfAborted();
      if (done) break;
      if (!(value instanceof Uint8Array) || value.length > spec.size - loaded) throw new Error(`Invalid bundle asset length: ${name}`);
      for (let offset = 0; offset < value.length; offset += chunkSize) {
        const bytes = value.subarray(offset, Math.min(offset + chunkSize, value.length));
        hash.update(bytes); loaded += bytes.length; yield bytes;
      }
    }
    signal?.throwIfAborted();
    if (loaded !== spec.size || hash.digest() !== spec.sha256) throw new Error(`Invalid bundle asset size or SHA-256: ${name}`);
  } finally {
    signal?.removeEventListener('abort', cancel);
    await reader.cancel().catch(() => undefined); reader.releaseLock();
  }
}
function iterableStream(iterator: AsyncGenerator<Uint8Array>, cancel?: () => void): ReadableStream<Uint8Array> {
  return new ReadableStream({
    async pull(controller) {
      try { const item = await iterator.next(); if (item.done) controller.close(); else controller.enqueue(item.value); }
      catch (error) { controller.error(error); await iterator.return(undefined); }
    },
    async cancel() { cancel?.(); await iterator.return(undefined); },
  }, { highWaterMark: 0 });
}
export function exportModelBundle(model: RegisteredModelProfile, backing: BundleBacking, signal?: AbortSignal): ReadableStream<Uint8Array> {
  const cancellation = new AbortController();
  signal = signal ? AbortSignal.any([signal, cancellation.signal]) : cancellation.signal;
  async function* chunks(): AsyncGenerator<Uint8Array> {
    // Verify every asset before exposing even the header; a corrupt installation is not exportable.
    for (const name of Object.keys(model.files) as ModelFileName[]) {
      signal?.throwIfAborted();
      const body = await backing.match(name);
      if (!body) throw new Error(`Offline bundle requires an installed asset: ${name}`);
      for await (const bytes of verifiedChunks(body, name, model, signal)) { void bytes; /* Consume verification without retaining weights. */ }
    }
    const metadata = new TextEncoder().encode(JSON.stringify(header(model)));
    const length = new Uint8Array(4); new DataView(length.buffer).setUint32(0, metadata.length);
    yield length; yield metadata;
    for (const name of Object.keys(model.files) as ModelFileName[]) {
      signal?.throwIfAborted();
      const body = await backing.match(name);
      if (!body) throw new Error(`Model asset disappeared during export: ${name}`);
      for await (const bytes of verifiedChunks(body, name, model, signal)) {
        // Transferable streams clone a view's entire backing buffer, not just its visible bytes.
        yield bytes.byteLength === bytes.buffer.byteLength ? bytes : new Uint8Array(bytes);
      }
    }
  }
  return iterableStream(chunks(), () => cancellation.abort(new DOMException('Bundle export cancelled', 'AbortError')));
}
async function staging(backing: BundleBacking): Promise<BundleStage> {
  const id = globalThis.crypto.randomUUID();
  if (!node) {
    const cacheName = `neko-bundle-staging-${id}`;
    const cache = await caches.open(cacheName);
    const key = (name: string) => `https://neko.invalid/bundle/${id}/${name}`;
    return { path: cacheName, async match(name) { return (await cache.match(key(name)))?.body ?? undefined; },
      async put(name, body) { await cache.put(key(name), new Response(body)); },
      async delete(name) { await cache.delete(key(name)); }, async cleanup() { await caches.delete(cacheName); } };
  }
  // Node filesystem/stream modules cannot be imported into a browser bundle.
  const protocol = 'node:';
  const fs: typeof Fs = await import(`${protocol}fs/promises`);
  const { Readable }: typeof Stream = await import(`${protocol}stream`);
  const root = await fs.lstat(backing.path);
  if (!root.isDirectory() || root.isSymbolicLink() || process.getuid && root.uid !== process.getuid() || await fs.realpath(backing.path) !== backing.path) throw new Error('Unsafe bundle staging directory');
  const path = await fs.mkdtemp(`${backing.path}/.neko-bundle-`);
  const entry = (name: string) => `${path}/${name.replaceAll('/', '_')}`;
  return { path, async match(name) {
    const handle = await fs.open(entry(name), 'r');
    return Readable.toWeb(handle.createReadStream()) as ReadableStream<Uint8Array>;
  }, async put(name, body) {
    const handle = await fs.open(entry(name), 'wx', 0o600);
    const reader = body.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        let offset = 0;
        while (offset < value.length) { const { bytesWritten } = await handle.write(value, offset); if (!bytesWritten) throw new Error('Bundle staging write made no progress'); offset += bytesWritten; }
      }
      await handle.sync();
    } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); await handle.close(); }
  }, async delete(name) { await fs.unlink(entry(name)); }, async cleanup() { await fs.rm(path, { recursive: true, force: true }); } };
}
export async function importModelBundle(source: ModelBundleSource, model: RegisteredModelProfile, backing: BundleBacking, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted();
  const input = typeof Blob !== 'undefined' && source instanceof Blob ? source.stream() : source;
  if (!(input instanceof ReadableStream)) throw new TypeError('Model bundle must be a Blob or ReadableStream');
  const reader = input.getReader();
  const cancel = () => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener('abort', cancel, { once: true });
  let pending: Uint8Array = new Uint8Array(0), offset = 0;
  async function take(maximum: number): Promise<Uint8Array | undefined> {
    while (offset === pending.length) {
      signal?.throwIfAborted();
      const item = await reader.read();
      signal?.throwIfAborted();
      if (item.done) return undefined;
      if (!(item.value instanceof Uint8Array)) throw new TypeError('Bundle stream chunks must be Uint8Array');
      pending = item.value; offset = 0;
    }
    signal?.throwIfAborted();
    const end = Math.min(pending.length, offset + maximum);
    const bytes = pending.subarray(offset, end); offset = end; return bytes;
  }
  async function exact(length: number): Promise<Uint8Array> {
    const bytes = new Uint8Array(length);
    let loaded = 0;
    while (loaded < length) { const value = await take(length - loaded); if (!value) throw new Error('Truncated model bundle header'); bytes.set(value, loaded); loaded += value.length; }
    return bytes;
  }
  let stage: BundleStage | undefined;
  const promoted: ModelFileName[] = [];
  try {
    const sizeBytes = await exact(4);
    const size = new DataView(sizeBytes.buffer).getUint32(0);
    if (!size || size > headerLimit) throw new Error('Invalid model bundle header size');
    const actual: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(await exact(size)));
    const expected = header(model);
    if (!actual || typeof actual !== 'object' || Array.isArray(actual)) throw new Error('Invalid model bundle header');
    const value = actual as typeof expected;
    if (Object.keys(value).sort().join(',') !== Object.keys(expected).sort().join(',') || value.format !== expected.format || value.version !== expected.version || value.model !== expected.model || value.revision !== expected.revision || value.profile !== expected.profile || !Array.isArray(value.files) || value.files.length !== expected.files.length) throw new Error('Model bundle identity or profile mismatch');
    for (let i = 0; i < expected.files.length; i++) {
      const file = value.files[i], pinned = expected.files[i]!;
      if (!file || Object.keys(file).sort().join(',') !== 'name,sha256,size' || file.name !== pinned.name || file.size !== pinned.size || file.sha256 !== pinned.sha256) throw new Error('Model bundle contains an unpinned path, length, or digest');
    }
    stage = await staging(backing);
    for (const file of expected.files) {
      const hash = await createHash();
      let remaining = file.size;
      const body = new ReadableStream<Uint8Array>({ async pull(controller) {
        try {
          signal?.throwIfAborted();
          if (!remaining) { if (hash.digest() !== file.sha256) throw new Error(`Invalid bundle SHA-256: ${file.name}`); controller.close(); return; }
          const bytes = await take(Math.min(remaining, chunkSize));
          if (!bytes) throw new Error(`Truncated model bundle asset: ${file.name}`);
          hash.update(bytes); remaining -= bytes.length; controller.enqueue(bytes);
        } catch (error) { controller.error(error); }
      } }, { highWaterMark: 0 });
      await stage.put(file.name, body);
    }
    if (await take(1)) throw new Error('Trailing model bundle bytes');
    // Validate ALL existing selected entries before mutation; preserve them rather than overwrite.
    const missing: ModelFileName[] = [];
    for (const file of expected.files) {
      const body = await backing.match(file.name);
      if (!body) missing.push(file.name);
      else for await (const bytes of verifiedChunks(body, file.name, model, signal)) { void bytes; /* Existing integrity failures are not replacement opportunities. */ }
    }
    signal?.throwIfAborted();
    for (const name of missing) {
      signal?.throwIfAborted();
      const body = (await stage.match(name))!;
      // Track before writing: native cache writers may report an error after atomic promotion.
      promoted.push(name);
      await backing.put(name, iterableStream(verifiedChunks(body, name, model, signal)));
    }
    signal?.throwIfAborted();
  } catch (error) {
    // Only entries this import added are rolled back; preexisting/unrelated assets are untouched.
    for (const name of promoted.reverse()) await backing.delete(name);
    throw error;
  } finally {
    signal?.removeEventListener('abort', cancel);
    await reader.cancel().catch(() => undefined); reader.releaseLock();
    await stage?.cleanup();
  }
}
