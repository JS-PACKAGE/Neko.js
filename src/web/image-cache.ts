import type { DecodedImage } from './image.js';

export interface ImagePreprocessCacheOptions { maxBytes?: number; maxEntries?: number; }
export interface ImagePreprocessCacheDiagnostics {
  scope: 'normalized-pixels';
  entries: number;
  bytes: number;
  maxBytes: number;
  maxEntries: number;
  hits: number;
  misses: number;
  evictions: number;
}
interface CachedPixels { image: DecodedImage; pixelDigest: string; }
interface CacheState {
  entries: Map<string, CachedPixels>;
  bytes: number;
  maxBytes: number;
  maxEntries: number;
  hits: number;
  misses: number;
  evictions: number;
}
const states = new WeakMap<ImagePreprocessCache, CacheState>();

export class ImagePreprocessCache {
  constructor(options: ImagePreprocessCacheOptions = {}) {
    const maxBytes = options.maxBytes ?? 32 * 1024 * 1024; const maxEntries = options.maxEntries ?? 128;
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 0 || maxBytes > 256 * 1024 * 1024) throw new RangeError('Image cache maxBytes must be between 0 and 268435456');
    if (!Number.isSafeInteger(maxEntries) || maxEntries < 0 || maxEntries > 4096) throw new RangeError('Image cache maxEntries must be between 0 and 4096');
    states.set(this, { entries: new Map(), bytes: 0, maxBytes, maxEntries, hits: 0, misses: 0, evictions: 0 });
  }
  clear(): void { const state = states.get(this)!; state.entries.clear(); state.bytes = 0; }
  diagnostics(): ImagePreprocessCacheDiagnostics {
    const { entries, bytes, maxBytes, maxEntries, hits, misses, evictions } = states.get(this)!;
    return { scope: 'normalized-pixels', entries: entries.size, bytes, maxBytes, maxEntries, hits, misses, evictions };
  }
}

export function cachedImage(cache: ImagePreprocessCache | undefined, key: string): CachedPixels | undefined {
  if (!cache) return undefined;
  const state = states.get(cache);
  if (!state) throw new TypeError('Image cache must be an ImagePreprocessCache');
  const entry = state.entries.get(key);
  if (!entry) { state.misses++; return undefined; }
  state.hits++; state.entries.delete(key); state.entries.set(key, entry);
  // Consumers own mutable pixel buffers; never expose the retained cache storage.
  return { image: { ...entry.image, data: new Uint8Array(entry.image.data) }, pixelDigest: entry.pixelDigest };
}

export function cacheImage(cache: ImagePreprocessCache | undefined, key: string, image: DecodedImage, pixelDigest: string): void {
  if (!cache) return;
  const state = states.get(cache);
  if (!state) throw new TypeError('Image cache must be an ImagePreprocessCache');
  if (!state.maxEntries || image.data.byteLength > state.maxBytes) return;
  const previous = state.entries.get(key);
  if (previous) { state.bytes -= previous.image.data.byteLength; state.entries.delete(key); }
  while (state.entries.size >= state.maxEntries || state.bytes + image.data.byteLength > state.maxBytes) {
    const oldest = state.entries.keys().next().value;
    if (oldest === undefined) break;
    const evicted = state.entries.get(oldest)!;
    state.bytes -= evicted.image.data.byteLength; state.entries.delete(oldest); state.evictions++;
  }
  const owned = { ...image, data: new Uint8Array(image.data) };
  state.entries.set(key, { image: owned, pixelDigest }); state.bytes += owned.data.byteLength;
}
