import { DynamicCache, Tensor } from '@huggingface/transformers';
import { hashBytes, hashValue } from '../web/source.js';
import type { GenerationStateHandle, ReuseCacheLimits, ReuseCacheInfo } from '../types.js';

export interface GenerationStateSnapshot { tokens: bigint[]; compatibility: string; cache: DynamicCache; bytes: number; }
interface RetainedState { snapshots: GenerationStateSnapshot[]; bytes: number; }
interface Feature { tensor: Tensor; bytes: number; }
function tensorBytes(tensor: Tensor): number { const data = tensor.data; if (!ArrayBuffer.isView(data)) throw new TypeError('Reuse requires numeric tensor storage'); return data.byteLength; }
/** GPU data must be downloaded before cloning; Tensor.clone() alone cannot clone GPU storage. */
export async function ownedTensor(tensor: Tensor): Promise<Tensor> {
  if (tensor.location !== 'cpu') await tensor.ort_tensor.getData(true);
  return tensor.clone();
}
export async function inputVisionIdentity(inputs: Record<string, unknown>): Promise<string> {
  const pixels = inputs.pixel_values; const grid = inputs.image_grid_thw;
  if (!(pixels instanceof Tensor)) return 'text';
  if (!(grid instanceof Tensor) || !ArrayBuffer.isView(pixels.data)) throw new TypeError('Invalid vision inputs for reuse');
  const data = pixels.data;
  return hashValue({ pixels: await hashBytes(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)), type: pixels.type, dims: pixels.dims, grid: Array.from(grid.data, String) });
}
function disposeTensors(cache: DynamicCache): void { for (const tensor of new Set(Object.values(cache))) tensor.dispose(); }
function limit(value: number | undefined, fallback: number): number { const result = value ?? fallback; if (!Number.isSafeInteger(result) || result < 1) throw new RangeError('Reuse limits must be positive safe integers'); return result; }
const PREFIX_ENTRIES = 4;

/** Engine-local, bounded caches. Handles never expose tensors; each continuation owns a full hybrid-cache clone. */
export class InferenceReuseCache {
  private readonly states = new Map<string, RetainedState>();
  private readonly features = new Map<string, Feature>();
  private readonly limits: Required<ReuseCacheLimits>;
  private readonly owner = globalThis.crypto.randomUUID();
  private stateBytes = 0;
  private visionBytes = 0;
  private hits = 0;
  private misses = 0;
  private evictions = 0;
  private readonly prefixes = new Map<string, GenerationStateSnapshot>();
  private prefixBytes = 0;
  private prefixHits = 0;
  private prefixMisses = 0;
  constructor(limits: ReuseCacheLimits = {}) {
    this.limits = { stateEntries: limit(limits.stateEntries, 4), stateBytes: limit(limits.stateBytes, 512 * 1024 * 1024), visionEntries: limit(limits.visionEntries, 8), visionBytes: limit(limits.visionBytes, 64 * 1024 * 1024) };
  }
  info(): ReuseCacheInfo { return { stateEntries: this.states.size, stateBytes: this.stateBytes, visionEntries: this.features.size, visionBytes: this.visionBytes, visionEncoderHits: this.hits, visionEncoderMisses: this.misses, prefixEntries: this.prefixes.size, prefixBytes: this.prefixBytes, prefixHits: this.prefixHits, prefixMisses: this.prefixMisses, evictions: this.evictions, limits: { ...this.limits } }; }
  checkout(handle: GenerationStateHandle, tokens: readonly bigint[], compatibility: string): { cache: DynamicCache; tokens: number } {
    const state = this.states.get(handle.id);
    if (!state) throw new TypeError('Generation state is not owned by this engine, has been released, or was evicted');
    let snapshot: GenerationStateSnapshot | undefined;
    for (const entry of state.snapshots) {
      if (entry.compatibility === compatibility && entry.tokens.length < tokens.length && (!snapshot || entry.tokens.length > snapshot.tokens.length) && entry.tokens.every((token, index) => token === tokens[index])) snapshot = entry;
    }
    if (!snapshot) throw new TypeError('Generation state requires the same model/schema/images and an exact extending token prefix');
    this.states.delete(handle.id); this.states.set(handle.id, state);
    const entries: Record<string, Tensor> = {};
    try { for (const [name, tensor] of Object.entries(snapshot.cache)) entries[name] = tensor.clone(); return { cache: new DynamicCache(entries), tokens: snapshot.tokens.length }; }
    catch (error) { for (const tensor of Object.values(entries)) tensor.dispose(); throw error; }
  }
  async snapshot(cache: DynamicCache, tokens: bigint[], compatibility: string): Promise<GenerationStateSnapshot> {
    if (cache.get_seq_length() !== tokens.length) throw new Error('Decoder cache length does not match the processed sequence prefix');
    const entries: Record<string, Tensor> = {};
    let bytes = 0;
    try {
      for (const [name, tensor] of Object.entries(cache)) { entries[name] = await ownedTensor(tensor); bytes += tensorBytes(entries[name]!); }
      if (!Object.keys(entries).some((name) => name.startsWith('past_recurrent.')) || !Object.keys(entries).some((name) => name.startsWith('past_conv.'))) throw new Error('Qwen3.5 reusable state is missing recurrent or convolution tensors');
      if (bytes > this.limits.stateBytes) throw new RangeError('Generation state exceeds the authorized state cache byte limit');
      return { tokens, compatibility, cache: new DynamicCache(entries), bytes };
    } catch (error) { for (const tensor of Object.values(entries)) tensor.dispose(); throw error; }
  }
  commitState(state: GenerationStateSnapshot, checkpoint?: GenerationStateSnapshot): GenerationStateHandle {
    const snapshots = checkpoint ? [state, checkpoint] : [state];
    const bytes = snapshots.reduce((total, entry) => total + entry.bytes, 0);
    if (bytes > this.limits.stateBytes) throw new RangeError('Generation state checkpoints exceed the authorized state cache byte limit');
    while (this.states.size >= this.limits.stateEntries || this.stateBytes + bytes > this.limits.stateBytes) { this.release({ id: this.states.keys().next().value! }); this.evictions++; }
    const handle = Object.freeze({ id: `${this.owner}:${globalThis.crypto.randomUUID()}` }); this.states.set(handle.id, { snapshots, bytes }); this.stateBytes += bytes; return handle;
  }
  discardState(state: GenerationStateSnapshot): void { disposeTensors(state.cache); }
  release(handle: GenerationStateHandle): void {
    if (!handle.id.startsWith(`${this.owner}:`)) throw new TypeError('Generation state is not owned by this engine');
    const state = this.states.get(handle.id); if (!state) return;
    this.states.delete(handle.id); this.stateBytes -= state.bytes; for (const snapshot of state.snapshots) this.discardState(snapshot);
  }
  feature(key: string): Tensor | undefined {
    const feature = this.features.get(key);
    if (!feature) { this.misses++; return undefined; }
    this.hits++; this.features.delete(key); this.features.set(key, feature); return feature.tensor;
  }
  async snapshotFeature(tensor: Tensor): Promise<Tensor> {
    const owned = await ownedTensor(tensor);
    if (tensorBytes(owned) > this.limits.visionBytes) { owned.dispose(); throw new RangeError('Vision features exceed the authorized encoder cache byte limit'); }
    return owned;
  }
  commitFeatures(staged: Map<string, Tensor>): void {
    for (const [key, tensor] of staged) {
      const previous = this.features.get(key);
      if (previous) { this.features.delete(key); this.visionBytes -= previous.bytes; previous.tensor.dispose(); }
      const bytes = tensorBytes(tensor);
      while (this.features.size >= this.limits.visionEntries || this.visionBytes + bytes > this.limits.visionBytes) {
        const oldest = this.features.keys().next().value!; const feature = this.features.get(oldest)!;
        this.features.delete(oldest); this.visionBytes -= feature.bytes; feature.tensor.dispose(); this.evictions++;
      }
      this.features.set(key, { tensor, bytes }); this.visionBytes += bytes;
    }
    staged.clear();
  }
  /** Engine-private text prefixes (for example a fixed structured-output instruction); never exposed as handles. */
  checkoutPrefix(key: string, tokens: readonly bigint[]): { cache: DynamicCache; tokens: number } | undefined {
    const entry = this.prefixes.get(key);
    if (!entry || entry.tokens.length >= tokens.length || !entry.tokens.every((token, index) => token === tokens[index])) { this.prefixMisses++; return undefined; }
    const entries: Record<string, Tensor> = {};
    try { for (const [name, tensor] of Object.entries(entry.cache)) entries[name] = tensor.clone(); }
    catch (error) { for (const tensor of Object.values(entries)) tensor.dispose(); throw error; }
    this.prefixes.delete(key); this.prefixes.set(key, entry); this.prefixHits++;
    return { cache: new DynamicCache(entries), tokens: entry.tokens.length };
  }
  commitPrefix(key: string, snapshot: GenerationStateSnapshot): void {
    const previous = this.prefixes.get(key);
    if (previous) { this.prefixes.delete(key); this.prefixBytes -= previous.bytes; this.discardState(previous); }
    while (this.prefixes.size >= PREFIX_ENTRIES || this.prefixBytes + snapshot.bytes > this.limits.stateBytes) {
      const oldest = this.prefixes.keys().next().value; if (oldest === undefined) break;
      this.dropPrefix(oldest); this.evictions++;
    }
    this.prefixes.set(key, snapshot); this.prefixBytes += snapshot.bytes;
  }
  private dropPrefix(key: string): void {
    const entry = this.prefixes.get(key); if (!entry) return;
    this.prefixes.delete(key); this.prefixBytes -= entry.bytes; this.discardState(entry);
  }
  clear(): void {
    for (const id of this.states.keys()) this.release({ id });
    for (const key of [...this.prefixes.keys()]) this.dropPrefix(key);
    for (const feature of this.features.values()) feature.tensor.dispose();
    this.features.clear(); this.visionBytes = 0;
  }
}
