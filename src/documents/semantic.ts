import { awaitUser, NekoError } from '../errors.js';

/**
 * Caller-supplied text embedder. Neko.js bundles no embedding model: the pinned multimodal model is not an
 * embedding model, and an unpinned download would bypass the SDK's integrity policy. Applications that choose a
 * model own its provenance, licensing and integrity checks.
 */
export interface DocumentEmbedder {
  /** Stable identity of the model and prompt format. Change it whenever the vectors could differ: cached vectors are keyed by it. */
  readonly id: string;
  readonly dimensions: number;
  /**
   * Returns one vector per input, in order. `kind` lets asymmetric models apply their own query/passage prefixes.
   * Vectors are untrusted callback output: length, finiteness and non-zero norm are validated before use.
   */
  embed(texts: readonly string[], options: { kind: 'query' | 'document'; signal?: AbortSignal }): Promise<readonly ArrayLike<number>[]>;
}
export interface SemanticSearchSettings {
  /** Chunks per embedder call, 1 through 256. Defaults to 32. */
  batchSize?: number;
  /** Maximum chunks this search may newly embed, 1 through 100000. Defaults to 4096; cached vectors do not count. */
  maxEmbeddedChunks?: number;
  /** Drop semantic candidates below this cosine similarity (-1 through 1). Lexical candidates are unaffected. */
  minSimilarity?: number;
}
export interface SemanticCoverage {
  embedderId: string;
  dimensions: number;
  /** Chunks embedded by this search. */
  embeddedChunks: number;
  /** Chunk vectors reused from the index-local cache. */
  cachedChunks: number;
  /** Chunks that passed the optional similarity floor. */
  scoredChunks: number;
  minSimilarity: number | null;
}
export interface ResolvedSemanticSettings { batchSize: number; maxEmbeddedChunks: number; minSimilarity: number | null; }

const fail = (message: string, code: 'INVALID_INPUT' | 'MODEL_OUTPUT' | 'OPERATION_FAILED' = 'INVALID_INPUT', cause?: unknown): never => {
  throw new NekoError(message, 'preprocess', code, cause === undefined ? undefined : { cause });
};
export function validateEmbedder(value: unknown): asserts value is DocumentEmbedder {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('embedder must be an object');
  const embedder = value as Partial<DocumentEmbedder>;
  if (typeof embedder.id !== 'string' || !embedder.id.trim() || embedder.id.length > 512 || /[\u0000-\u001f]/u.test(embedder.id)) fail('embedder.id must be non-empty text of at most 512 characters without control characters');
  if (!Number.isSafeInteger(embedder.dimensions) || embedder.dimensions! < 1 || embedder.dimensions! > 8192) fail('embedder.dimensions must be an integer between 1 and 8192');
  if (typeof embedder.embed !== 'function') fail('embedder.embed must be a function');
}
export function semanticSettings(options: SemanticSearchSettings): ResolvedSemanticSettings {
  const batchSize = options.batchSize ?? 32; const maxEmbeddedChunks = options.maxEmbeddedChunks ?? 4096; const minSimilarity = options.minSimilarity ?? null;
  if (!Number.isSafeInteger(batchSize) || batchSize < 1 || batchSize > 256) fail('batchSize must be an integer between 1 and 256');
  if (!Number.isSafeInteger(maxEmbeddedChunks) || maxEmbeddedChunks < 1 || maxEmbeddedChunks > 100_000) fail('maxEmbeddedChunks must be an integer between 1 and 100000');
  if (minSimilarity !== null && (typeof minSimilarity !== 'number' || !Number.isFinite(minSimilarity) || minSimilarity < -1 || minSimilarity > 1)) fail('minSimilarity must be a finite number between -1 and 1');
  return { batchSize, maxEmbeddedChunks, minSimilarity };
}
function unitVector(raw: unknown, dimensions: number): Float32Array {
  if (!raw || typeof raw !== 'object' || typeof (raw as ArrayLike<number>).length !== 'number' || (raw as ArrayLike<number>).length !== dimensions) return fail('Embedder returned a vector with the wrong dimensions', 'MODEL_OUTPUT');
  const vector = new Float32Array(dimensions); let norm = 0;
  for (let i = 0; i < dimensions; i++) {
    const value = (raw as ArrayLike<unknown>)[i];
    if (typeof value !== 'number' || !Number.isFinite(value)) return fail('Embedder returned a non-finite vector component', 'MODEL_OUTPUT');
    vector[i] = value; norm += vector[i]! * vector[i]!;
  }
  if (!Number.isFinite(norm) || norm === 0) return fail('Embedder returned a zero or overflowing vector', 'MODEL_OUTPUT');
  const scale = 1 / Math.sqrt(norm); for (let i = 0; i < dimensions; i++) vector[i]! *= scale;
  return vector;
}
/** Embeds in bounded batches; honors cancellation before and after every embedder call. */
export async function embedTexts(embedder: DocumentEmbedder, texts: readonly string[], kind: 'query' | 'document', batchSize: number, signal?: AbortSignal): Promise<Float32Array[]> {
  const output: Float32Array[] = [];
  for (let start = 0; start < texts.length; start += batchSize) {
    signal?.throwIfAborted();
    const batch = texts.slice(start, start + batchSize);
    let result: readonly ArrayLike<number>[];
    try { result = await awaitUser(() => embedder.embed(batch, { kind, ...(signal ? { signal } : {}) }), signal, 'preprocess'); }
    catch (cause) { if (cause instanceof NekoError) throw cause; return fail(cause instanceof Error ? cause.message : 'Embedder failed', 'OPERATION_FAILED', cause); }
    if (!Array.isArray(result) || result.length !== batch.length) fail('Embedder must return exactly one vector per input', 'MODEL_OUTPUT');
    for (const vector of result) output.push(unitVector(vector, embedder.dimensions));
  }
  return output;
}
export interface Ranked { id: string; score: number; }
const compare = (left: string, right: string): number => left < right ? -1 : left > right ? 1 : 0;
export const rankOrder = (a: Ranked, b: Ranked): number => b.score - a.score || compare(a.id, b.id);
export function cosineRanking(query: Float32Array, vectors: ReadonlyMap<string, Float32Array>, minimum: number | null): Ranked[] {
  const result: Ranked[] = [];
  for (const [id, vector] of vectors) {
    let dot = 0; for (let i = 0; i < query.length; i++) dot += query[i]! * vector[i]!;
    if (minimum === null || dot >= minimum) result.push({ id, score: dot });
  }
  return result.sort(rankOrder);
}
/** Reciprocal-rank fusion is scale-free, so BM25 and cosine scores never need calibration. */
export function reciprocalRankFusion(lists: readonly (readonly Ranked[])[], k = 60): Ranked[] {
  const scores = new Map<string, number>();
  for (const list of lists) list.forEach(({ id }, rank) => scores.set(id, (scores.get(id) ?? 0) + 1 / (k + rank + 1)));
  return [...scores].map(([id, score]) => ({ id, score })).sort(rankOrder);
}

/** Bounded index-local vector cache. Keys are content-derived chunk versions, so unchanged text is never re-embedded. */
export class EmbeddingStore {
  private readonly vectors = new Map<string, Float32Array>();
  private bytes = 0;
  constructor(private readonly maxBytes = 128 * 1024 * 1024) {}
  private key(embedderId: string, versionId: string): string { return `${embedderId}\u0000${versionId}`; }
  get(embedderId: string, versionId: string, dimensions: number): Float32Array | undefined {
    const key = this.key(embedderId, versionId); const vector = this.vectors.get(key);
    if (!vector) return undefined;
    if (vector.length !== dimensions) { this.drop(key, vector); return undefined; }
    this.vectors.delete(key); this.vectors.set(key, vector);
    return vector;
  }
  set(embedderId: string, versionId: string, vector: Float32Array): void {
    if (vector.byteLength > this.maxBytes) return;
    const key = this.key(embedderId, versionId); const previous = this.vectors.get(key); if (previous) this.drop(key, previous);
    this.vectors.set(key, vector); this.bytes += vector.byteLength;
    for (const [oldest, value] of this.vectors) { if (this.bytes <= this.maxBytes) break; this.drop(oldest, value); }
  }
  /** Keep only vectors for chunks that still exist after an index mutation. */
  retain(versionIds: ReadonlySet<string>): void {
    for (const [key, vector] of this.vectors) if (!versionIds.has(key.slice(key.indexOf('\u0000') + 1))) this.drop(key, vector);
  }
  private drop(key: string, vector: Float32Array): void { this.vectors.delete(key); this.bytes -= vector.byteLength; }
  get size(): number { return this.vectors.size; }
}
