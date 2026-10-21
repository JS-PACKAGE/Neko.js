import type { Page } from '../types.js';
import { canonicalJson, hashText, validatePage } from '../web/source.js';
import { NekoError } from '../errors.js';
import { cosineRanking, EmbeddingStore, embedTexts, rankOrder, reciprocalRankFusion, semanticSettings, validateEmbedder, type DocumentEmbedder, type Ranked, type SemanticCoverage, type SemanticSearchSettings } from './semantic.js';

export interface DocumentSource {
  kind: 'text' | 'page' | 'pdf' | 'ocr';
  pageNumber?: number;
  paragraphId?: string;
  /** Top-left x/y/width/height in geometryUnit, matching PDF/OCR block geometry. */
  bounds?: [number, number, number, number];
  confidence?: number;
  provenance?: 'native-text' | 'model-ocr-untrusted';
  geometryUnit?: 'pdf-points' | 'pixels';
}
export interface DocumentBlock {
  id: string;
  /** Exact UTF-16 offsets into DocumentInput.text, not into original HTML/PDF bytes. */
  startOffset: number;
  endOffset: number;
  source?: DocumentSource;
}
export interface DocumentInput {
  id: string;
  text: string;
  title?: string;
  url?: string;
  blocks?: DocumentBlock[];
}
export interface DocumentIndexOptions {
  chunkSize?: number;
  maxDocuments?: number;
  maxCharacters?: number;
  maxChunks?: number;
}
export interface DocumentIndexConfig {
  chunkSize: number;
  maxDocuments: number;
  maxCharacters: number;
  maxChunks: number;
}
export interface IndexedDocument extends DocumentInput { versionId: string; }
export interface DocumentChunk {
  id: string;
  documentId: string;
  documentVersionId: string;
  versionId: string;
  startOffset: number;
  endOffset: number;
  text: string;
  blockId?: string;
  source?: DocumentSource;
}
export interface DocumentQuote {
  kind: 'document-quote';
  indexId: string;
  documentId: string;
  documentVersionId: string;
  chunkId: string;
  chunkVersionId: string;
  /** UTF-16 offsets into the complete normalized document, not into its chunk. */
  startOffset: number;
  endOffset: number;
  quote: string;
  source?: DocumentSource;
}
export interface DocumentSearchOptions {
  topK?: number;
  documentIds?: string[];
  maxScoredChunks?: number;
}
export interface DocumentHybridSearchOptions extends DocumentSearchOptions, SemanticSearchSettings {
  embedder: DocumentEmbedder;
  signal?: AbortSignal;
}
export interface DocumentSearchHit { chunk: DocumentChunk; score: number; citation: DocumentQuote; }
export interface RetrievalCoverage {
  indexId: string;
  documents: number;
  chunks: number;
  eligibleChunks: number;
  scoredChunks: number;
  returnedChunks: number;
  queryTerms: string[];
  matchedTerms: string[];
  candidateLimitReached: boolean;
  /** Searching all lexical candidates is not a guarantee of semantic recall or completeness. */
  exhaustive: false;
  strategy: 'bm25-cjk-v1' | 'hybrid-bm25-vector-rrf-v1';
  /** Present only for hybrid retrieval. */
  semantic?: SemanticCoverage;
}
export interface DocumentSearchResult { hits: DocumentSearchHit[]; coverage: RetrievalCoverage; }
export interface DocumentIndexSnapshot {
  format: 'neko-document-index';
  version: 1;
  algorithm: 'bm25-cjk-v1';
  revision: number;
  options: DocumentIndexConfig;
  documents: IndexedDocument[];
  id: string;
}
interface ChunkEntry { chunk: DocumentChunk; length: number; }
interface IndexState {
  snapshot: DocumentIndexSnapshot;
  chunks: Map<string, ChunkEntry>;
  postings: Map<string, Map<string, number>>;
  averageLength: number;
  documents: Map<string, IndexedDocument>;
}
const compare = (left: string, right: string): number => left < right ? -1 : left > right ? 1 : 0;
const stopwords: Record<string, true | undefined> = { a: true, an: true, and: true, are: true, as: true, at: true, be: true, by: true, for: true, from: true, how: true, in: true, is: true, it: true, of: true, on: true, or: true, that: true, the: true, this: true, to: true, was: true, what: true, when: true, where: true, which: true, who: true, with: true };
const cjkClass = '\\p{Script=Han}\\p{Script=Hiragana}\\p{Script=Katakana}\\p{Script=Hangul}';
const cjk = new RegExp(`[${cjkClass}]`, 'u');
const segments = new RegExp(`([${cjkClass}]+)|[^${cjkClass}]+`, 'gu');
/** No Han, kana or Hangul code point lies below U+1100, so plain Latin words never need the Unicode-property test. */
function mayContainCjk(word: string): boolean {
  for (let i = 0; i < word.length; i++) if (word.charCodeAt(i) >= 0x1100) return cjk.test(word);
  return false;
}
/** Lowercase NFKC words, minus stopwords; CJK runs contribute every character and every adjacent pair. */
function terms(text: string): string[] {
  const result: string[] = [];
  for (const [word] of text.normalize('NFKC').toLowerCase().matchAll(/[\p{L}\p{N}]+/gu)) {
    // Most matches are plain words; only a match containing CJK needs segmenting.
    if (!mayContainCjk(word)) { if (!Object.hasOwn(stopwords, word)) result.push(word); continue; }
    for (const [segment, run] of word.matchAll(segments)) {
      if (run === undefined) { if (!Object.hasOwn(stopwords, segment)) result.push(segment); continue; }
      let previous: string | undefined;
      for (const character of segment) {
        if (previous !== undefined) result.push(previous + character);
        result.push(character); previous = character;
      }
    }
  }
  return result;
}
function object(value: unknown, label: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object`);
}
function keys(value: Record<string, unknown>, allowed: string[], label: string): void {
  if (Object.keys(value).some((key) => !allowed.includes(key))) throw new TypeError(`${label} contains an unknown field`);
}
function integer(value: unknown, minimum: number, maximum: number, label: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) throw new RangeError(`${label} must be between ${minimum} and ${maximum}`);
}
function identifier(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || !value.trim() || value.length > 512) throw new TypeError(`${label} must be non-empty text of at most 512 UTF-16 units`);
}
function boundary(text: string, offset: number): boolean {
  return offset === 0 || offset === text.length || !(/[\uD800-\uDBFF]/u.test(text[offset - 1]!) && /[\uDC00-\uDFFF]/u.test(text[offset]!));
}
function config(input: DocumentIndexOptions = {}): DocumentIndexConfig {
  object(input, 'Index options'); keys(input, ['chunkSize', 'maxDocuments', 'maxCharacters', 'maxChunks'], 'Index options');
  const result = { chunkSize: input.chunkSize ?? 1200, maxDocuments: input.maxDocuments ?? 1000, maxCharacters: input.maxCharacters ?? 8_000_000, maxChunks: input.maxChunks ?? 100_000 };
  integer(result.chunkSize, 64, 8192, 'chunkSize'); integer(result.maxDocuments, 1, 10_000, 'maxDocuments');
  integer(result.maxCharacters, 1, 32_000_000, 'maxCharacters'); integer(result.maxChunks, 1, 100_000, 'maxChunks');
  return Object.freeze({ chunkSize: result.chunkSize, maxDocuments: result.maxDocuments, maxCharacters: result.maxCharacters, maxChunks: result.maxChunks });
}
function ownSource(input: unknown): DocumentSource {
  object(input, 'Document source'); keys(input, ['kind', 'pageNumber', 'paragraphId', 'bounds', 'confidence', 'provenance', 'geometryUnit'], 'Document source');
  if (!['text', 'page', 'pdf', 'ocr'].includes(String(input.kind))) throw new TypeError('Invalid document source kind');
  if (input.pageNumber !== undefined) integer(input.pageNumber, 1, 1_000_000, 'pageNumber');
  if (input.paragraphId !== undefined) identifier(input.paragraphId, 'paragraphId');
  if (input.bounds !== undefined && (!Array.isArray(input.bounds) || input.bounds.length !== 4 || input.bounds.some((n) => typeof n !== 'number' || !Number.isFinite(n) || n < 0))) throw new TypeError('Bounds must contain finite non-negative x/y/width/height values');
  if (input.confidence !== undefined && (typeof input.confidence !== 'number' || !Number.isFinite(input.confidence) || input.confidence < 0 || input.confidence > 1)) throw new TypeError('Confidence must be between zero and one');
  if (input.provenance !== undefined && !['native-text', 'model-ocr-untrusted'].includes(String(input.provenance))) throw new TypeError('Invalid document provenance');
  if (input.geometryUnit !== undefined && !['pdf-points', 'pixels'].includes(String(input.geometryUnit))) throw new TypeError('Invalid document geometry unit');
  const result: DocumentSource = { kind: input.kind as DocumentSource['kind'] };
  if (input.pageNumber !== undefined) result.pageNumber = input.pageNumber as number;
  if (input.paragraphId !== undefined) result.paragraphId = input.paragraphId as string;
  if (input.bounds !== undefined) result.bounds = Object.freeze([...input.bounds as number[]]) as unknown as NonNullable<DocumentSource['bounds']>;
  if (input.confidence !== undefined) result.confidence = input.confidence as number;
  if (input.provenance !== undefined) result.provenance = input.provenance as NonNullable<DocumentSource['provenance']>;
  if (input.geometryUnit !== undefined) result.geometryUnit = input.geometryUnit as NonNullable<DocumentSource['geometryUnit']>;
  return Object.freeze(result);
}
function ownDocument(input: unknown, limits: DocumentIndexConfig): DocumentInput {
  object(input, 'Document'); keys(input, ['id', 'text', 'title', 'url', 'blocks'], 'Document'); identifier(input.id, 'Document ID');
  if (typeof input.text !== 'string' || input.text.length > limits.maxCharacters) throw new TypeError('Document text exceeds the index limit or is not text');
  for (const key of ['title', 'url']) if (input[key] !== undefined && (typeof input[key] !== 'string' || (input[key] as string).length > 8192)) throw new TypeError(`Document ${key} must be text of at most 8192 UTF-16 units`);
  const result: DocumentInput = { id: input.id, text: input.text };
  if (input.title !== undefined) result.title = input.title as string;
  if (input.url !== undefined) result.url = input.url as string;
  if (input.blocks !== undefined) {
    if (!Array.isArray(input.blocks) || input.blocks.length > limits.maxChunks) throw new TypeError('Document blocks exceed the index limit or are not an array');
    const ids = new Set<string>(); let previousEnd = 0;
    result.blocks = input.blocks.map((block: unknown): DocumentBlock => {
      object(block, 'Document block'); keys(block, ['id', 'startOffset', 'endOffset', 'source'], 'Document block'); identifier(block.id, 'Block ID');
      if (ids.has(block.id)) throw new TypeError('Duplicate document block ID'); ids.add(block.id);
      integer(block.startOffset, previousEnd, result.text.length, 'Block startOffset'); integer(block.endOffset, block.startOffset + 1, result.text.length, 'Block endOffset');
      if (!boundary(result.text, block.startOffset) || !boundary(result.text, block.endOffset)) throw new TypeError('Block offsets split a UTF-16 surrogate pair');
      previousEnd = block.endOffset;
      return Object.freeze({ id: block.id, startOffset: block.startOffset, endOffset: block.endOffset, ...(block.source === undefined ? {} : { source: ownSource(block.source) }) });
    });
    Object.freeze(result.blocks);
  }
  return Object.freeze(result);
}
function ranges(document: DocumentInput): { startOffset: number; endOffset: number; block?: DocumentBlock }[] {
  const result: { startOffset: number; endOffset: number; block?: DocumentBlock }[] = []; let offset = 0;
  for (const block of document.blocks ?? []) {
    if (offset < block.startOffset) result.push({ startOffset: offset, endOffset: block.startOffset });
    result.push({ startOffset: block.startOffset, endOffset: block.endOffset, block }); offset = block.endOffset;
  }
  if (offset < document.text.length) result.push({ startOffset: offset, endOffset: document.text.length });
  return result;
}
/** Indexed documents are deep-frozen snapshots, so their canonical JSON can be cached for the lifetime of the object. */
const canonicalDocuments = new WeakMap<IndexedDocument, string>();
function canonicalDocument(document: IndexedDocument): string {
  let text = canonicalDocuments.get(document);
  if (text === undefined) { text = canonicalJson(document); canonicalDocuments.set(document, text); }
  return text;
}
interface CreatedChunk { id: string; entry: ChunkEntry; frequencies: Map<string, number>; }
function chunkDocument(document: DocumentInput, versionId: string, options: DocumentIndexConfig, capacity: number): CreatedChunk[] {
  const created: CreatedChunk[] = [];
  for (const range of ranges(document)) {
    let start = range.startOffset;
    while (start < range.endOffset) {
      let end = Math.min(start + options.chunkSize, range.endOffset);
      if (!boundary(document.text, end)) end--;
      // Prefer an existing paragraph/sentence boundary, but never drop intervening text.
      if (end < range.endOffset) {
        const slice = document.text.slice(start, end); let split = -1;
        for (const match of slice.matchAll(/\n+|[.!?。！？]\s+/gu)) if (match.index + match[0].length >= options.chunkSize / 2) split = match.index + match[0].length;
        if (split > 0) end = start + split;
      }
      const text = document.text.slice(start, end);
      if (text.trim()) {
        if (created.length >= capacity) throw new RangeError('Document index chunk capacity exceeded');
        const id = JSON.stringify([document.id, start, end]);
        const chunk: DocumentChunk = Object.freeze({ id, documentId: document.id, documentVersionId: versionId, versionId: `${versionId}:${start}:${end}`, startOffset: start, endOffset: end, text,
          ...(range.block === undefined ? {} : { blockId: range.block.id, ...(range.block.source === undefined ? {} : { source: range.block.source }) }) });
        const frequencies = new Map<string, number>(); const tokens = terms(text);
        for (const term of tokens) frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
        created.push({ id, entry: { chunk, length: tokens.length }, frequencies });
      }
      start = end;
    }
  }
  return created;
}
/**
 * Builds the index state. With `previous` (same options), documents whose content is unchanged keep their version ID, chunks and
 * postings; only changed, added and removed documents are re-tokenized. The result is identical to a from-scratch build.
 */
async function build(documents: DocumentInput[], options: DocumentIndexConfig, revision: number, previous?: IndexState): Promise<IndexState> {
  if (documents.length > options.maxDocuments || documents.reduce((n, doc) => n + doc.text.length, 0) > options.maxCharacters) throw new RangeError('Document index capacity exceeded');
  const sorted = [...documents].sort((a, b) => compare(a.id, b.id));
  const priorDocuments = previous?.snapshot.options === options ? new Map(previous.snapshot.documents.map((document) => [document.id, document])) : undefined;
  const unchanged = (document: DocumentInput): IndexedDocument | undefined => {
    const prior = priorDocuments?.get(document.id);
    return prior && prior.text === document.text && prior.title === document.title && prior.url === document.url && prior.blocks === document.blocks ? prior : undefined;
  };
  // Patching only pays off while most of the corpus is untouched.
  let changedCharacters = 0; let totalCharacters = 0;
  for (const document of sorted) { totalCharacters += document.text.length; if (!unchanged(document)) changedCharacters += document.text.length; }
  const incremental = previous !== undefined && priorDocuments !== undefined && changedCharacters * 2 < totalCharacters;
  const priorChunks = new Map<string, ChunkEntry[]>();
  if (incremental) for (const entry of previous.chunks.values()) { const list = priorChunks.get(entry.chunk.documentId); if (list) list.push(entry); else priorChunks.set(entry.chunk.documentId, [entry]); }
  const indexed: IndexedDocument[] = []; const chunks = new Map<string, ChunkEntry>(); const created: CreatedChunk[] = []; let totalLength = 0;
  const keptDocuments = new Set<string>();
  for (const document of sorted) {
    if (indexed.at(-1)?.id === document.id) throw new TypeError('Duplicate document ID');
    const prior = incremental ? unchanged(document) : undefined;
    let versionId: string; let entries: { id: string; entry: ChunkEntry }[]; let canonicalText: string | undefined;
    if (prior) {
      versionId = prior.versionId; keptDocuments.add(document.id);
      entries = (priorChunks.get(document.id) ?? []).map((entry) => ({ id: entry.chunk.id, entry }));
    } else {
      canonicalText = canonicalJson(document); versionId = await hashText(canonicalText);
      const fresh = chunkDocument(document, versionId, options, options.maxChunks - chunks.size);
      for (const item of fresh) created.push(item);
      entries = fresh;
    }
    const indexedDocument = prior ?? Object.freeze({ ...document, versionId });
    // 'versionId' sorts after every other document key, so the indexed form is the hashed form with one member appended.
    if (canonicalText !== undefined) canonicalDocuments.set(indexedDocument, `${canonicalText.slice(0, -1)},"versionId":${JSON.stringify(versionId)}}`);
    indexed.push(indexedDocument);
    for (const { id, entry } of entries) {
      if (chunks.size >= options.maxChunks) throw new RangeError('Document index chunk capacity exceeded');
      chunks.set(id, entry); totalLength += entry.length;
    }
  }
  const postings = new Map<string, Map<string, number>>();
  if (incremental) {
    // Postings of untouched terms are immutable and shared with the previous state; only terms of changed, removed or added chunks are rebuilt.
    for (const [term, posting] of previous.postings) postings.set(term, posting);
    const removed = new Set<string>(); const touched = new Set<string>();
    for (const entry of previous.chunks.values()) {
      if (keptDocuments.has(entry.chunk.documentId)) continue;
      removed.add(entry.chunk.id); for (const term of terms(entry.chunk.text)) touched.add(term);
    }
    const additions = new Map<string, [string, number][]>();
    for (const { id, frequencies } of created) for (const [term, count] of frequencies) {
      touched.add(term); const list = additions.get(term); if (list) list.push([id, count]); else additions.set(term, [[id, count]]);
    }
    const order = new Map<string, number>(); let position = 0;
    for (const id of chunks.keys()) order.set(id, position++);
    for (const term of touched) {
      const entries: [string, number][] = [];
      for (const item of previous.postings.get(term) ?? []) if (!removed.has(item[0])) entries.push(item);
      const added = additions.get(term);
      if (added) { for (const item of added) entries.push(item); entries.sort((left, right) => order.get(left[0])! - order.get(right[0])!); }
      if (entries.length) postings.set(term, new Map(entries)); else postings.delete(term);
    }
  } else {
    for (const { id, frequencies } of created) for (const [term, count] of frequencies) {
      let posting = postings.get(term); if (!posting) { posting = new Map(); postings.set(term, posting); } posting.set(id, count);
    }
  }
  Object.freeze(indexed);
  const payload = { format: 'neko-document-index' as const, version: 1 as const, algorithm: 'bm25-cjk-v1' as const, revision, options, documents: indexed };
  // Equal to hashValue(payload) (keys sorted: algorithm, documents, format, options, revision, version) without re-serializing unchanged documents.
  const canonicalPayload = `{"algorithm":"bm25-cjk-v1","documents":[${indexed.map(canonicalDocument).join(',')}],"format":"neko-document-index","options":${canonicalJson(options)},"revision":${revision},"version":1}`;
  return { snapshot: Object.freeze({ ...payload, id: await hashText(canonicalPayload) }), chunks, postings, averageLength: chunks.size ? totalLength / chunks.size : 0, documents: new Map(indexed.map((document) => [document.id, document])) };
}
function unversioned(document: IndexedDocument): DocumentInput {
  return { id: document.id, text: document.text, ...(document.title === undefined ? {} : { title: document.title }), ...(document.url === undefined ? {} : { url: document.url }), ...(document.blocks === undefined ? {} : { blocks: document.blocks }) };
}

/** Caller-owned local index. Persistence contains source text; it is never uploaded by this module. */
export class DocumentIndex {
  private pending: Promise<unknown> = Promise.resolve();
  private readonly embeddings = new EmbeddingStore();
  private constructor(private state: IndexState) {}
  static async create(input: readonly DocumentInput[] = [], options: DocumentIndexOptions = {}): Promise<DocumentIndex> {
    if (!Array.isArray(input)) throw new TypeError('Documents must be an array');
    const limits = config(options); if (input.length > limits.maxDocuments) throw new RangeError('Document index capacity exceeded');
    const documents = input.map((document) => ownDocument(document, limits));
    return new DocumentIndex(await build(documents, limits, 0));
  }
  static async importSnapshot(input: unknown): Promise<DocumentIndex> {
    object(input, 'Index snapshot'); keys(input, ['format', 'version', 'algorithm', 'revision', 'options', 'documents', 'id'], 'Index snapshot');
    if (input.format !== 'neko-document-index' || input.version !== 1 || input.algorithm !== 'bm25-cjk-v1' || typeof input.id !== 'string' || !/^[a-f\d]{64}$/u.test(input.id)) throw new TypeError('Unsupported or invalid document index snapshot');
    integer(input.revision, 0, Number.MAX_SAFE_INTEGER, 'Index revision'); const limits = config(input.options as DocumentIndexOptions);
    const expectedId = input.id; const revision = input.revision;
    if (!Array.isArray(input.documents) || input.documents.length > limits.maxDocuments) throw new TypeError('Invalid snapshot document array');
    const versions = new Map<string, string>();
    const documents = input.documents.map((document: unknown) => {
      object(document, 'Snapshot document'); keys(document, ['id', 'text', 'title', 'url', 'blocks', 'versionId'], 'Snapshot document');
      if (typeof document.versionId !== 'string' || !/^[a-f\d]{64}$/u.test(document.versionId)) throw new TypeError('Invalid document version');
      const { versionId, ...value } = document; const owned = ownDocument(value, limits); versions.set(owned.id, versionId); return owned;
    });
    // Rebuild rather than trusting persisted postings, scores or chunk offsets.
    const state = await build(documents, limits, revision);
    if (state.snapshot.id !== expectedId || state.snapshot.documents.some((document) => versions.get(document.id) !== document.versionId)) throw new TypeError('Document index snapshot integrity mismatch');
    return new DocumentIndex(state);
  }
  get id(): string { return this.state.snapshot.id; }
  get revision(): number { return this.state.snapshot.revision; }
  get documentCount(): number { return this.state.snapshot.documents.length; }
  get chunkCount(): number { return this.state.chunks.size; }
  /** Immutable snapshot; JSON.stringify is suitable for caller-managed local persistence. */
  exportSnapshot(): DocumentIndexSnapshot { return this.state.snapshot; }
  getDocument(id: string): IndexedDocument | undefined { return this.state.documents.get(id); }
  private mutate(change: (documents: DocumentInput[]) => DocumentInput[]): Promise<void> {
    const operation = this.pending.then(async () => {
      integer(this.revision + 1, 1, Number.MAX_SAFE_INTEGER, 'Index revision');
      const next = change(this.state.snapshot.documents.map(unversioned));
      const state = await build(next, this.state.snapshot.options, this.revision + 1, this.state); this.state = state;
      this.embeddings.retain(new Set([...state.chunks.values()].map(({ chunk }) => chunk.versionId)));
    });
    this.pending = operation.catch(() => undefined); return operation;
  }
  /** Upsert is atomic. Existing answers become stale after any successful mutation. */
  updateDocument(input: DocumentInput): Promise<void> {
    const owned = ownDocument(input, this.state.snapshot.options);
    return this.mutate((documents) => [...documents.filter((document) => document.id !== owned.id), owned]);
  }
  replaceDocuments(input: readonly DocumentInput[]): Promise<void> {
    if (!Array.isArray(input) || input.length > this.state.snapshot.options.maxDocuments) throw new TypeError('Replacement documents exceed the index limit or are not an array');
    const owned = input.map((document) => ownDocument(document, this.state.snapshot.options));
    return this.mutate(() => owned);
  }
  removeDocument(id: string): Promise<void> {
    identifier(id, 'Document ID');
    return this.mutate((documents) => {
      if (!documents.some((document) => document.id === id)) throw new TypeError('Cannot remove an unknown document');
      return documents.filter((document) => document.id !== id);
    });
  }
  validateCitation(value: unknown): boolean {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
    const citation = value as Partial<DocumentQuote>; const chunk = typeof citation.chunkId === 'string' ? this.state.chunks.get(citation.chunkId)?.chunk : undefined;
    if (!chunk || citation.kind !== 'document-quote' || citation.indexId !== this.id || citation.documentId !== chunk.documentId || citation.documentVersionId !== chunk.documentVersionId || citation.chunkVersionId !== chunk.versionId || !Number.isSafeInteger(citation.startOffset) || !Number.isSafeInteger(citation.endOffset) || typeof citation.quote !== 'string' || !citation.quote.length) return false;
    const start = citation.startOffset!; const end = citation.endOffset!;
    if (start < chunk.startOffset || end > chunk.endOffset || end <= start || !boundary(chunk.text, start - chunk.startOffset) || !boundary(chunk.text, end - chunk.startOffset) || chunk.text.slice(start - chunk.startOffset, end - chunk.startOffset) !== citation.quote) return false;
    return JSON.stringify(citation.source) === JSON.stringify(chunk.source);
  }
  quote(chunkId: string, text?: string): DocumentQuote {
    const chunk = this.state.chunks.get(chunkId)?.chunk; if (!chunk) throw new TypeError('Unknown document chunk');
    const quote = text ?? chunk.text; const offset = chunk.text.indexOf(quote);
    if (!quote.length || offset < 0 || !boundary(chunk.text, offset) || !boundary(chunk.text, offset + quote.length)) throw new TypeError('Quote must be an exact non-empty chunk substring');
    return Object.freeze({ kind: 'document-quote', indexId: this.id, documentId: chunk.documentId, documentVersionId: chunk.documentVersionId, chunkId: chunk.id, chunkVersionId: chunk.versionId,
      startOffset: chunk.startOffset + offset, endOffset: chunk.startOffset + offset + quote.length, quote, ...(chunk.source === undefined ? {} : { source: chunk.source }) });
  }
  private rank(question: string, options: DocumentSearchOptions): { state: IndexState; selected: Set<string> | undefined; topK: number; eligible: number; candidates: Map<string, number>; queryTerms: string[]; matchedTerms: string[]; candidateLimitReached: boolean } {
    if (typeof question !== 'string' || !question.trim() || question.length > 8192) throw new TypeError('Search query must be non-empty text of at most 8192 UTF-16 units');
    const topK = options.topK ?? 8; const maxScored = options.maxScoredChunks ?? 10_000;
    integer(topK, 1, 100, 'topK'); integer(maxScored, 1, 100_000, 'maxScoredChunks');
    let selected: Set<string> | undefined;
    if (options.documentIds !== undefined) {
      if (!Array.isArray(options.documentIds) || options.documentIds.length > this.documentCount) throw new TypeError('documentIds must contain unique known document IDs');
      selected = new Set(); for (const id of options.documentIds) { identifier(id, 'Document ID'); if (selected.has(id) || !this.getDocument(id)) throw new TypeError('documentIds must contain unique known document IDs'); selected.add(id); }
    }
    const queryTerms = [...new Set(terms(question))].sort(compare); if (queryTerms.length > 256) throw new RangeError('Search query contains more than 256 lexical terms');
    // Rare terms get admission priority when candidate scoring is capped, retaining late precise matches.
    queryTerms.sort((a, b) => (this.state.postings.get(a)?.size ?? 0) - (this.state.postings.get(b)?.size ?? 0) || compare(a, b));
    const candidates = new Map<string, number>(); const matchedTerms: string[] = []; let candidateLimitReached = false;
    const state = this.state; const eligible = selected ? [...state.chunks.values()].filter(({ chunk }) => selected.has(chunk.documentId)).length : state.chunks.size;
    for (const term of queryTerms) {
      const posting = state.postings.get(term); if (!posting) continue;
      let matched = false;
      const idf = Math.log(1 + (state.chunks.size - posting.size + 0.5) / (posting.size + 0.5));
      for (const [id, frequency] of posting) {
        const entry = state.chunks.get(id)!; if (selected && !selected.has(entry.chunk.documentId)) continue;
        matched = true;
        if (!candidates.has(id) && candidates.size >= maxScored) { candidateLimitReached = true; continue; }
        const normalization = 1.2 * (0.25 + 0.75 * entry.length / (state.averageLength || 1));
        candidates.set(id, (candidates.get(id) ?? 0) + idf * frequency * 2.2 / (frequency + normalization));
      }
      if (matched) matchedTerms.push(term);
    }
    return { state, selected, topK, eligible, candidates, queryTerms, matchedTerms, candidateLimitReached };
  }
  search(question: string, options: DocumentSearchOptions = {}): DocumentSearchResult {
    object(options, 'Search options'); keys(options, ['topK', 'documentIds', 'maxScoredChunks'], 'Search options');
    const { state, topK, eligible, candidates, queryTerms, matchedTerms, candidateLimitReached } = this.rank(question, options);
    // Only retain topK scores; ties are ordered independently of insertion/mutation history.
    const best: Ranked[] = [];
    for (const [id, score] of candidates) {
      const candidate = { id, score }; if (best.length === topK && rankOrder(candidate, best[best.length - 1]!) >= 0) continue;
      let index = 0; while (index < best.length && rankOrder(best[index]!, candidate) <= 0) index++;
      best.splice(index, 0, candidate); if (best.length > topK) best.pop();
    }
    return { hits: best.map(({ id, score }) => ({ chunk: state.chunks.get(id)!.chunk, score, citation: this.quote(id) })), coverage: {
      indexId: this.id, documents: this.documentCount, chunks: this.chunkCount, eligibleChunks: eligible, scoredChunks: candidates.size, returnedChunks: best.length,
      queryTerms, matchedTerms, candidateLimitReached, exhaustive: false, strategy: 'bm25-cjk-v1',
    } };
  }
  /**
   * BM25 plus caller-supplied embeddings, fused by reciprocal rank. Hit scores are fusion scores, not similarities.
   * Chunk vectors are cached by content version; unchanged text is not re-embedded after an index mutation.
   * Retrieval is still not exhaustive or a relevance guarantee, and cited text remains an exact chunk substring.
   */
  async searchHybrid(question: string, options: DocumentHybridSearchOptions): Promise<DocumentSearchResult> {
    object(options, 'Search options'); keys(options, ['topK', 'documentIds', 'maxScoredChunks', 'embedder', 'batchSize', 'maxEmbeddedChunks', 'minSimilarity', 'signal'], 'Search options');
    validateEmbedder(options.embedder); const embedder = options.embedder; const settings = semanticSettings(options);
    if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) throw new TypeError('signal must be an AbortSignal');
    const signal = options.signal; signal?.throwIfAborted();
    const { state, selected, topK, eligible, candidates, queryTerms, matchedTerms, candidateLimitReached } = this.rank(question, options);
    const vectors = new Map<string, Float32Array>(); const missing: DocumentChunk[] = [];
    for (const { chunk } of state.chunks.values()) {
      if (selected && !selected.has(chunk.documentId)) continue;
      const cached = this.embeddings.get(embedder.id, chunk.versionId, embedder.dimensions);
      if (cached) vectors.set(chunk.id, cached); else missing.push(chunk);
    }
    if (missing.length > settings.maxEmbeddedChunks) throw new NekoError(`Search would embed ${missing.length} chunks, above maxEmbeddedChunks (${settings.maxEmbeddedChunks}); narrow documentIds or raise the limit`, 'preprocess', 'INVALID_INPUT');
    const cachedChunks = vectors.size;
    const [queryVector] = await embedTexts(embedder, [question], 'query', 1, signal);
    for (let start = 0; start < missing.length; start += settings.batchSize) {
      const batch = missing.slice(start, start + settings.batchSize);
      const embedded = await embedTexts(embedder, batch.map(({ text }) => text), 'document', settings.batchSize, signal);
      batch.forEach((chunk, offset) => { vectors.set(chunk.id, embedded[offset]!); this.embeddings.set(embedder.id, chunk.versionId, embedded[offset]!); });
    }
    signal?.throwIfAborted();
    const semantic = cosineRanking(queryVector!, vectors, settings.minSimilarity);
    const lexical = [...candidates].map(([id, score]) => ({ id, score })).sort(rankOrder);
    const fused = reciprocalRankFusion([lexical, semantic]).slice(0, topK);
    return { hits: fused.map(({ id, score }) => ({ chunk: state.chunks.get(id)!.chunk, score, citation: this.quote(id) })), coverage: {
      indexId: state.snapshot.id, documents: state.snapshot.documents.length, chunks: state.chunks.size, eligibleChunks: eligible, scoredChunks: new Set([...candidates.keys(), ...semantic.map(({ id }) => id)]).size, returnedChunks: fused.length,
      queryTerms, matchedTerms, candidateLimitReached, exhaustive: false, strategy: 'hybrid-bm25-vector-rrf-v1',
      semantic: { embedderId: embedder.id, dimensions: embedder.dimensions, embeddedChunks: missing.length, cachedChunks, scoredChunks: semantic.length, minSimilarity: settings.minSimilarity },
    } };
  }
}
export function createDocumentIndex(documents: readonly DocumentInput[] = [], options: DocumentIndexOptions = {}): Promise<DocumentIndex> { return DocumentIndex.create(documents, options); }
export function importDocumentIndex(snapshot: unknown): Promise<DocumentIndex> { return DocumentIndex.importSnapshot(snapshot); }
export function documentFromPage(page: Page, id: string = page.url): DocumentInput {
  validatePage(page); identifier(id, 'Document ID'); let text = ''; const blocks: DocumentBlock[] = [];
  for (const paragraph of page.paragraphs) {
    if (text.length) text += '\n\n'; const startOffset = text.length; text += paragraph.text;
    blocks.push({ id: paragraph.id, startOffset, endOffset: text.length, source: { kind: 'page', paragraphId: paragraph.id } });
  }
  return { id, text, url: page.url, ...(page.title === undefined ? {} : { title: page.title }), blocks };
}
export { askDocuments } from './query.js';
export type { AskDocumentsOptions, DocumentQueryHost, DocumentQueryInferenceResult, DocumentsAnswer, DocumentsAnswerClaim, DocumentAnswerRetrieval } from './query.js';
export { extractPdf, documentForIndex } from './pdf/index.js';
export { ocrImage } from './ocr.js';
export type * from './pdf/types.js';
export type { DocumentEmbedder, SemanticSearchSettings, SemanticCoverage } from './semantic.js';
