// Kept out of tokenizer.ts: these exports name Transformers.js types, and tokenizer.ts is reachable from the public declarations,
// which must not require consumers to resolve '@huggingface/transformers' types.
import { Qwen3VLProcessor, TokenizersBackend, env } from '@huggingface/transformers';
import type * as NodeWorkerThreads from 'node:worker_threads';
import type { ModelFileName } from '../cache/manifest.js';
import { getRegisteredModelProfile, type RegisteredModelProfile } from '../cache/registry.js';
import { onPageMainThread, runPageWorker, yieldToPage } from '../runtime/page-worker.js';

export async function pinnedTokenizerResource(name: ModelFileName, selected = getRegisteredModelProfile()): Promise<Response> {
  const url = `${selected.baseUrl}${name}`;
  if (!env.customCache) throw new Error('Tokenizer planning requires an installed verified cache');
  let response: unknown = await env.customCache.match(url);
  if (!(response instanceof Response)) {
    // Installed verified cache fetch hooks enforce the selected registry manifest before returning bytes.
    response = await env.fetch(url);
    if (response instanceof Response && response.ok) await env.customCache.put(url, response.clone());
  }
  if (!(response instanceof Response) || !response.ok) throw new Error(`Verified processor resource is missing: ${name}`);
  return response;
}
export interface PinnedTextResources {
  /** Shared and never mutated; engines and planners only encode/decode with it. */
  readonly tokenizer: TokenizersBackend;
  readonly template: string;
  readonly configuration: { text_config?: { max_position_embeddings?: unknown; vocab_size?: unknown } };
  /** Text-only processor for planning. */
  readonly processor: Qwen3VLProcessor;
}
const textResources = new WeakMap<object, Map<string, Promise<PinnedTextResources>>>();
async function loadTextResources(selected: RegisteredModelProfile): Promise<PinnedTextResources> {
  const [tokenizerJSON, tokenizerConfig, template, configuration] = await Promise.all([
    pinnedTokenizerResource('tokenizer.json', selected).then(parseTokenizerJSON),
    pinnedTokenizerResource('tokenizer_config.json', selected).then((response) => response.json()),
    pinnedTokenizerResource('chat_template.jinja', selected).then((response) => response.text()),
    pinnedTokenizerResource('config.json', selected).then((response) => response.json()),
  ]);
  const tokenizer = new TokenizersBackend(tokenizerJSON, tokenizerConfig);
  releaseConstructionJSON(tokenizer);
  memoizeByteLevel(tokenizer);
  return { tokenizer, template, configuration, processor: new Qwen3VLProcessor({}, { tokenizer }, template) };
}
const isNode = typeof process !== 'undefined' && process.release?.name === 'node';
// Runs on a worker thread. For a BPE model, replaces the vocabulary and merges with the exact keys the BPE constructor would derive
// (vocabulary entries in `Object.entries` order; merge ranks keyed by `JSON.stringify` of each split pair), joined by NUL, which neither
// JSON.stringify output nor these keys (checked) contain. Any other shape posts nothing usable, and the caller parses on its own thread.
const TOKENIZER_TRANSFORM = `function nekoTokenizerMessage(bytes) {
  const json = JSON.parse(new TextDecoder().decode(bytes));
  const model = json && json.model;
  if (model && model.type === 'BPE' && model.vocab && typeof model.vocab === 'object' && !Array.isArray(model.vocab) && Array.isArray(model.merges) && model.merges.length > 0) {
    const keys = Object.keys(model.vocab);
    const ids = new Float64Array(keys.length);
    let usable = true;
    for (let i = 0; i < keys.length && usable; i++) { const id = model.vocab[keys[i]]; ids[i] = id; usable = typeof id === 'number' && !keys[i].includes('\\0'); }
    const pairs = Array.isArray(model.merges[0]) ? model.merges : model.merges.map((x) => x.split(' ', 2));
    const ranks = pairs.map((x) => JSON.stringify(x));
    if (usable && keys.length > 0 && ranks.every((key) => typeof key === 'string')) {
      delete model.vocab; delete model.merges;
      return { json, vocabulary: keys.join('\\0'), ids, ranks: ranks.join('\\0') };
    }
  }
  return { json: null };
}`;
const TOKENIZER_WORKER = `'use strict';
${TOKENIZER_TRANSFORM}
const { parentPort, workerData } = require('node:worker_threads');
const message = nekoTokenizerMessage(workerData);
parentPort.postMessage(message, message.ids ? [message.ids.buffer] : []);`;
const BROWSER_TOKENIZER_WORKER = `${TOKENIZER_TRANSFORM}
self.onmessage = (event) => { const message = nekoTokenizerMessage(event.data); self.postMessage(message, message.ids ? [message.ids.buffer] : []); };`;
type TokenizerWorkerMessage = { json: Record<string, unknown> & { model: Record<string, unknown> }; vocabulary: string; ids: Float64Array; ranks: string } | { json: null };
const MAP_BUILD_CHUNK = 8192;
const MAP_BUILD_SLICE_MS = 8;
/**
 * Parsing the pinned 19 MB tokenizer.json and building its 248k-entry vocabulary and 247k-entry merge-rank Maps took about 0.5 s of the
 * Node main thread, which model loading also needs to create ONNX sessions; in Chromium the parse and both key derivations were three
 * long tasks of about 140-180 ms each. On Node and on a browser page's main thread the parse runs on a worker, which returns the Map keys
 * as two strings; the calling thread only builds the Maps, which the bundled BPE constructor accepts as-is (scripts/build.mjs). The
 * resulting tokenizer is the same; if the worker is unavailable (for example a CSP that forbids blob workers) or fails, the calling
 * thread parses the bytes itself. An SDK worker already keeps the page free and parses in place.
 */
async function parseTokenizerJSON(response: Response): Promise<unknown> {
  const browserWorker = !isNode && onPageMainThread();
  if (!isNode && !browserWorker) return response.json();
  const bytes = new Uint8Array(await response.arrayBuffer());
  let message: TokenizerWorkerMessage = { json: null };
  try {
    if (isNode) message = await parseOnNodeWorker(bytes);
    // The page worker takes the bytes it is given, so it gets a copy: if it fails at any point, this thread still parses the original.
    else { const copy = bytes.slice(); message = await runPageWorker<TokenizerWorkerMessage>(BROWSER_TOKENIZER_WORKER, copy, [copy.buffer]); }
  } catch { /* Parsed below on this thread, which surfaces any error in the file itself. */ }
  if (message.json === null) return JSON.parse(new TextDecoder().decode(bytes));
  const vocabulary = new Map<string, number>(); const ranks = new Map<string, number>();
  const keys = message.vocabulary.split('\0'); const ids = message.ids; const rankKeys = message.ranks.split('\0');
  // Inserting about 500k keys took about 140 ms in Chromium: on a page, yield between slices so it keeps rendering.
  let sliceStarted = performance.now();
  for (let start = 0, total = keys.length + rankKeys.length; start < total; start += MAP_BUILD_CHUNK) {
    const end = Math.min(start + MAP_BUILD_CHUNK, total);
    for (let i = start; i < end && i < keys.length; i++) vocabulary.set(keys[i]!, ids[i]!);
    for (let i = Math.max(start, keys.length); i < end; i++) ranks.set(rankKeys[i - keys.length]!, i - keys.length);
    if (!isNode && end < total && performance.now() - sliceStarted >= MAP_BUILD_SLICE_MS) { await yieldToPage(); sliceStarted = performance.now(); }
  }
  return { ...message.json, model: { ...message.json.model, vocab: vocabulary, merges: [], bpe_ranks: ranks } };
}
async function parseOnNodeWorker(bytes: Uint8Array): Promise<TokenizerWorkerMessage> {
  // Node-only module in code shared with browser bundles, which must not resolve it statically.
  const protocol = 'node:';
  const { Worker }: typeof NodeWorkerThreads = await import(`${protocol}worker_threads`);
  const worker = new Worker(TOKENIZER_WORKER, { eval: true, workerData: bytes });
  try {
    const { promise, resolve, reject } = Promise.withResolvers<TokenizerWorkerMessage>();
    worker.once('message', resolve); worker.once('error', reject);
    worker.once('exit', (code) => { reject(new Error(`Tokenizer worker exited (${code})`)); });
    return await promise;
  } finally { void worker.terminate(); }
}
type BuiltTokenizer = TokenizersBackend & {
  _tokenizerJSON: Record<string, unknown>;
  _tokenizer: { tokenizer: unknown; model: { config: Record<string, unknown>; merges?: unknown } };
};
/**
 * Transformers.js 4.2.0 tokenizers keep their parsed tokenizer.json (the 248k-entry vocabulary object and 247k merge pairs) after building
 * the vocabulary Map and BPE ranks they actually use: about 30 of 65 MB of JavaScript heap, which every major GC then re-marks. Only the
 * constructors read these fields; the decoder and added tokens stay for `tokenizerByteTrie`.
 */
function releaseConstructionJSON(tokenizer: TokenizersBackend): void {
  const built = tokenizer as BuiltTokenizer;
  const json = built._tokenizerJSON;
  const retained = { decoder: json.decoder, added_tokens: json.added_tokens };
  built._tokenizerJSON = retained;
  built._tokenizer.tokenizer = retained;
  const modelConfig = { ...built._tokenizer.model.config };
  delete modelConfig.vocab; delete modelConfig.merges; delete modelConfig.bpe_ranks;
  built._tokenizer.model.config = modelConfig;
  delete built._tokenizer.model.merges;
}
const BYTE_LEVEL_MEMO_ENTRIES = 1 << 16;
const BYTE_LEVEL_MEMO_LENGTH = 256;
/**
 * The ByteLevel pre-tokenizer UTF-8-encodes every pre-token and rebuilds it one character per byte, which was about half of encoding a
 * prompt; pre-tokens (words, JSON punctuation, numbers) repeat heavily, and report planning counts thousands of similar prompts. Its
 * output depends only on the pre-token, so results for pre-tokens of up to 256 UTF-16 units are memoized (at most 65,536, then the
 * memo restarts) and returned as copies; encodings are unchanged.
 */
function memoizeByteLevel(tokenizer: TokenizersBackend): void {
  const visit = (stage: unknown): void => {
    // Pre-tokenizers are callable objects (functions), not plain objects.
    if (!stage || (typeof stage !== 'object' && typeof stage !== 'function')) return;
    const sequence: unknown = (stage as { tokenizers?: unknown }).tokenizers;
    if (Array.isArray(sequence)) { for (const child of sequence) visit(child); return; }
    const level = stage as { byte_encoder?: unknown; text_encoder?: unknown; pre_tokenize_text?: unknown };
    if (typeof level.pre_tokenize_text !== 'function' || !level.byte_encoder || !(level.text_encoder instanceof TextEncoder)) return;
    const original = level.pre_tokenize_text.bind(stage) as (text: string, options?: unknown) => string[];
    const memo = new Map<string, string[]>();
    level.pre_tokenize_text = (text: string, options?: unknown): string[] => {
      if (text.length > BYTE_LEVEL_MEMO_LENGTH) return original(text, options);
      let tokens = memo.get(text);
      if (!tokens) {
        tokens = original(text, options);
        if (memo.size >= BYTE_LEVEL_MEMO_ENTRIES) memo.clear();
        memo.set(text, tokens);
      }
      return tokens.slice();
    };
  };
  visit((tokenizer as unknown as { _tokenizer?: { pre_tokenizer?: unknown } })._tokenizer?.pre_tokenizer);
}
/**
 * Verified tokenizer, chat template and configuration, read, hashed and parsed once per installed verified cache and model.
 * Re-reading them cost a full SHA-256 of tokenizer.json plus a JSON parse and tokenizer build (seconds) on every text-only plan.
 * The entry lives as long as the installation's cache object, or until `forgetPinnedTextResources`.
 */
export function pinnedTextResources(selected: RegisteredModelProfile): Promise<PinnedTextResources> {
  const owner = env.customCache;
  if (!owner) return loadTextResources(selected);
  let entries = textResources.get(owner);
  if (!entries) { entries = new Map(); textResources.set(owner, entries); }
  const key = `${selected.id}@${selected.revision}`;
  let loading = entries.get(key);
  if (!loading) {
    const created = loadTextResources(selected);
    loading = created;
    entries.set(key, created);
    created.catch(() => { if (entries.get(key) === created) entries.delete(key); });
  }
  return loading;
}
/** Drops the shared text resources of the current installation so their memory can be reclaimed. */
export function forgetPinnedTextResources(): void { const owner = env.customCache; if (owner) textResources.delete(owner); }
