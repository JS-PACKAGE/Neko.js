// Kept out of tokenizer.ts: these exports name Transformers.js types, and tokenizer.ts is reachable from the public declarations,
// which must not require consumers to resolve '@huggingface/transformers' types.
import { Qwen3VLProcessor, TokenizersBackend, env } from '@huggingface/transformers';
import type { ModelFileName } from '../cache/manifest.js';
import { getRegisteredModelProfile, type RegisteredModelProfile } from '../cache/registry.js';

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
    pinnedTokenizerResource('tokenizer.json', selected).then((response) => response.json()),
    pinnedTokenizerResource('tokenizer_config.json', selected).then((response) => response.json()),
    pinnedTokenizerResource('chat_template.jinja', selected).then((response) => response.text()),
    pinnedTokenizerResource('config.json', selected).then((response) => response.json()),
  ]);
  const tokenizer = new TokenizersBackend(tokenizerJSON, tokenizerConfig);
  releaseConstructionJSON(tokenizer);
  memoizeByteLevel(tokenizer);
  return { tokenizer, template, configuration, processor: new Qwen3VLProcessor({}, { tokenizer }, template) };
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
  delete modelConfig.vocab; delete modelConfig.merges;
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
