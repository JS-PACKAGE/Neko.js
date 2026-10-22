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
