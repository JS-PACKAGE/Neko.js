import { AutoImageProcessor, Qwen3VLProcessor, TokenizersBackend, Qwen3_5ForConditionalGeneration, RawImage, TextStreamer, Tensor, InterruptableStoppingCriteria, LogitsProcessorList, cat, type PreTrainedModel, type Processor } from '@huggingface/transformers';
import { type ModelProfileId, type ModelDtype } from '../cache/manifest.js';
import { getRegisteredModelProfile, type ModelId } from '../cache/registry.js';
import type { BackendInfo } from '../backend/index.js';
import { prepareImageInternal, prepareImageRegionsInternal, type ImageInput, type ImageOptions, type InternalImageOptions, type DecodedImage, type ImageObservation } from '../web/image.js';
import { ImagePreprocessCache } from '../web/image-cache.js';
export type { ImageObservation } from '../web/image.js';
import type { ResourcePolicy } from '../web/policy.js';
import { atStage, NekoError } from '../errors.js';
import { generationSettings, NucleusProcessor, StopBuffer, type GenerationOptions } from './generation.js';
import { compileStructuredSchema, validateStructuredValue, type CompiledStructuredSchema, type SchemaValue, type StructuredMode } from './structured.js';
import type { ExecutionInfo } from '../types.js';
import { inferenceChat, validateInferenceOptions } from './preflight.js';
import { JsonBoundary } from './json-boundary.js';
import { JsonGrammarProcessor, tokenizerByteTrie, type TokenByteTrie } from './json-grammar.js';
import { pinnedTokenizerResource, promptTokenCache, renderInferenceChat, structuredInstruction, type PromptTokenCache, type PreprocessingReuse } from './tokenizer.js';

export type ChatContent = { type: 'text'; text: string } | { type: 'image'; image: ImageInput };
export interface ChatMessage { role: 'system' | 'user' | 'assistant'; content: string | ChatContent[]; }
export interface InferenceBudget { remainingTokens: number; deadline: number; inputTokens: number; outputTokens: number; timings?: { preprocessMs: number; generationMs: number }; }
export interface InferOptions extends ImageOptions {
  image?: ImageInput;
  images?: ImageInput[];
  prompt?: string;
  messages?: ChatMessage[];
  maxNewTokens?: number;
  contextWindowTokens?: number;
  generation?: GenerationOptions;
  onToken?: (text: string) => void;
  hardDeadlineMs?: number;
}
export interface StructuredInferOptions<S = unknown> extends InferOptions { schema: S; structuredMode?: StructuredMode; }
export interface InferencePlanOptions extends InferOptions { schema?: unknown; structuredMode?: StructuredMode; }
export interface InternalInferenceContext { budget?: InferenceBudget | undefined; preparedImages?: PreparedImage[] | undefined; structured?: CompiledStructuredSchema | undefined; }
export interface InferencePlan {
  inputTokens: number;
  maxNewTokens: number;
  contextLimit: number;
  availableOutputTokens: number;
  fits: boolean;
  model: ModelIdentity;
  images?: ImageObservation[];
  execution?: ExecutionInfo;
  preprocessing?: PreprocessingReuse;
}
export interface PreparedImage { readonly input: ImageInput; readonly observation: ImageObservation; }
export interface ModelIdentity { id: string; revision: string; profile: ModelProfileId; dtype: ModelDtype; }
export type LoadedBackend = BackendInfo & { sessions: { name: string; device: string; dtype: string }[]; providerEvidence: 'loaded-session-configuration'; };
export interface RuntimeReadiness { loaded: true; textReady: boolean; visionReady: boolean; model: ModelIdentity; backend: LoadedBackend; memory: { jsHeapBytes: null; gpuBytes: null }; }
export interface InferenceResult {
  text: string;
  finishReason: 'stop' | 'length';
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
  model: ModelIdentity;
  backend: LoadedBackend;
  images?: ImageObservation[];
  timings: { loadMs: number; preprocessMs: number; firstTokenMs: number | null; generationMs: number; totalMs: number; queueWaitMs?: number };
  memory: { jsHeapBytes: number | null; gpuBytes: null };
  execution?: ExecutionInfo;
  preprocessing?: PreprocessingReuse;
}
export interface StructuredInferenceResult<T = unknown> extends InferenceResult { value: T; structured: { mode: 'tokenizer-constrained-runtime-validation' | 'json-boundary-runtime-validation'; dialect: 'draft-07' }; }
function disposeInputs(inputs: Record<string, unknown>): void { for (const value of Object.values(inputs)) if (value instanceof Tensor) value.dispose(); }

export class VisionEngine {
  readonly memory = null;
  private disposePromise?: Promise<void>;
  private textReady = false;
  private visionReady = false;
  private readonly preparedImages = new WeakMap<PreparedImage, { image: DecodedImage; observation: ImageObservation }>();
  private readonly imageCache = new ImagePreprocessCache();
  private tokenTrie?: TokenByteTrie;
  private readonly tokenCache: PromptTokenCache;
  private constructor(private readonly model: PreTrainedModel, private readonly processor: Processor, readonly backend: BackendInfo, private readonly loadMs: number, private readonly profiling: boolean, readonly modelContextTokens: number, private readonly vocabularySize: number, readonly identity: ModelIdentity, private readonly policy: ResourcePolicy | undefined, private readonly offline: boolean) { this.tokenCache = promptTokenCache(identity); }
  preprocessingCacheInfo() { return this.tokenCache.info(); }
  static async load(backend: BackendInfo, localFilesOnly: boolean, signal?: AbortSignal, progressCallback?: (event: unknown) => void, profilePrefix?: string, started = performance.now(), config: { profile?: ModelProfileId; model?: ModelId; policy?: ResourcePolicy } = {}): Promise<VisionEngine> {
    return atStage('load', signal, async () => {
      const selected = getRegisteredModelProfile(config.profile, config.model);
      const options = { revision: selected.revision, local_files_only: localFilesOnly, ...(progressCallback ? { progress_callback: progressCallback } : {}) };
      const [tokenizerJSON, tokenizerConfig, imageProcessor, chatTemplate, configuration] = await Promise.all([
        pinnedTokenizerResource('tokenizer.json', selected).then((response) => response.json()), pinnedTokenizerResource('tokenizer_config.json', selected).then((response) => response.json()),
        AutoImageProcessor.from_pretrained(selected.id, options), pinnedTokenizerResource('chat_template.jinja', selected).then((response) => response.text()), pinnedTokenizerResource('config.json', selected).then((response) => response.json()),
      ]);
      signal?.throwIfAborted();
      const context: unknown = configuration.text_config?.max_position_embeddings;
      if (typeof context !== 'number' || !Number.isSafeInteger(context) || context < 1) throw new Error('Pinned model has no valid text context limit');
      const vocabulary: unknown = configuration.text_config?.vocab_size;
      if (typeof vocabulary !== 'number' || !Number.isSafeInteger(vocabulary) || vocabulary < 1) throw new Error('Pinned model has no valid vocabulary size');
      const processor = new Qwen3VLProcessor({}, { tokenizer: new TokenizersBackend(tokenizerJSON, tokenizerConfig), image_processor: imageProcessor }, chatTemplate);
      const model = await Qwen3_5ForConditionalGeneration.from_pretrained(selected.id, { ...options, dtype: selected.dtype, device: backend.device,
        session_options: { executionProviders: backend.executionProviders, ...(profilePrefix && backend.runtime === 'node' ? { enableProfiling: true, profileFilePrefix: profilePrefix } : {}) },
      });
      try {
        signal?.throwIfAborted();
        if (!processor.tokenizer) throw new Error('Loaded processor has no tokenizer');
        return new VisionEngine(model, processor, backend, performance.now() - started, !!profilePrefix && backend.runtime === 'node', context, vocabulary,
          { id: selected.id, revision: selected.revision, profile: selected.profile, dtype: selected.dtype }, config.policy, localFilesOnly);
      } catch (error) { await model.dispose(); throw error; }
    }, (engine) => engine.dispose());
  }
  contextLimit(requested?: number): number {
    const limit = requested ?? Math.min(4096, this.modelContextTokens);
    if (!Number.isSafeInteger(limit) || limit < 32 || limit > this.modelContextTokens) throw new RangeError(`contextWindowTokens must be between 32 and ${this.modelContextTokens}`);
    return limit;
  }
  countPrompt(prompt: string, structured?: CompiledStructuredSchema): number {
    return this.processor.tokenizer!.encode(renderInferenceChat(this.processor, { prompt }, structured ? structuredInstruction(structured) : undefined).text).length;
  }
  splitText(text: string, maxTokens: number): string[] {
    if (!Number.isSafeInteger(maxTokens) || maxTokens < 1) throw new RangeError('Chunk token budget must be positive');
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      let low = 1; let high = text.length - start; let best = start;
      while (low <= high) {
        const length = Math.floor((low + high) / 2); let end = start + length;
        if (end < text.length && /[\uDC00-\uDFFF]/.test(text[end]!)) end--;
        const count = this.processor.tokenizer!.encode(text.slice(start, end), { add_special_tokens: false }).length;
        if (count <= maxTokens) { best = Math.max(best, end); low = length + 1; } else high = length - 1;
      }
      if (best === start) throw new NekoError('A source character cannot fit within the chunk token budget', 'report', 'CONTEXT_LIMIT');
      chunks.push(text.slice(start, best)); start = best;
    }
    return chunks;
  }
  async prepareImage(input: ImageInput, options: ImageOptions = {}): Promise<PreparedImage> {
    const imageOptions: InternalImageOptions = { ...options, _offline: this.offline, _policy: this.policy };
    const { image, observation } = await atStage('image', options.signal, () => prepareImageInternal(input, imageOptions, this.imageCache));
    const prepared: PreparedImage = Object.freeze({ input, observation });
    this.preparedImages.set(prepared, { image, observation });
    return prepared;
  }
  private async jointInputs(text: string, images: RawImage[]): Promise<Record<string, unknown>> {
    if (!images.length) return this.processor(text);
    if (images.length === 1) return this.processor(text, images);
    const parts: Record<string, unknown>[] = [];
    try {
      // 4.2.0 treats an image array as temporal frames. Process independently before concatenating patches/grids.
      for (const image of images) parts.push(await this.processor.image_processor!([image]));
      const grids = parts.map((part) => { if (!(part.image_grid_thw instanceof Tensor)) throw new Error('Invalid image grid'); return part.image_grid_thw; });
      const pixels = parts.map((part) => { if (!(part.pixel_values instanceof Tensor)) throw new Error('Invalid image patches'); return part.pixel_values; });
      const config: unknown = this.processor.image_processor!.config;
      const merge: unknown = typeof config === 'object' && config !== null && 'merge_size' in config ? config.merge_size : undefined;
      if (typeof merge !== 'number' || !Number.isSafeInteger(merge) || merge < 1) throw new Error('Invalid pinned image merge size');
      let index = 0;
      const expanded = text.replaceAll('<|image_pad|>', () => {
        const grid = grids[index++]; if (!grid) throw new TypeError('Image markers exceed supplied images');
        const count = Array.from(grid.data, Number).reduce((a, b) => a * b, 1) / (merge * merge);
        if (!Number.isSafeInteger(count) || count < 1) throw new Error('Invalid image token expansion');
        return '<|image_pad|>'.repeat(count);
      });
      if (index !== images.length) throw new TypeError('Supplied images exceed image markers');
      const inputs = this.processor.tokenizer!(expanded) as Record<string, unknown>;
      try {
        inputs.pixel_values = cat(pixels, 0);
        inputs.image_grid_thw = cat(grids, 0);
        return inputs;
      } catch (error) { disposeInputs(inputs); throw error; }
    } finally { for (const part of parts) disposeInputs(part); }
  }
  private loadedBackend(): LoadedBackend {
    const sessions = Object.entries(this.model.sessions).map(([name, value]: [string, unknown]) => {
      if (typeof value !== 'object' || value === null || !('config' in value)) throw new Error('Loaded session configuration is unavailable');
      const config: unknown = value.config;
      if (typeof config !== 'object' || config === null || !('device' in config) || !('dtype' in config) || typeof config.device !== 'string' || typeof config.dtype !== 'string') throw new Error('Loaded session configuration is invalid');
      return { name, device: config.device, dtype: config.dtype };
    });
    return { ...this.backend, sessions, providerEvidence: 'loaded-session-configuration' };
  }
  readiness(): RuntimeReadiness { return { loaded: true, textReady: this.textReady, visionReady: this.visionReady, model: this.identity, backend: this.loadedBackend(), memory: { jsHeapBytes: null, gpuBytes: null } }; }
  async warmup(signal?: AbortSignal): Promise<RuntimeReadiness> {
    await this.infer({ prompt: 'Say OK.', maxNewTokens: 2, ...(signal ? { signal } : {}) });
    await this.infer({ prompt: 'Name the color.', image: { width: 32, height: 32, channels: 3, data: new Uint8Array(32 * 32 * 3).fill(127) }, maxNewTokens: 2, ...(signal ? { signal } : {}) });
    return this.readiness();
  }
  async inferStructured<S>(options: StructuredInferOptions<S>, context: InternalInferenceContext = {}): Promise<StructuredInferenceResult<SchemaValue<S>>> {
    const compiled = compileStructuredSchema(options.schema, options.structuredMode);
    const effective = context.structured ?? compiled;
    const result = await this.runInference(options, structuredInstruction(effective), effective, context);
    try {
      if (result.finishReason === 'length') throw new Error('Structured generation exhausted its output budget');
      const value = validateStructuredValue(compiled, JSON.parse(result.text));
      return { ...result, value, structured: { mode: effective.mode === 'constrained' ? 'tokenizer-constrained-runtime-validation' : 'json-boundary-runtime-validation', dialect: 'draft-07' } };
    } catch (cause) { throw new NekoError(cause instanceof Error ? cause.message : 'Invalid structured output', 'generate', 'STRUCTURED_OUTPUT', { cause }); }
  }
  infer(options: InferOptions, context: InternalInferenceContext = {}): Promise<InferenceResult> { return this.runInference(options, undefined, undefined, context); }
  private async prepareInputs(options: InferOptions, instruction?: string, planning = false, context: InternalInferenceContext = {}) {
    const started = performance.now();
    let allocatedInputs: Record<string, unknown> | undefined;
    const budget = context.budget;
    return atStage('preprocess', options.signal, async () => {
      validateInferenceOptions(options, planning);
      if (budget && performance.now() >= budget.deadline) throw new NekoError('Report duration budget exhausted', 'preprocess', 'BUDGET_EXCEEDED');
      let maxNewTokens = options.maxNewTokens ?? 128;
      const settings = generationSettings(options.generation, this.vocabularySize);
      const { images } = inferenceChat(options);
      const handles = context.preparedImages;
      if (handles !== undefined && (!Array.isArray(handles) || handles.length !== images.length || options.tiling !== undefined)) throw new TypeError('Prepared image handles must match ordered inputs and cannot be tiled again');
      const decoded: { image: DecodedImage; observation: ImageObservation }[] = [];
      const imageCounts: number[] = [];
      for (let index = 0; index < images.length; index++) {
        const input = images[index]!;
        if (options.tiling !== undefined) {
          const regions = await prepareImageRegionsInternal(input, { ...options, _offline: this.offline, _policy: this.policy }, this.imageCache);
          if (decoded.length + regions.length > 64) throw new RangeError('At most 64 processed image regions are supported per request');
          decoded.push(...regions); imageCounts.push(regions.length);
        } else {
          const handle = handles ? handles[index]! : await this.prepareImage(input, options);
          const prepared = this.preparedImages.get(handle);
          if (!prepared || handles && input !== handle.input) throw new TypeError('Prepared image handle is not owned by this engine or does not match its input');
          decoded.push(prepared); imageCounts.push(1);
        }
      }
      const chat = renderInferenceChat(this.processor, options, instruction, imageCounts);
      const cached = decoded.length ? undefined : await this.tokenCache.inputs(chat.text, () => this.jointInputs(chat.text, []));
      const inputs = cached?.inputs ?? await this.jointInputs(chat.text, decoded.map(({ image }) => new RawImage(image.data, image.width, image.height, image.channels)));
      allocatedInputs = inputs;
      const ids = inputs.input_ids;
      if (!(ids instanceof Tensor) || ids.dims.length !== 2) throw new Error('Processor produced invalid input IDs');
      const inputTokens = ids.dims[1]!;
      if (budget) {
        if (performance.now() >= budget.deadline) throw new NekoError('Report duration budget exhausted', 'preprocess', 'BUDGET_EXCEEDED');
        if (inputTokens + 1 > budget.remainingTokens) throw new NekoError('Report total token budget exhausted before generation', 'preprocess', 'BUDGET_EXCEEDED');
        maxNewTokens = Math.min(maxNewTokens, budget.remainingTokens - inputTokens);
      }
      const contextLimit = this.contextLimit(options.contextWindowTokens);
      if (!planning && inputTokens + maxNewTokens > contextLimit) throw new NekoError(`Input (${inputTokens}) plus output budget (${maxNewTokens}) exceeds context limit (${contextLimit})`, 'preprocess', 'CONTEXT_LIMIT');
      return { inputs, inputTokens, maxNewTokens, contextLimit, settings, images: decoded.map(({ observation }) => observation), preprocessing: cached?.reuse };
    }).catch((error: unknown) => { if (allocatedInputs) disposeInputs(allocatedInputs); throw error; })
      .finally(() => { if (budget?.timings) budget.timings.preprocessMs += performance.now() - started; });
  }
  async planInference(options: InferencePlanOptions, compiled?: CompiledStructuredSchema): Promise<InferencePlan> {
    const schema = compiled ?? (options.schema === undefined ? undefined : compileStructuredSchema(options.schema, options.structuredMode));
    const prepared = await this.prepareInputs(options, schema ? structuredInstruction(schema) : undefined, true);
    try {
      const { inputTokens, maxNewTokens, contextLimit, images } = prepared;
      return { inputTokens, maxNewTokens, contextLimit, availableOutputTokens: Math.max(0, contextLimit - inputTokens), fits: inputTokens + maxNewTokens <= contextLimit, model: this.identity, ...(images.length ? { images } : {}), ...(prepared.preprocessing ? { preprocessing: prepared.preprocessing } : {}) };
    } finally { disposeInputs(prepared.inputs); }
  }
  private async runInference(options: InferOptions, instruction?: string, compiled?: CompiledStructuredSchema, context: InternalInferenceContext = {}): Promise<InferenceResult> {
    const signal = options.signal; const started = performance.now(); const structured = compiled !== undefined;
    const budget = context.budget;
    const checkDeadline = () => { if (budget && performance.now() >= budget.deadline) throw new NekoError('Report duration budget exhausted', 'generate', 'BUDGET_EXCEEDED'); };
    const constrainedTrie = compiled?.grammar ? (this.tokenTrie ??= tokenizerByteTrie(this.processor.tokenizer!)) : undefined;
    const prepared = await this.prepareInputs(options, instruction, false, context);
    const { inputs, inputTokens, maxNewTokens, settings, images } = prepared;
    const generationStarted = performance.now();
    const incrementalDecoder = constrainedTrie ? new TextDecoder('utf-8', { fatal: true }) : undefined;
    const stopping = new InterruptableStoppingCriteria();
    const interrupt = () => stopping.interrupt();
    signal?.addEventListener('abort', interrupt, { once: true });
    let response = ''; let firstTokenMs: number | null = null; let callbackError: unknown; let callbackFailed = false; let output: unknown;
    let outputTokens = 0; let explicitStop = false; let deadlineExceeded = false;
    const generated: bigint[] | undefined = !constrainedTrie && (structured || settings.stop.length) ? [] : undefined;
    const boundary = structured ? new JsonBoundary() : undefined;
    let structuredDecoded = '';
    const buffer = new StopBuffer(settings.stop, (text) => {
      if (signal?.aborted || callbackFailed) return;
      response += text;
      try { options.onToken?.(text); } catch (error) { callbackFailed = true; callbackError = error; interrupt(); }
    });
    const acceptStructured = (decoded: string, final = false) => {
      // A partial UTF-8 token may decode to replacement characters until its next token.
      const stable = final || incrementalDecoder ? decoded : decoded.replace(/\uFFFD+$/, '');
      if (!stable.startsWith(structuredDecoded)) throw new NekoError('Structured token decoding changed already streamed text', 'generate', 'STRUCTURED_OUTPUT');
      const delta = stable.slice(structuredDecoded.length);
      structuredDecoded = stable;
      boundary!.push(delta);
      buffer.push(delta);
      if (boundary!.complete || boundary!.invalid || buffer.stopped) { explicitStop = true; interrupt(); }
    };
    const streamer = new TextStreamer(this.processor.tokenizer!, { skip_prompt: true, skip_special_tokens: true, callback_function: (text: string) => { buffer.push(text); if (buffer.stopped) { explicitStop = true; interrupt(); } } });
    const originalPut = streamer.put.bind(streamer);
    if (structured) streamer.end = () => {};
    let prompt = true;
    streamer.put = (value: bigint[][]) => {
      if (prompt) { prompt = false; if (!structured) originalPut(value); return; }
      const tokens = value[0]!;
      outputTokens += tokens.length;
      if (budget) { budget.outputTokens += tokens.length; budget.remainingTokens -= tokens.length; }
      if (firstTokenMs === null) firstTokenMs = performance.now() - generationStarted;
      const stopIndex = tokens.findIndex((id) => settings.stopTokenIds.includes(Number(id)));
      const visible = stopIndex < 0 ? tokens : tokens.slice(0, stopIndex);
      generated?.push(...visible);
      if (!structured && visible.length) originalPut([visible]);
      if (stopIndex >= 0) { explicitStop = true; interrupt(); }
      if (incrementalDecoder && constrainedTrie) {
        for (const id of visible) {
          const bytes = constrainedTrie.pieces.get(Number(id));
          if (bytes) acceptStructured(structuredDecoded + incrementalDecoder.decode(bytes, { stream: true }));
        }
      } else if (generated?.length) {
        const decoded = this.processor.tokenizer!.decode(generated, { skip_special_tokens: true });
        if (structured) acceptStructured(decoded);
        if (settings.stop.some((stop) => decoded.includes(stop))) { explicitStop = true; interrupt(); }
      }
      if (budget && performance.now() >= budget.deadline) { deadlineExceeded = true; interrupt(); }
    };
    const processors = new LogitsProcessorList();
    if (compiled?.grammar && constrainedTrie) {
      const eos = this.model.generation_config?.eos_token_id;
      const eosIds = (Array.isArray(eos) ? eos : [eos]).filter((id): id is number => typeof id === 'number');
      processors.push(new JsonGrammarProcessor(compiled.grammar, constrainedTrie, inputTokens, eosIds));
    }
    if (settings.sampling && settings.topP < 1) processors.push(new NucleusProcessor(settings.topP, settings.temperature, settings.topK));
    try {
      return await atStage('generate', signal, async () => {
        checkDeadline();
        if (budget) { budget.inputTokens += inputTokens; budget.remainingTokens -= inputTokens; }
        output = await this.model.generate({ ...inputs, do_sample: settings.sampling, temperature: settings.temperature, top_k: settings.topK,
          repetition_penalty: settings.repetitionPenalty, no_repeat_ngram_size: settings.noRepeatNgramSize,
          max_new_tokens: maxNewTokens, stopping_criteria: stopping, logits_processor: processors, streamer,
        });
        if (structured) acceptStructured(incrementalDecoder ? structuredDecoded + incrementalDecoder.decode() : generated?.length ? this.processor.tokenizer!.decode(generated, { skip_special_tokens: true }) : '', true);
        buffer.end();
        if (callbackFailed) throw new NekoError(callbackError instanceof Error ? callbackError.message : String(callbackError), 'generate', 'OPERATION_FAILED', { cause: callbackError });
        signal?.throwIfAborted();
        if (deadlineExceeded) throw new NekoError('Report duration budget exhausted during generation', 'generate', 'BUDGET_EXCEEDED');
        if (!(output instanceof Tensor)) throw new NekoError('Model returned invalid output sequences', 'generate', 'MODEL_OUTPUT');
        const last = Number(output.data[output.data.length - 1]);
        const eos = this.model.generation_config?.eos_token_id;
        const stopped = explicitStop || (Array.isArray(eos) ? eos : [eos]).some((id) => id === last);
        if (boundary && !boundary.complete && !boundary.invalid && !stopped && budget && budget.remainingTokens < 1 && maxNewTokens < (options.maxNewTokens ?? 128)) {
          throw new NekoError('Report aggregate token budget exhausted before completing structured output', 'generate', 'BUDGET_EXCEEDED');
        }
        if (!response.trim() && !explicitStop) throw new NekoError('Model produced no decoded response', 'generate', structured ? 'STRUCTURED_OUTPUT' : 'MODEL_OUTPUT');
        if (boundary) {
          if (stopped) boundary.end();
          if (boundary.invalid || !boundary.complete) throw new NekoError('Generated text is not one complete JSON value', 'generate', 'STRUCTURED_OUTPUT');
          try { JSON.parse(structuredDecoded); }
          catch (cause) { throw new NekoError('Generated text is not valid JSON', 'generate', 'STRUCTURED_OUTPUT', { cause }); }
        }
        if (images.length) this.visionReady = true; else this.textReady = true;
        const now = performance.now();
        return { text: response, finishReason: stopped ? 'stop' : 'length', usage: { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens }, model: this.identity, backend: this.loadedBackend(), ...(images.length ? { images } : {}),
          timings: { loadMs: this.loadMs, preprocessMs: generationStarted - started, firstTokenMs, generationMs: now - generationStarted, totalMs: now - started }, memory: { jsHeapBytes: null, gpuBytes: null }, ...(prepared.preprocessing ? { preprocessing: prepared.preprocessing } : {}) };
      });
    } finally {
      if (budget?.timings) budget.timings.generationMs += performance.now() - generationStarted;
      signal?.removeEventListener('abort', interrupt);
      disposeInputs(inputs);
      if (output instanceof Tensor) output.dispose();
    }
  }
  dispose(): Promise<void> {
    this.disposePromise ??= (async () => {
      try {
        if (this.profiling) for (const value of Object.values(this.model.sessions)) {
          const session: unknown = value;
          if (typeof session === 'object' && session !== null && 'endProfiling' in session && typeof session.endProfiling === 'function') session.endProfiling();
        }
      } finally { this.tokenCache.clear(); this.imageCache.clear(); await this.model.dispose(); }
    })();
    return this.disposePromise;
  }
}
