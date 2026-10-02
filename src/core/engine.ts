import { AutoImageProcessor, Qwen3VLProcessor, TokenizersBackend, env, Qwen3_5ForConditionalGeneration, RawImage, TextStreamer, Tensor, InterruptableStoppingCriteria, type PreTrainedModel, type Processor } from '@huggingface/transformers';
import { MODEL_ID, MODEL_REVISION, modelFileUrl, getModelProfile, type ModelFileName, type ModelProfileId, type ModelDtype } from '../cache/manifest.js';
import type { BackendInfo } from '../backend/index.js';
import { readImage, type ImageInput, type ImageOptions } from '../web/image.js';
import { atStage, NekoError } from '../errors.js';

export interface InferOptions extends ImageOptions {
  image?: ImageInput;
  prompt: string;
  maxNewTokens?: number;
  /** Practical working-set bound; cannot exceed the verified model's context window. */
  contextWindowTokens?: number;
  /** Receives decoded text chunks, not individual token IDs. */
  onToken?: (text: string) => void;
}
export interface ModelIdentity { id: string; revision: string; profile: ModelProfileId; dtype: ModelDtype; }
export interface InferenceResult {
  text: string;
  finishReason: 'stop' | 'length';
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
  model: ModelIdentity;
  backend: BackendInfo & { sessions: { name: string; device: string; dtype: string }[]; providerEvidence: 'loaded-session-configuration'; };
  timings: { loadMs: number; preprocessMs: number; firstTokenMs: number | null; generationMs: number; totalMs: number; };
  memory: { jsHeapBytes: number | null; gpuBytes: null; };
}

async function pinnedResource(name: ModelFileName): Promise<Response> {
  const response: unknown = await env.customCache?.match(modelFileUrl(name));
  if (!(response instanceof Response)) throw new Error(`Verified processor resource is missing: ${name}`);
  return response;
}

export class VisionEngine {
  readonly memory = null;
  private disposePromise?: Promise<void>;
  private constructor(private readonly model: PreTrainedModel, private readonly processor: Processor, readonly backend: BackendInfo, private readonly loadMs: number, private readonly profiling: boolean, readonly modelContextTokens: number, readonly identity: ModelIdentity) {}
  static async load(backend: BackendInfo, localFilesOnly: boolean, signal?: AbortSignal, progressCallback?: (event: unknown) => void, profilePrefix?: string, started = performance.now(), config: { profile?: ModelProfileId } = {}): Promise<VisionEngine> {
    return atStage('load', signal, async () => {
      const selected = getModelProfile(config.profile);
      const options = { revision: MODEL_REVISION, local_files_only: localFilesOnly, ...(progressCallback ? { progress_callback: progressCallback } : {}) };
      // 4.2.0 tokenizer discovery probes main even with revision; assemble public components from verified pins.
      const [tokenizerJSON, tokenizerConfig, imageProcessor, chatTemplate, configuration] = await Promise.all([
        pinnedResource('tokenizer.json').then((response) => response.json()),
        pinnedResource('tokenizer_config.json').then((response) => response.json()),
        AutoImageProcessor.from_pretrained(MODEL_ID, options),
        pinnedResource('chat_template.jinja').then((response) => response.text()),
        pinnedResource('config.json').then((response) => response.json()),
      ]);
      signal?.throwIfAborted();
      const context: unknown = configuration.text_config?.max_position_embeddings;
      if (typeof context !== 'number' || !Number.isSafeInteger(context) || context < 1) throw new Error('Pinned model has no valid text context limit');
      const processor = new Qwen3VLProcessor({}, { tokenizer: new TokenizersBackend(tokenizerJSON, tokenizerConfig), image_processor: imageProcessor }, chatTemplate);
      const model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
        ...options, dtype: selected.dtype, device: backend.device,
        session_options: { executionProviders: backend.executionProviders, ...(profilePrefix && backend.runtime === 'node' ? { enableProfiling: true, profileFilePrefix: profilePrefix } : {}) },
      });
      try {
        signal?.throwIfAborted();
        if (!processor.tokenizer) throw new Error('Loaded processor has no tokenizer');
        return new VisionEngine(model, processor, backend, performance.now() - started, !!profilePrefix && backend.runtime === 'node', context, { id: selected.id, revision: selected.revision, profile: selected.profile, dtype: selected.dtype });
      } catch (error) { await model.dispose(); throw error; }
    }, (engine) => engine.dispose());
  }

  contextLimit(requested?: number): number {
    // 4096 is a conservative SDK working-set budget, not a claim about model capability.
    const limit = requested ?? Math.min(4096, this.modelContextTokens);
    if (!Number.isSafeInteger(limit) || limit < 32 || limit > this.modelContextTokens) throw new RangeError(`contextWindowTokens must be between 32 and ${this.modelContextTokens}`);
    return limit;
  }
  private chat(prompt: string, image: boolean): string {
    const text = this.processor.apply_chat_template([{ role: 'user', content: [...(image ? [{ type: 'image' }] : []), { type: 'text', text: prompt }] }], { add_generation_prompt: true, tokenizer_kwargs: { enable_thinking: false } });
    if (typeof text !== 'string') throw new Error('Processor chat template did not produce text');
    return text;
  }
  countPrompt(prompt: string): number { return this.processor.tokenizer!.encode(this.chat(prompt, false)).length; }
  splitText(text: string, maxTokens: number): string[] {
    if (!Number.isSafeInteger(maxTokens) || maxTokens < 1) throw new RangeError('Chunk token budget must be positive');
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      let low = 1;
      let high = text.length - start;
      let best = start;
      while (low <= high) {
        const length = Math.floor((low + high) / 2);
        let end = start + length;
        // Never split a UTF-16 surrogate pair or decode partial UTF-8 tokenizer bytes.
        if (end < text.length && /[\uDC00-\uDFFF]/.test(text[end]!)) end--;
        const count = this.processor.tokenizer!.encode(text.slice(start, end), { add_special_tokens: false }).length;
        if (count <= maxTokens) { best = Math.max(best, end); low = length + 1; }
        else high = length - 1;
      }
      if (best === start) throw new NekoError('A source character cannot fit within the chunk token budget', 'report', 'CONTEXT_LIMIT');
      chunks.push(text.slice(start, best));
      start = best;
    }
    return chunks;
  }

  async infer(options: InferOptions): Promise<InferenceResult> {
    const signal = options.signal;
    const started = performance.now();
    let allocatedInputs: Record<string, unknown> | undefined;
    const prepared = await atStage('preprocess', signal, async () => {
      if (typeof options.prompt !== 'string' || !options.prompt.trim()) throw new TypeError('A non-empty prompt is required');
      const maxNewTokens = options.maxNewTokens ?? 128;
      if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 2048) throw new RangeError('maxNewTokens must be between 1 and 2048');
      const contextLimit = this.contextLimit(options.contextWindowTokens);
      const decoded = options.image === undefined ? undefined : await atStage('image', signal, () => readImage(options.image!, options));
      const image = decoded ? new RawImage(decoded.data, decoded.width, decoded.height, decoded.channels) : undefined;
      const text = this.chat(options.prompt, image !== undefined);
      const inputs: Record<string, unknown> = image ? await this.processor(text, [image]) : await this.processor(text);
      allocatedInputs = inputs;
      const inputIds = inputs.input_ids;
      if (!(inputIds instanceof Tensor) || inputIds.dims.length !== 2) throw new Error('Processor produced invalid input IDs');
      const inputTokens = inputIds.dims[1]!;
      if (inputTokens + maxNewTokens > contextLimit) {
        throw new NekoError(`Input (${inputTokens}) plus output budget (${maxNewTokens}) exceeds context limit (${contextLimit})`, 'preprocess', 'CONTEXT_LIMIT');
      }
      return { inputs, inputTokens, maxNewTokens };
    }).catch((error: unknown) => {
      if (allocatedInputs) for (const value of Object.values(allocatedInputs)) if (value instanceof Tensor) value.dispose();
      throw error;
    });
    const { inputs, inputTokens, maxNewTokens } = prepared;
    const generationStarted = performance.now();
    const stopping = new InterruptableStoppingCriteria();
    const interrupt = () => stopping.interrupt();
    signal?.addEventListener('abort', interrupt, { once: true });
    let response = '';
    let firstTokenMs: number | null = null;
    let callbackError: unknown;
    let callbackFailed = false;
    let output: unknown;
    try {
      return await atStage('generate', signal, async () => {
        output = await this.model.generate({ ...inputs, do_sample: false, max_new_tokens: maxNewTokens, stopping_criteria: stopping,
          streamer: new TextStreamer(this.processor.tokenizer!, { skip_prompt: true, skip_special_tokens: true,
            token_callback_function: () => { if (firstTokenMs === null) firstTokenMs = performance.now() - generationStarted; },
            callback_function: (text: string) => {
              if (signal?.aborted || callbackFailed) return;
              response += text;
              try { options.onToken?.(text); } catch (error) { callbackFailed = true; callbackError = error; stopping.interrupt(); }
            },
          }),
        });
        signal?.throwIfAborted();
        if (callbackFailed) throw callbackError;
        if (!(output instanceof Tensor)) throw new NekoError('Model returned invalid output sequences', 'generate', 'MODEL_OUTPUT');
        if (!response.trim()) throw new NekoError('Model produced no decoded response', 'generate', 'MODEL_OUTPUT');
        const outputTokens = output.dims[1]! - inputTokens;
        const last = Number(output.data[output.data.length - 1]);
        const eos = this.model.generation_config?.eos_token_id;
        const stopped = (Array.isArray(eos) ? eos : [eos]).some((id) => id === last);
        const sessions = Object.entries(this.model.sessions).map(([name, value]: [string, unknown]) => {
          if (typeof value !== 'object' || value === null || !('config' in value)) throw new Error('Loaded session configuration is unavailable');
          const config: unknown = value.config;
          if (typeof config !== 'object' || config === null || !('device' in config) || !('dtype' in config) || typeof config.device !== 'string' || typeof config.dtype !== 'string') throw new Error('Loaded session configuration is invalid');
          return { name, device: config.device, dtype: config.dtype };
        });
        const now = performance.now();
        return { text: response, finishReason: stopped ? 'stop' : 'length', usage: { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens }, model: this.identity, backend: { ...this.backend, sessions, providerEvidence: 'loaded-session-configuration' },
          timings: { loadMs: this.loadMs, preprocessMs: generationStarted - started, firstTokenMs, generationMs: now - generationStarted, totalMs: now - started }, memory: { jsHeapBytes: null, gpuBytes: null } };
      });
    } finally {
      signal?.removeEventListener('abort', interrupt);
      for (const value of Object.values(inputs)) if (value instanceof Tensor) value.dispose();
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
      } finally { await this.model.dispose(); }
    })();
    return this.disposePromise;
  }
}
