import { AutoImageProcessor, Qwen3VLProcessor, TokenizersBackend, env, Qwen3_5ForConditionalGeneration, RawImage, TextStreamer, Tensor, type PreTrainedModel, type Processor } from '@huggingface/transformers';
import { MODEL_ID, MODEL_REVISION, MODEL_DTYPE, modelFileUrl, type ModelFileName } from '../cache/manifest.js';
import type { BackendInfo } from '../backend/index.js';

export interface InferOptions { image: string | URL | Blob | RawImage; prompt: string; maxNewTokens?: number; }
export interface PrototypeResult {
  text: string;
  model: { id: string; revision: string; dtype: typeof MODEL_DTYPE };
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
  private constructor(private readonly model: PreTrainedModel, private readonly processor: Processor, readonly backend: BackendInfo, private readonly loadMs: number, private readonly profiling: boolean) {}
  static async load(backend: BackendInfo, localFilesOnly: boolean, progressCallback?: (event: unknown) => void, profilePrefix?: string, started = performance.now()): Promise<VisionEngine> {
    const options = { revision: MODEL_REVISION, local_files_only: localFilesOnly, ...(progressCallback ? { progress_callback: progressCallback } : {}) };
    // 4.2.0 tokenizer discovery probes main even with revision; assemble its public components from verified pins.
    const [tokenizerJSON, tokenizerConfig, imageProcessor, chatTemplate] = await Promise.all([
      pinnedResource('tokenizer.json').then((response) => response.json()),
      pinnedResource('tokenizer_config.json').then((response) => response.json()),
      AutoImageProcessor.from_pretrained(MODEL_ID, options),
      pinnedResource('chat_template.jinja').then((response) => response.text()),
    ]);
    const processor = new Qwen3VLProcessor({}, { tokenizer: new TokenizersBackend(tokenizerJSON, tokenizerConfig), image_processor: imageProcessor }, chatTemplate);
    const model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
      ...options, dtype: MODEL_DTYPE, device: backend.device === 'cpu' && backend.runtime === 'browser' ? 'wasm' : backend.device,
      session_options: { executionProviders: backend.executionProviders, ...(profilePrefix && backend.runtime === 'node' ? { enableProfiling: true, profileFilePrefix: profilePrefix } : {}) },
    });
    try {
      if (!processor.tokenizer) throw new Error('Loaded processor has no tokenizer');
      return new VisionEngine(model, processor, backend, performance.now() - started, !!profilePrefix && backend.runtime === 'node');
    } catch (error) { await model.dispose(); throw error; }
  }

  async infer(options: InferOptions): Promise<PrototypeResult> {
    if (!options.prompt.trim()) throw new TypeError('A non-empty prompt is required');
    const maxNewTokens = options.maxNewTokens ?? 128;
    if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 2048) throw new RangeError('maxNewTokens must be between 1 and 2048');
    const started = performance.now();
    let source = options.image;
    if (typeof source === 'string' || source instanceof URL) {
      const url = new URL(source, typeof globalThis.location === 'object' ? globalThis.location.href : undefined);
      if (url.protocol !== 'https:' && url.protocol !== 'http:' && url.protocol !== 'blob:' && url.protocol !== 'data:') throw new TypeError('Filesystem images must be supplied as a Blob or RawImage; the Node runner reads --image files');
      const response = await globalThis.fetch(url);
      if (!response.ok) throw new Error(`Image fetch failed with HTTP ${response.status}`);
      source = await response.blob();
    }
    const image = await RawImage.read(source);
    const text = this.processor.apply_chat_template([{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: options.prompt }] }], {
      add_generation_prompt: true, tokenizer_kwargs: { enable_thinking: false },
    });
    const inputs: Record<string, unknown> = await this.processor(text, [image]);
    const generationStarted = performance.now();
    let response = '';
    let firstTokenMs: number | null = null;
    let output: unknown;
    try {
      output = await this.model.generate({ ...inputs, do_sample: false, max_new_tokens: maxNewTokens,
        streamer: new TextStreamer(this.processor.tokenizer!, { skip_prompt: true, skip_special_tokens: true, callback_function: (token: string) => {
          if (token && firstTokenMs === null) firstTokenMs = performance.now() - generationStarted;
          response += token;
        } }),
      });
      if (!response.trim()) throw new Error('Model produced no decoded response');
      const sessions = Object.entries(this.model.sessions).map(([name, value]: [string, unknown]) => {
        if (typeof value !== 'object' || value === null || !('config' in value)) throw new Error('Loaded session configuration is unavailable');
        const config: unknown = value.config;
        if (typeof config !== 'object' || config === null || !('device' in config) || !('dtype' in config) || typeof config.device !== 'string' || typeof config.dtype !== 'string') throw new Error('Loaded session configuration is invalid');
        return { name, device: config.device, dtype: config.dtype };
      });
      const now = performance.now();
      return { text: response, model: { id: MODEL_ID, revision: MODEL_REVISION, dtype: MODEL_DTYPE }, backend: { ...this.backend, sessions, providerEvidence: 'loaded-session-configuration' },
        timings: { loadMs: this.loadMs, preprocessMs: generationStarted - started, firstTokenMs, generationMs: now - generationStarted, totalMs: now - started },
        memory: { jsHeapBytes: null, gpuBytes: null },
      };
    } finally {
      for (const value of Object.values(inputs)) if (value instanceof Tensor) value.dispose();
      if (output instanceof Tensor) output.dispose();
    }
  }
  async dispose(): Promise<void> {
    try {
      if (this.profiling) for (const value of Object.values(this.model.sessions)) {
        const session: unknown = value;
        if (typeof session === 'object' && session !== null && 'endProfiling' in session && typeof session.endProfiling === 'function') session.endProfiling();
      }
    } finally { await this.model.dispose(); }
  }
}
