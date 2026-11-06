import { TextStreamer, DynamicCache, LogitsProcessor, LogitsProcessorList, StoppingCriteria, Tensor, type PreTrainedModel, type Processor } from '@huggingface/transformers';
import { NekoError } from '../errors.js';
import type { StructuredBatchRequest, StructuredInferenceResult, LoadedBackend, ModelIdentity } from './engine.js';
import { StopBuffer, type GenerationOptions } from './generation.js';
import { JsonBoundary } from './json-boundary.js';
import { JsonGrammarProcessor, type TokenByteTrie } from './json-grammar.js';
import { validateStructuredValue, type CompiledStructuredSchema } from './structured.js';
import { attachGenerationDiagnostic, jsonErrorPosition, type GenerationDiagnostic } from './diagnostics.js';
import { PreviousLogitsRelease } from './decoder-patches.js';
import { BatchDecoder } from './batch-decode.js';
import type { PreprocessingReuse } from './tokenizer.js';

export interface PreparedBatchRow {
  request: StructuredBatchRequest;
  compiled: CompiledStructuredSchema;
  effective: CompiledStructuredSchema;
  inputTokens: number;
  preprocessing?: PreprocessingReuse | undefined;
  prefixTokens?: bigint[] | undefined;
}
interface BatchRuntime {
  model: PreTrainedModel;
  processor: Processor;
  identity: ModelIdentity;
  backend: LoadedBackend;
  loadMs: number;
  node: boolean;
  started: number;
  trie?: TokenByteTrie | undefined;
  prefill?: ((inputs: Record<string, unknown>, forward: (inputs: Record<string, unknown>) => Promise<Record<string, unknown>>, check: () => void) => Promise<Record<string, unknown>>) | undefined;
}

/** One generation loop, with independent recognizers and terminal accounting for every row. */
export async function generateStructuredBatch(runtime: BatchRuntime, inputs: Record<string, Tensor>, rows: PreparedBatchRow[], settings: Required<GenerationOptions>, maxNewTokens: number): Promise<PromiseSettledResult<StructuredInferenceResult>[]> {
  const { model, processor, trie } = runtime;
  const generationStarted = performance.now();
  const eos = model.generation_config?.eos_token_id;
  const eosIds = (Array.isArray(eos) ? eos : [eos]).filter((id): id is number => typeof id === 'number');
  const filler = eosIds[0] ?? Number(inputs.input_ids!.data[0]);
  const width = inputs.input_ids!.dims[1]!;
  const states = rows.map((row) => ({ row, done: false, charged: false, tokens: 0, generated: [] as bigint[], decoded: '', response: '', firstTokenMs: null as number | null,
    boundary: new JsonBoundary(), decoder: trie ? new TextDecoder('utf-8', { fatal: true }) : undefined,
    grammar: row.effective.grammar && trie ? new JsonGrammarProcessor(row.effective.grammar, trie, width, eosIds) : undefined,
    buffer: undefined as StopBuffer | undefined, result: undefined as PromiseSettledResult<StructuredInferenceResult> | undefined }));
  const decoder = new BatchDecoder(rows.length, (index) => states[index]!.done);
  let globalError: unknown;
  let globalFailed = false;
  let output: unknown;
  let cache: DynamicCache | undefined;
  let latestOutputs: Record<string, unknown> | undefined;
  const originalForward = model.forward;
  let firstForward = true;
  const listeners: { signal: AbortSignal; listener: () => void }[] = [];
  const checkGlobal = () => {
    if (globalFailed) return true;
    for (const { request } of rows) {
      if (request.options.signal?.aborted) { const cause: unknown = request.options.signal.reason; globalError = cause instanceof NekoError ? cause : new NekoError('Operation was cancelled', 'generate', 'ABORTED', { cause }); globalFailed = true; break; }
      if (request.context.budget && performance.now() >= request.context.budget.deadline) { globalError = new NekoError('Report duration budget exhausted during generation', 'generate', 'BUDGET_EXCEEDED'); globalFailed = true; break; }
    }
    return globalFailed;
  };
  const reject = (state: typeof states[number], cause: unknown, finishReason: GenerationDiagnostic['finishReason'] = 'error') => {
    if (state.result) return;
    const { options, context } = state.row.request;
    // Each row needs its own error object: attaching another row's diagnostic must not overwrite this one.
    const error = cause instanceof NekoError ? new NekoError(cause.message, cause.stage, cause.code, { cause }) : new NekoError(cause instanceof Error ? cause.message : String(cause), state.charged ? 'generate' : 'preprocess', 'OPERATION_FAILED', { cause });
    if (options.diagnostics) {
      let schema: GenerationDiagnostic['schema'];
      try { schema = state.row.compiled.validator.validate(JSON.parse(state.decoded)).errors.map(({ keyword, keywordLocation, instanceLocation }) => ({ keyword, keywordLocation, instanceLocation })); } catch { /* Invalid JSON has no schema locations. */ }
      const inputTokens = state.charged ? state.row.inputTokens : 0;
      const position = jsonErrorPosition(cause);
      const budget = context.budget;
      attachGenerationDiagnostic(error, { version: 1, stageId: context.diagnosticStage?.id ?? 'structured', attempt: context.diagnosticStage?.attempt ?? 1, stage: error.stage, code: error.code,
        finishReason: options.signal?.aborted ? 'aborted' : finishReason, usage: { inputTokens, outputTokens: state.tokens, totalTokens: inputTokens + state.tokens }, outputCharacters: state.decoded.length,
        json: { complete: state.boundary.complete, invalid: state.boundary.invalid, ...(position !== undefined ? { position } : !state.boundary.complete ? { position: state.decoded.length } : {}) },
        ...(schema?.length ? { schema } : {}), ...(budget ? { aggregateUsage: { inputTokens: budget.inputTokens, outputTokens: budget.outputTokens, totalTokens: budget.inputTokens + budget.outputTokens } } : {}),
        ...(options.diagnostics !== true && options.diagnostics.capture ? { capture: { output: state.decoded.slice(0, options.diagnostics.capture.maxCharacters), truncated: state.decoded.length > options.diagnostics.capture.maxCharacters, maxCharacters: options.diagnostics.capture.maxCharacters } } : {}) });
    }
    state.done = true; state.result = { status: 'rejected', reason: error };
  };
  const accept = (state: typeof states[number], decoded: string, final = false) => {
    const stable = final || state.decoder ? decoded : decoded.replace(/\uFFFD+$/, '');
    if (!stable.startsWith(state.decoded)) throw new NekoError('Structured token decoding changed already streamed text', 'generate', 'STRUCTURED_OUTPUT');
    const delta = stable.slice(state.decoded.length); state.decoded = stable;
    state.boundary.push(delta); state.buffer!.push(delta);
  };
  const finish = (state: typeof states[number], stopped: boolean) => {
    if (state.result) return;
    try {
      accept(state, state.decoder ? state.decoded + state.decoder.decode() : processor.tokenizer!.decode(state.generated, { skip_special_tokens: true }), true);
      state.buffer!.end();
      if (stopped) state.boundary.end();
      if (!stopped || state.boundary.invalid || !state.boundary.complete) throw new NekoError('Generated text is not one complete JSON value', 'generate', 'STRUCTURED_OUTPUT');
      const value = validateStructuredValue(state.row.compiled, JSON.parse(state.response));
      const now = performance.now();
      state.done = true;
      state.result = { status: 'fulfilled', value: { text: state.response, value, finishReason: 'stop', usage: { inputTokens: state.row.inputTokens, outputTokens: state.tokens, totalTokens: state.row.inputTokens + state.tokens },
        model: runtime.identity, backend: runtime.backend, structured: { mode: state.row.effective.mode === 'constrained' ? 'tokenizer-constrained-runtime-validation' : 'json-boundary-runtime-validation', dialect: 'draft-07' },
        timings: { loadMs: runtime.loadMs, preprocessMs: generationStarted - runtime.started, firstTokenMs: state.firstTokenMs, generationMs: now - generationStarted, totalMs: now - runtime.started }, memory: { jsHeapBytes: null, gpuBytes: null },
        ...(state.row.preprocessing ? { preprocessing: state.row.preprocessing } : {}) } };
    } catch (cause) { reject(state, cause instanceof NekoError ? cause : new NekoError('Invalid structured output', 'generate', 'STRUCTURED_OUTPUT', { cause }), stopped ? 'stop' : 'length'); }
  };
  for (const state of states) state.buffer = new StopBuffer(settings.stop, (text) => { state.response += text; });
  class RowProcessor extends LogitsProcessor {
    override _call<T extends { dims: number[]; data: ArrayLike<number | bigint> }>(ids: bigint[][], logits: T): T {
      const vocabulary = logits.dims.at(-1)!;
      const values = logits.data as Float32Array;
      if (ids.length !== states.length || values.length !== states.length * vocabulary) throw new NekoError('Invalid batch logits dimensions', 'generate', 'MODEL_OUTPUT');
      if (checkGlobal()) throw globalError;
      for (let index = 0; index < states.length; index++) {
        const state = states[index]!; const offset = index * vocabulary;
        if (!state.done && !globalFailed) {
          try { state.grammar?.([ids[index]!], { dims: [1, vocabulary], data: values.subarray(offset, offset + vocabulary) }); }
          catch (error) { reject(state, error); }
        }
        // Upstream continues sampling every row until ALL stopping criteria are true.
        if (state.done || globalFailed) { values.fill(-Infinity, offset, offset + vocabulary); values[offset + filler] = 0; }
      }
      return logits;
    }
  }
  class BatchStopping extends StoppingCriteria { override _call(): boolean[] { checkGlobal(); return states.map((state) => globalFailed || state.done); } }
  class BatchStreamer extends TextStreamer {
    private prompt = true;
    override put(tokens: bigint[][]): void {
      if (this.prompt) { this.prompt = false; return; }
      if (tokens.length !== states.length) throw new NekoError('Invalid batch token dimensions', 'generate', 'MODEL_OUTPUT');
      // Charge the entire decoder step before observing abort/deadline or validating any row.
      const active = states.map((state) => !state.done);
      for (let index = 0; index < states.length; index++) {
        const state = states[index]!; if (!active[index]) continue;
        const count = tokens[index]!.length;
        state.tokens += count;
        const budget = state.row.request.context.budget;
        if (budget) { budget.outputTokens += count; budget.remainingTokens -= count; }
        state.firstTokenMs ??= performance.now() - generationStarted;
      }
      const interrupted = checkGlobal();
      for (let index = 0; index < states.length; index++) {
        const state = states[index]!; if (!active[index]) continue;
        if (interrupted) { reject(state, globalError); continue; }
        try {
          for (const token of tokens[index]!) {
            const id = Number(token);
            if (settings.stopTokenIds.includes(id) || eosIds.includes(id)) { finish(state, true); break; }
            state.generated.push(token);
            if (state.decoder && trie) { const bytes = trie.pieces.get(id); if (bytes) accept(state, state.decoded + state.decoder.decode(bytes, { stream: true })); }
            else accept(state, processor.tokenizer!.decode(state.generated, { skip_special_tokens: true }));
            if (state.boundary.complete || state.boundary.invalid || state.buffer!.stopped) { finish(state, true); break; }
            if (state.tokens >= maxNewTokens) { finish(state, false); break; }
          }
        } catch (cause) { reject(state, cause instanceof NekoError ? cause : new NekoError('Invalid structured token output', 'generate', 'STRUCTURED_OUTPUT', { cause })); }
      }
    }
    override end(): void {}
  }
  const processors = new LogitsProcessorList(); processors.push(new RowProcessor());
  if (runtime.node) processors.push(new PreviousLogitsRelease());
  try {
    for (const { request } of rows) if (request.options.signal) {
      const signal = request.options.signal;
      const listener = () => { checkGlobal(); };
      signal.addEventListener('abort', listener, { once: true }); listeners.push({ signal, listener });
    }
    const forward = async (inputs: Record<string, unknown>) => {
      const result = await originalForward.call(model, inputs) as unknown as Record<string, unknown>;
      latestOutputs = result;
      return result;
    };
    model.forward = async (modelInputs: Record<string, unknown>) => {
      if (modelInputs.past_key_values instanceof DynamicCache) cache = modelInputs.past_key_values;
      if (checkGlobal()) throw globalError;
      const result = firstForward
        ? runtime.prefill ? await runtime.prefill(modelInputs, forward, () => { if (checkGlobal()) throw globalError; }) : await forward(modelInputs)
        : await decoder.forward(modelInputs, forward);
      firstForward = false;
      latestOutputs = result as unknown as Record<string, unknown>;
      return result;
    };
    if (checkGlobal()) throw globalError;
    for (const state of states) {
      state.charged = true;
      const budget = state.row.request.context.budget;
      if (budget) { budget.inputTokens += state.row.inputTokens; budget.remainingTokens -= state.row.inputTokens; }
    }
    output = await model.generate({ ...inputs, do_sample: false, temperature: settings.temperature, top_k: settings.topK, repetition_penalty: settings.repetitionPenalty, no_repeat_ngram_size: settings.noRepeatNgramSize,
      max_new_tokens: maxNewTokens, logits_processor: processors, stopping_criteria: new BatchStopping(), streamer: new BatchStreamer(processor.tokenizer!) });
    if (!(output instanceof Tensor) || output.dims.length !== 2 || output.dims[0] !== rows.length) throw new NekoError('Model returned invalid batch sequences', 'generate', 'MODEL_OUTPUT');
    checkGlobal();
    for (const state of states) if (!state.result) { if (globalFailed) reject(state, globalError); else finish(state, false); }
  } catch (error) { for (const state of states) if (!state.result) reject(state, error); }
  finally {
    model.forward = originalForward;
    decoder.dispose();
    for (const { signal, listener } of listeners) signal.removeEventListener('abort', listener);
    const elapsed = (performance.now() - generationStarted) / rows.length;
    for (const { request } of rows) if (request.context.budget?.timings) request.context.budget.timings.generationMs += elapsed;
    // generate() has no exception-finally cache cleanup. Capture its live cache and final decoder outputs as the single path does.
    const tensors = new Set<Tensor>(cache ? Object.values(cache) : []);
    for (const value of Object.values(latestOutputs ?? {})) if (value instanceof Tensor) tensors.add(value);
    for (const tensor of tensors) if (tensor.location !== 'none') tensor.dispose();
    if (output instanceof Tensor) output.dispose();
  }
  return states.map((state) => state.result!);
}
