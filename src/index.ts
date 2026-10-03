import { env } from '@huggingface/transformers';
import { EngineCache, type EngineCacheStatus } from './cache/engine.js';
import { captureModelSource, installVerifiedCache, type VerifiedCacheInstallation, type ModelCacheStatus, type ModelCacheDiagnostics, type ModelSource, type CacheProgress } from './cache/model.js';
import { inspectBackend, type BackendInfo, type BackendDevice } from './backend/index.js';
import { VisionEngine, type InferOptions, type InferenceResult, type InferencePlanOptions, type InferencePlan, type StructuredInferOptions, type StructuredInferenceResult, type RuntimeReadiness } from './core/engine.js';
import { generateReport, planReport as generateReportPlan, validateDescribeOptions, ReportError, type DescribeOptions, type ReportPlan } from './report/generate.js';
import type { Page, StructuredReport, ExecutionInfo, DocumentAnswer } from './types.js';
import { atStage, NekoError, type ErrorStage } from './errors.js';
import { RequestQueue, type QueueStatus } from './core-queue.js';
import { authorizeNetwork, type ResourcePolicy } from './web/policy.js';
import type { ModelProfileId } from './cache/manifest.js';
import { getRegisteredModelProfile, type ModelId } from './cache/registry.js';
import { createWorkerClient } from './runtime/client.js';
import { selectPage } from './web/source.js';
import { extractPage } from './web/extract.js';
import { compileStructuredSchema, type SchemaValue } from './core/structured.js';
import { setActiveRequest, workerExecution } from './runtime/context.js';
import { inferenceChat, validateInferenceOptions } from './core/preflight.js';
import { createInferenceStream, type InferStreamOptions, type InferenceStreamEvent } from './runtime/stream.js';
import { planTextInference } from './core/tokenizer.js';
import { createConversationSession, type ConversationSession, type SessionOptions } from './core/session.js';
import { assertHardDeadlineUnsupported, createRuntimeDiagnostics, validateHealthOptions, type HealthOptions, type RuntimeHealth, type RuntimeDiagnostics } from './runtime/diagnostics.js';
import { queuedReadable } from './queued-stream.js';
import { askDocument, type AskOptions } from './web/query.js';
import type { ModelBundleSource } from './cache/bundle.js';
import { generateExtractiveReport, planExtractiveReport } from './report/extractive.js';
import { inferTools, type ToolDefinitions, type ToolInferOptions, type ToolInferenceResult } from './core/tools.js';
import type { GenerationStateHandle, ReuseCacheInfo, ReuseCacheLimits } from './types.js';
import { attachGenerationDiagnostic, getGenerationDiagnostic } from './core/diagnostics.js';

export * from './types.js';
export * from './web/index.js';
export * from './report/index.js';
export * from './cache/manifest.js';
export * from './cache/registry.js';
export * from './backend/index.js';
export * from './errors.js';
export * from './core/tools.js';
export { getGenerationDiagnostic } from './core/diagnostics.js';
export type { GenerationDiagnostic, GenerationDiagnosticOptions, DiagnosticCapture } from './core/diagnostics.js';
export { ReportError } from './report/generate.js';
export type { InferOptions, InferenceResult, InferencePlanOptions, InferencePlan, StructuredInferOptions, StructuredInferenceResult, RuntimeReadiness, ChatContent, ChatMessage, ModelIdentity, ImageObservation } from './core/engine.js';
export type { GenerationOptions } from './core/generation.js';
export type { DescribeOptions, ReportCheckpoint, ReportEvent, ReportBudget, ReportPlan, PartialReport } from './report/generate.js';
export type { ModelCacheStatus, ModelCacheDiagnostics, CacheProgress, ModelSource } from './cache/model.js';
export type { EngineCacheStatus } from './cache/engine.js';
export type { QueueStatus } from './core-queue.js';
export type { InferStreamOptions, InferenceStreamEvent } from './runtime/stream.js';
export type { SchemaValue, StructuredMode } from './core/structured.js';
export type { ConversationSession, SessionOptions, SessionSendOptions, SessionControlOptions, SessionSnapshot } from './core/session.js';
export type { HealthOptions, RuntimeHealth, RuntimeDiagnostics } from './runtime/diagnostics.js';
export type { ModelBundleSource } from './cache/bundle.js';

export interface NekoOptions {
  device?: BackendDevice;
  model?: ModelId;
  modelProfile?: ModelProfileId;
  modelSource?: ModelSource;
  execution?: 'inline' | 'worker';
  policy?: ResourcePolicy;
  queue?: { maxPending?: number };
  cacheDir?: string;
  cache?: { engine?: boolean; engineTtlMs?: number };
  reuseCache?: ReuseCacheLimits;
  downloadConcurrency?: number;
  resumeDownloads?: boolean;
  onCacheProgress?: (event: CacheProgress) => void;
  localFilesOnly?: boolean;
  signal?: AbortSignal;
  progressCallback?: (event: unknown) => void;
  wasmPaths?: { wasm: string; mjs: string };
  profilePrefix?: string;
}
type OwnedNekoOptions = NekoOptions & { policy: ResourcePolicy; model: ModelId; modelProfile: ModelProfileId; localFilesOnly: boolean };
export interface Neko {
  readonly cache: {
    model: {
      prefetch(signal?: AbortSignal): Promise<void>;
      status(signal?: AbortSignal): Promise<ModelCacheStatus>;
      clear(signal?: AbortSignal): Promise<void>;
      exportBundle(signal?: AbortSignal): ReadableStream<Uint8Array>;
      importBundle(source: ModelBundleSource, signal?: AbortSignal): Promise<void>;
      diagnostics(signal?: AbortSignal): Promise<ModelCacheDiagnostics>;
    };
    engine: { status(): Promise<EngineCacheStatus>; release(): Promise<void> };
  };
  readonly backend: { current(): Promise<BackendInfo | null>; detect(device?: BackendDevice, signal?: AbortSignal): Promise<BackendInfo> };
  infer(options: InferOptions): Promise<InferenceResult>;
  inferStream(options: InferStreamOptions): AsyncIterable<InferenceStreamEvent>;
  inferStructured<const S>(options: StructuredInferOptions<S>): Promise<StructuredInferenceResult<SchemaValue<S>>>;
  inferTools<const T extends ToolDefinitions>(options: ToolInferOptions<T>): Promise<ToolInferenceResult<T>>;
  releaseGenerationState(handle: GenerationStateHandle): Promise<void>;
  reuseCacheInfo(): Promise<ReuseCacheInfo | null>;
  clearReuseCaches(): Promise<void>;
  planInference(options: InferencePlanOptions): Promise<InferencePlan>;
  planReport(input: string | Page, options?: DescribeOptions): Promise<ReportPlan>;
  session(options?: SessionOptions): ConversationSession;
  ask(input: string | Page, question: string, options?: AskOptions): Promise<DocumentAnswer>;
  describe(input: string | Page, options: DescribeOptions & { format: 'markdown' }): Promise<string>;
  describe(input: string | Page, options?: DescribeOptions & { format?: 'json' }): Promise<StructuredReport>;
  describe(input: string | Page, options: DescribeOptions): Promise<StructuredReport | string>;
  load(signal?: AbortSignal): Promise<RuntimeReadiness>;
  warmup(signal?: AbortSignal): Promise<RuntimeReadiness>;
  runtimeStatus(): Promise<RuntimeReadiness | null>;
  queueStatus(): Promise<QueueStatus>;
  diagnostics(): Promise<RuntimeDiagnostics>;
  health(options?: HealthOptions): Promise<RuntimeHealth>;
  restart(): Promise<void>;
  dispose(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
}

class LocalNeko implements Neko {
  private closed = false;
  private readonly lifetime = new AbortController();
  private disposePromise?: Promise<void>;
  private readiness: RuntimeReadiness | null = null;
  readonly cache: Neko['cache'];
  readonly backend: Neko['backend'];
  constructor(private readonly engines: EngineCache<VisionEngine>, private readonly installation: VerifiedCacheInstallation, private readonly restore: () => void, backend: BackendInfo, private readonly options: OwnedNekoOptions, private readonly admission: RequestQueue) {
    this.cache = {
      model: {
        prefetch: (signal) => this.run('cache', signal, (abort) => installation.prefetch(abort)),
        status: (signal) => this.run('cache', signal, (abort) => installation.status(abort)),
        clear: (signal) => this.run('cache', signal, async (abort) => { await engines.release(); this.readiness = null; abort.throwIfAborted(); await installation.clear(abort); }),
        exportBundle: (signal) => {
          this.checkRequest('cache', signal);
          return queuedReadable((abort, operation) => this.run('cache', abort, operation), (abort) => installation.exportBundle(abort), signal);
        },
        importBundle: (source, signal) => this.run('cache', signal, (abort) => installation.importBundle(source, abort)),
        diagnostics: (signal) => this.run('cache', signal, (abort) => installation.diagnostics(abort)),
      },
      engine: { status: async () => { if (this.closed) throw new NekoError('Neko is disposed', 'cache', 'DISPOSED'); return engines.status(); }, release: () => this.run('cache', undefined, async () => { await engines.release(); this.readiness = null; }) },
    };
    this.backend = { current: async () => { if (this.closed) throw new NekoError('Neko is disposed', 'backend', 'DISPOSED'); return backend; }, detect: (device = backend.device, signal) => this.run('backend', signal, () => inspectBackend(device, options.modelProfile)) };
  }
  private run<T>(stage: ErrorStage, signal: AbortSignal | undefined, operation: (signal: AbortSignal, queueWaitMs: number) => Promise<T>, contextSignal = signal): Promise<T> {
    if (this.closed) return Promise.reject(new NekoError('Neko is disposed', stage, 'DISPOSED'));
    const abort = AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])]);
    return this.admission.enqueue(stage, abort, async (active, queueWaitMs) => {
      setActiveRequest(this, contextSignal);
      try { return await this.installation.withSignal(active, () => operation(active, queueWaitMs)); }
      finally { setActiveRequest(this, undefined); }
    });
  }
  private async use<T>(signal: AbortSignal, operation: (engine: VisionEngine) => Promise<T>): Promise<T> {
    return this.engines.use(async (engine) => { try { return await operation(engine); } finally { this.readiness = engine.readiness(); } }, signal);
  }
  private execution(): ExecutionInfo { return workerExecution(this) ?? { mode: 'inline', runtime: typeof process !== 'undefined' && process.release?.name === 'node' ? 'node' : 'browser' }; }
  private checkRequest(stage: ErrorStage, signal?: AbortSignal): void {
    if (this.closed) throw new NekoError('Neko is disposed', stage, 'DISPOSED');
    if (signal instanceof AbortSignal && signal.aborted) throw new NekoError('Operation was cancelled', stage, 'ABORTED', { cause: signal.reason });
  }
  private captureSource(input: string | Page): Promise<{ ok: true; value: string | Page } | { ok: false; error: unknown }> {
    return typeof input === 'string' ? Promise.resolve({ ok: true, value: input }) : selectPage(input).then(
      (value) => ({ ok: true as const, value }), (error: unknown) => ({ ok: false as const, error }));
  }
  async infer(options: InferOptions): Promise<InferenceResult> {
    this.checkRequest('generate', options?.signal);
    validateInferenceOptions(options);
    assertHardDeadlineUnsupported(options.hardDeadlineMs);
    return this.run('generate', options.signal, (signal, queueWaitMs) => this.use(signal, async (engine) => { const result = await engine.infer({ ...options, signal }); return { ...result, timings: { ...result.timings, queueWaitMs }, execution: this.execution() }; }));
  }
  inferStream(options: InferStreamOptions): AsyncIterable<InferenceStreamEvent> { return createInferenceStream(options, (request) => this.infer(request)); }
  async inferStructured<const S>(options: StructuredInferOptions<S>): Promise<StructuredInferenceResult<SchemaValue<S>>> {
    this.checkRequest('generate', options?.signal);
    validateInferenceOptions(options);
    assertHardDeadlineUnsupported(options.hardDeadlineMs);
    const compiled = compileStructuredSchema(options.schema, options.structuredMode);
    return this.run('generate', options.signal, (signal, queueWaitMs) => {
      return this.use(signal, async (engine) => { const result = await engine.inferStructured({ ...options, signal }, { structured: compiled }); return { ...result, timings: { ...result.timings, queueWaitMs }, execution: this.execution() }; });
    });
  }
  inferTools<const T extends ToolDefinitions>(options: ToolInferOptions<T>): Promise<ToolInferenceResult<T>> { return inferTools(this, options); }
  releaseGenerationState(handle: GenerationStateHandle): Promise<void> {
    return this.run('cache', undefined, async () => this.engines.inspect((engine) => {
      if (!engine) throw new NekoError('Generation state is unavailable', 'cache', 'INVALID_INPUT');
      engine.releaseGenerationState(handle);
    }));
  }
  reuseCacheInfo(): Promise<ReuseCacheInfo | null> { return this.run('cache', undefined, async () => this.engines.inspect((engine) => engine?.reuseCacheInfo() ?? null)); }
  clearReuseCaches(): Promise<void> { return this.run('cache', undefined, async () => this.engines.inspect((engine) => { engine?.clearReuseCaches(); })); }
  async planInference(options: InferencePlanOptions): Promise<InferencePlan> {
    this.checkRequest('preprocess', options?.signal);
    validateInferenceOptions(options, true);
    assertHardDeadlineUnsupported(options.hardDeadlineMs, 'preprocess');
    const compiled = options.schema === undefined ? undefined : compileStructuredSchema(options.schema, options.structuredMode);
    return this.run('preprocess', options.signal, async (signal) => {
      const request = { ...options, signal };
      const plan = inferenceChat(request).images.length
        ? await this.use(signal, (engine) => engine.planInference(request, compiled))
        : await planTextInference(request, { profile: this.options.modelProfile, model: this.options.model }, compiled);
      return { ...plan, execution: this.execution() };
    });
  }
  planReport(input: string | Page, options: DescribeOptions = {}): Promise<ReportPlan> {
    this.checkRequest('preprocess', options?.signal);
    validateDescribeOptions(options);
    assertHardDeadlineUnsupported(options.hardDeadlineMs, 'preprocess');
    const captured = this.captureSource(input);
    return this.run('preprocess', options.signal, async (signal) => {
      const source = await captured; if (!source.ok) throw source.error;
      const extracted = typeof source.value === 'string' ? await extractPage(source.value, { ...options, signal }, { policy: this.options.policy, offline: this.options.localFilesOnly }) : source.value;
      const page = await selectPage(extracted, options.sources, signal);
      if (options.mode === 'extractive') return planExtractiveReport(page, { ...options, signal }, { selectionApplied: true });
      return this.use(signal, (engine) => generateReportPlan(engine, page, { ...options, signal }, { selectionApplied: true }));
    });
  }
  session(options: SessionOptions = {}): ConversationSession {
    this.checkRequest('generate');
    const profile = getRegisteredModelProfile(this.options.modelProfile, this.options.model);
    return createConversationSession(this, { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype }, options);
  }
  ask(input: string | Page, question: string, options: AskOptions = {}): Promise<DocumentAnswer> {
    this.checkRequest('generate', options?.signal);
    assertHardDeadlineUnsupported(options.hardDeadlineMs);
    const captured = this.captureSource(input);
    return this.run('generate', options.signal, async (signal) => {
      const source = await captured; if (!source.ok) throw source.error;
      return askDocument(source.value, question,
        (request) => this.use(signal, (engine) => engine.inferStructured({ ...request, signal })),
        { ...options, signal }, { policy: this.options.policy, offline: this.options.localFilesOnly });
    });
  }
  describe(input: string | Page, options: DescribeOptions & { format: 'markdown' }): Promise<string>;
  describe(input: string | Page, options?: DescribeOptions & { format?: 'json' }): Promise<StructuredReport>;
  describe(input: string | Page, options: DescribeOptions): Promise<StructuredReport | string>;
  async describe(input: string | Page, options: DescribeOptions = {}): Promise<StructuredReport | string> {
    this.checkRequest('report', options?.signal);
    validateDescribeOptions(options);
    assertHardDeadlineUnsupported(options.hardDeadlineMs, 'report');
    const started = performance.now();
    const captured = this.captureSource(input);
    const deadline = new AbortController(); let timer: Parameters<typeof globalThis.clearTimeout>[0];
    const duration = options.budget?.maxDurationMs;
    if (duration !== undefined && Number.isSafeInteger(duration) && duration > 0 && duration <= 2_147_483_647) {
      const remaining = duration - (options.resume?.elapsedMs ?? 0) - (performance.now() - started);
      const exhausted = () => deadline.abort(new NekoError('Report aggregate duration budget exhausted', 'report', 'BUDGET_EXCEEDED'));
      if (remaining <= 0) exhausted(); else timer = setTimeout(exhausted, Math.ceil(remaining));
    }
    const requestSignal = AbortSignal.any([deadline.signal, ...(options.signal ? [options.signal] : [])]);
    return this.run('report', requestSignal, async (signal, queueWaitMs) => {
      const source = await captured; if (!source.ok) throw source.error;
      const sourceInput = source.value;
      const extracted = typeof sourceInput === 'string' ? await atStage('extract', signal, () => extractPage(sourceInput, { ...options, signal }, { policy: this.options.policy, offline: this.options.localFilesOnly })) : sourceInput;
      const page = await atStage('extract', signal, () => selectPage(extracted, options.sources, signal));
      if (options.mode === 'extractive') {
        const profile = getRegisteredModelProfile(this.options.modelProfile, this.options.model);
        return generateExtractiveReport(page, { ...options, signal }, {
          model: { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype },
          execution: this.execution(), startedAt: started, queueWaitMs, selectionApplied: true,
        });
      }
      const loadStarted = performance.now();
      return this.use(signal, (engine) => generateReport(engine, page, { ...options, signal }, { startedAt: started, queueWaitMs, execution: this.execution(), selectionApplied: true, requestLoadMs: performance.now() - loadStarted }));
    }, options.signal).catch((cause: unknown) => {
      if (deadline.signal.aborted && cause instanceof NekoError && cause.code === 'ABORTED') {
        const error = deadline.signal.reason as NekoError;
        const diagnostic = getGenerationDiagnostic(cause);
        if (diagnostic) attachGenerationDiagnostic(error, { ...diagnostic, code: error.code });
        throw cause instanceof ReportError ? new ReportError(error, cause.checkpoint, cause.partial) : error;
      }
      throw cause;
    }).finally(() => { clearTimeout(timer); });
  }
  load(signal?: AbortSignal): Promise<RuntimeReadiness> { return this.run('load', signal, (abort) => this.use(abort, async (engine) => engine.readiness())); }
  warmup(signal?: AbortSignal): Promise<RuntimeReadiness> { return this.run('load', signal, (abort) => this.use(abort, (engine) => engine.warmup(abort))); }
  async runtimeStatus(): Promise<RuntimeReadiness | null> { if (this.closed) throw new NekoError('Neko is disposed', 'load', 'DISPOSED'); return this.engines.status().loaded ? this.readiness : null; }
  async queueStatus(): Promise<QueueStatus> { if (this.closed) throw new NekoError('Neko is disposed', 'generate', 'DISPOSED'); return this.admission.status(); }
  async diagnostics(): Promise<RuntimeDiagnostics> {
    this.checkRequest('load');
    return createRuntimeDiagnostics({ execution: this.execution(), readiness: await this.runtimeStatus(), backend: await this.backend.current(), engine: this.engines.status(), queue: this.admission.status() });
  }
  async health(options: HealthOptions = {}): Promise<RuntimeHealth> {
    this.checkRequest('load', options.signal);
    validateHealthOptions(options);
    return { healthy: true, execution: this.execution(), roundTripMs: null, checkedAt: Date.now() };
  }
  async restart(): Promise<void> {
    this.checkRequest('create');
    throw new NekoError('Worker recreation requires execution: worker; inline owners must be disposed and created explicitly', 'create', 'UNSUPPORTED_BACKEND');
  }
  dispose(): Promise<void> {
    if (!this.disposePromise) {
      this.closed = true; this.lifetime.abort(new Error('Neko disposed'));
      this.disposePromise = atStage('dispose', undefined, async () => { try { await this.admission.idle(); await this.engines.release(); this.readiness = null; } finally { this.restore(); } });
    }
    return this.disposePromise;
  }
  [Symbol.asyncDispose](): Promise<void> { return this.dispose(); }
}

function capturePolicy(policy?: ResourcePolicy): ResourcePolicy {
  if (policy !== undefined && (typeof policy !== 'object' || policy === null || Array.isArray(policy))) throw new TypeError('policy must be a resource policy record');
  const owned: ResourcePolicy = Object.create(null);
  const network = policy?.network;
  const localFiles = policy?.localFiles;
  if (network !== undefined) {
    if (typeof network !== 'function') throw new TypeError('policy.network must be a function');
    owned.network = network.bind(policy);
  }
  if (localFiles !== undefined) {
    if (typeof localFiles !== 'function') throw new TypeError('policy.localFiles must be a function');
    owned.localFiles = localFiles.bind(policy);
  }
  return Object.freeze(owned);
}

/** Inline instances own process-global hooks; worker instances own independent runtime realms. */
export async function createNeko(configuration: NekoOptions = {}): Promise<Neko> {
  return atStage('create', configuration.signal, async () => {
    const profile = getRegisteredModelProfile(configuration.modelProfile, configuration.model);
    if (configuration.execution !== undefined && configuration.execution !== 'inline' && configuration.execution !== 'worker') throw new TypeError('execution must be inline or worker');
    const { policy, modelSource, ...settings } = configuration;
    const source = captureModelSource(modelSource);
    const options: OwnedNekoOptions = { ...settings, model: profile.id, modelProfile: profile.profile, ...(source === undefined ? {} : { modelSource: source }), policy: capturePolicy(policy), localFilesOnly: settings.localFilesOnly ?? false, cache: { ...settings.cache }, queue: { ...settings.queue }, ...(settings.reuseCache === undefined ? {} : { reuseCache: Object.freeze({ ...settings.reuseCache }) }) };
    if (options.execution === 'worker') return createWorkerClient(options);
    const admission = new RequestQueue(options.queue?.maxPending);
    const backend = await atStage('backend', options.signal, () => inspectBackend(options.device ?? 'webgpu', options.modelProfile));
    if (backend.runtime === 'browser' && backend.device === 'cpu') throw new NekoError('CPU/WASM cannot execute this pinned model: GatherBlockQuantized(1) is unavailable. Use a supported WebGPU browser; no automatic fallback is performed.', 'backend', 'UNSUPPORTED_BACKEND');
    let installation: VerifiedCacheInstallation | undefined; let restore: (() => void) | undefined;
    try {
      installation = await installVerifiedCache({ ...(options.cacheDir ? { cacheDir: options.cacheDir } : {}), ...(options.modelSource === undefined ? {} : { modelSource: options.modelSource }), localFilesOnly: options.localFilesOnly, model: options.model, profile: options.modelProfile, policy: options.policy, ...(options.downloadConcurrency === undefined ? {} : { downloadConcurrency: options.downloadConcurrency }), ...(options.resumeDownloads === undefined ? {} : { resumeDownloads: options.resumeDownloads }), ...(options.onCacheProgress ?? options.progressCallback ? { onProgress: options.onCacheProgress ?? options.progressCallback } : {}) });
      const cache = installation; const previousWasmCache = env.useWasmCache; const wasm = env.backends.onnx.wasm;
      const previous = wasm && { paths: wasm.wasmPaths, binary: wasm.wasmBinary, threads: wasm.numThreads, proxy: wasm.proxy };
      let moduleUrl: string | undefined;
      restore = () => {
        env.useWasmCache = previousWasmCache;
        if (wasm && previous) { for (const [key, value] of Object.entries({ wasmPaths: previous.paths, wasmBinary: previous.binary, numThreads: previous.threads, proxy: previous.proxy })) { if (value === undefined) delete (wasm as unknown as Record<string, unknown>)[key]; else (wasm as unknown as Record<string, unknown>)[key] = value; } }
        if (moduleUrl) URL.revokeObjectURL(moduleUrl); cache.restore();
      };
      const configureRuntime = async (signal?: AbortSignal) => {
        if (backend.runtime !== 'browser') return;
        if (!wasm) throw new Error('ONNX WASM environment configuration is unavailable');
        wasm.numThreads = 1; wasm.proxy = false; env.useWasmCache = true;
        const binaryUrl = new URL(options.wasmPaths?.wasm ?? './assets/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url);
        await authorizeNetwork(options.policy, binaryUrl, 'runtime', options.localFilesOnly, undefined, signal);
        const binary = await env.fetch(binaryUrl.href, { signal }); if (!binary.ok) throw new Error(`Runtime binary download failed (${binary.status})`);
        wasm.wasmBinary = new Uint8Array(await binary.arrayBuffer());
        // Embedded pinned JS avoids unchecked dynamic network imports; custom modules are fetched through the policy broker first.
        if (options.wasmPaths?.mjs) {
          const destination = new URL(options.wasmPaths.mjs, import.meta.url); await authorizeNetwork(options.policy, destination, 'runtime', options.localFilesOnly, undefined, signal);
          const response = await env.fetch(destination.href, { signal }); if (!response.ok) throw new Error(`Runtime module download failed (${response.status})`);
          if (moduleUrl) URL.revokeObjectURL(moduleUrl); moduleUrl = URL.createObjectURL(new Blob([await response.arrayBuffer()], { type: 'text/javascript' })); wasm.wasmPaths = { mjs: moduleUrl };
        } else delete wasm.wasmPaths;
        signal?.throwIfAborted();
      };
      options.signal?.throwIfAborted();
      const engines = new EngineCache((signal) => cache.withSignal(signal, async () => {
        const started = performance.now(); await configureRuntime(signal); await atStage('load', signal, () => cache.prefetch(signal));
        return VisionEngine.load(backend, options.localFilesOnly, signal, options.progressCallback, options.profilePrefix, started, { model: options.model, profile: options.modelProfile, policy: options.policy, ...(options.reuseCache ? { reuseCacheLimits: options.reuseCache } : {}) });
      }), options.cache?.engine ?? true, options.cache?.engineTtlMs ?? 30 * 60_000);
      return new LocalNeko(engines, cache, restore, backend, options, admission);
    } catch (error) { if (restore) restore(); else installation?.restore(); throw error; }
  }, (instance) => instance.dispose());
}
export const Neko = { create: createNeko };
