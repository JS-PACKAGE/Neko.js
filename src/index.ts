import { env } from '@huggingface/transformers';
import { EngineCache } from './cache/engine.js';
import { captureModelSource, installVerifiedCache, type VerifiedCacheInstallation, type ModelSource } from './cache/model.js';
import { inspectBackend, type BackendInfo, type BackendDevice } from './backend/index.js';
import { VisionEngine, type InferOptions, type InferenceResult } from './core/engine.js';
import { generateReport, type DescribeOptions } from './report/generate.js';
import type { StructuredReport } from './types.js';
import { atStage, NekoError, type ErrorStage } from './errors.js';
import { getModelProfile, type ModelProfileId } from './cache/manifest.js';
import { authorizeNetwork, type ResourcePolicy } from './web/policy.js';

export * from './types.js';
export * from './web/index.js';
export * from './report/index.js';
export * from './cache/manifest.js';
export * from './backend/index.js';
export * from './errors.js';
export type { InferOptions, InferenceResult, ModelIdentity } from './core/engine.js';
export type { DescribeOptions } from './report/generate.js';
export type { ModelCacheStatus, CacheProgress, ModelSource } from './cache/model.js';
export type { EngineCacheStatus } from './cache/engine.js';

export interface NekoOptions {
  device?: BackendDevice;
  modelProfile?: ModelProfileId;
  modelSource?: ModelSource;
  policy?: ResourcePolicy;
  cacheDir?: string;
  cache?: { engine?: boolean; engineTtlMs?: number };
  localFilesOnly?: boolean;
  signal?: AbortSignal;
  progressCallback?: (event: unknown) => void;
  wasmPaths?: { wasm: string; mjs: string };
  profilePrefix?: string;
}

type OwnedNekoOptions = NekoOptions & { policy: ResourcePolicy; modelProfile: ModelProfileId; localFilesOnly: boolean };

/** One owner of Transformers.js's process-global runtime hooks; model loading is lazy. */
export class Neko {
  private closed = false;
  private queue: Promise<unknown> = Promise.resolve();
  private readonly lifetime = new AbortController();
  private disposePromise?: Promise<void>;
  readonly cache;
  readonly backend;

  private constructor(private readonly engines: EngineCache<VisionEngine>, private readonly installation: VerifiedCacheInstallation, private readonly restore: () => void, backend: BackendInfo, private readonly options: OwnedNekoOptions) {
    this.cache = {
      model: {
        prefetch: (signal?: AbortSignal) => this.run('cache', signal, (abort) => installation.prefetch(abort)),
        status: (signal?: AbortSignal) => this.run('cache', signal, (abort) => installation.status(abort)),
        clear: (signal?: AbortSignal) => this.run('cache', signal, async (abort) => { await engines.release(); abort.throwIfAborted(); await installation.clear(abort); }),
      },
      engine: {
        status: () => engines.status(),
        release: () => this.run('cache', undefined, async () => { await engines.release(); }),
      },
    };
    this.backend = {
      current: () => backend,
      detect: (device: BackendDevice = backend.device, signal?: AbortSignal) => this.run('backend', signal, () => inspectBackend(device, options.modelProfile)),
    };
  }

  static async create(configuration: NekoOptions = {}): Promise<Neko> {
    return atStage('create', configuration.signal, async () => {
      const profile = getModelProfile(configuration.modelProfile);
      const { policy, modelSource, ...settings } = configuration;
      const source = captureModelSource(modelSource);
      const options: OwnedNekoOptions = { ...settings, modelProfile: profile.profile, ...(source === undefined ? {} : { modelSource: source }), policy: capturePolicy(policy), localFilesOnly: settings.localFilesOnly ?? false, cache: { ...settings.cache } };
      const backend = await atStage('backend', options.signal, () => inspectBackend(options.device ?? 'webgpu', options.modelProfile));
      if (backend.runtime === 'browser' && backend.device === 'cpu') throw new NekoError('CPU/WASM cannot execute this pinned model: GatherBlockQuantized(1) is unavailable. Use a supported WebGPU browser; no automatic fallback is performed.', 'backend', 'UNSUPPORTED_BACKEND');
      let installation: VerifiedCacheInstallation | undefined; let restore: (() => void) | undefined;
      try {
        installation = await installVerifiedCache({ ...(options.cacheDir ? { cacheDir: options.cacheDir } : {}), ...(options.modelSource === undefined ? {} : { modelSource: options.modelSource }), localFilesOnly: options.localFilesOnly ?? false, profile: options.modelProfile, policy: options.policy, ...(options.progressCallback ? { onProgress: options.progressCallback } : {}) });
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
          return VisionEngine.load(backend, options.localFilesOnly ?? false, signal, options.progressCallback, options.profilePrefix, started, { profile: options.modelProfile, policy: options.policy });
        }), options.cache?.engine ?? true, options.cache?.engineTtlMs ?? 30 * 60_000);
        return new Neko(engines, cache, restore, backend, options);
      } catch (error) { if (restore) restore(); else installation?.restore(); throw error; }
    }, (instance) => instance.dispose());
  }

  private run<T>(stage: ErrorStage, signal: AbortSignal | undefined, operation: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new NekoError('Neko is disposed', stage, 'DISPOSED'));
    const abort = AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])]);
    const work = this.queue.catch(() => undefined).then(() => atStage(stage, abort, () => this.installation.withSignal(abort, () => operation(abort))));
    this.queue = work;
    return work;
  }
  infer(options: InferOptions): Promise<InferenceResult> {
    return this.run('generate', options.signal, (signal) => this.engines.use((engine) => engine.infer({ ...options, signal, _policy: this.options.policy, _offline: this.options.localFilesOnly }), signal));
  }
  describe(input: string, options: DescribeOptions & { format: 'markdown' }): Promise<string>;
  describe(input: string, options?: DescribeOptions & { format?: 'json' }): Promise<StructuredReport>;
  describe(input: string, options: DescribeOptions): Promise<StructuredReport | string>;
  describe(input: string, options: DescribeOptions = {}): Promise<StructuredReport | string> {
    return this.run('report', options.signal, (signal) => this.engines.use((engine) => generateReport(engine, input, { ...options, signal, _policy: this.options.policy, _offline: this.options.localFilesOnly }), signal));
  }
  dispose(): Promise<void> {
    if (!this.disposePromise) {
      this.closed = true;
      this.lifetime.abort(new Error('Neko disposed'));
      this.disposePromise = atStage('dispose', undefined, async () => {
        try { await this.queue.catch(() => undefined); await this.engines.release(); }
        finally { this.restore(); }
      });
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

export function createNeko(options: NekoOptions = {}): Promise<Neko> { return Neko.create(options); }
