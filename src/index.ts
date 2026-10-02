import { env } from '@huggingface/transformers';
import { EngineCache } from './cache/engine.js';
import { installVerifiedCache, type VerifiedCacheInstallation } from './cache/model.js';
import { inspectBackend, type BackendInfo, type BackendDevice } from './backend/index.js';
import { VisionEngine, type InferOptions, type InferenceResult } from './core/engine.js';
import { generateReport, type DescribeOptions } from './report/generate.js';
import type { StructuredReport } from './types.js';
import { atStage, NekoError, type ErrorStage } from './errors.js';

export * from './types.js';
export * from './web/index.js';
export * from './report/index.js';
export * from './cache/manifest.js';
export * from './backend/index.js';
export * from './errors.js';
export type { InferOptions, InferenceResult } from './core/engine.js';
export type { DescribeOptions } from './report/generate.js';
export type { ModelCacheStatus, CacheProgress } from './cache/model.js';
export type { EngineCacheStatus } from './cache/engine.js';

export interface NekoOptions {
  device?: BackendDevice;
  cacheDir?: string;
  cache?: { engine?: boolean; engineTtlMs?: number };
  localFilesOnly?: boolean;
  signal?: AbortSignal;
  progressCallback?: (event: unknown) => void;
  wasmPaths?: { wasm: string; mjs: string };
  profilePrefix?: string;
}

/** One owner of Transformers.js's process-global runtime hooks; model loading is lazy. */
export class Neko {
  private closed = false;
  private queue: Promise<unknown> = Promise.resolve();
  private readonly lifetime = new AbortController();
  private disposePromise?: Promise<void>;
  readonly cache;
  readonly backend;

  private constructor(private readonly engines: EngineCache<VisionEngine>, private readonly installation: VerifiedCacheInstallation, private readonly restore: () => void, backend: BackendInfo) {
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
      detect: (device: BackendDevice = backend.device, signal?: AbortSignal) => this.run('backend', signal, () => inspectBackend(device)),
    };
  }

  static async create(options: NekoOptions = {}): Promise<Neko> {
    return atStage('create', options.signal, async () => {
      const backend = await atStage('backend', options.signal, () => inspectBackend(options.device ?? 'webgpu'));
      if (backend.runtime === 'browser' && backend.device === 'cpu') throw new NekoError('CPU/WASM cannot execute this pinned model: GatherBlockQuantized(1) is unavailable. Use a supported WebGPU browser; no automatic fallback is performed.', 'backend', 'UNSUPPORTED_BACKEND');
      let installation: VerifiedCacheInstallation | undefined;
      let restore: (() => void) | undefined;
      try {
        installation = await installVerifiedCache({ ...(options.cacheDir ? { cacheDir: options.cacheDir } : {}), localFilesOnly: options.localFilesOnly ?? false, ...(options.progressCallback ? { onProgress: options.progressCallback } : {}) });
        const cache = installation;
        const previousWasmCache = env.useWasmCache;
        const wasm = env.backends.onnx.wasm;
        const previousWasmPaths = wasm?.wasmPaths;
        restore = () => {
          env.useWasmCache = previousWasmCache;
          if (wasm) { if (previousWasmPaths === undefined) delete wasm.wasmPaths; else wasm.wasmPaths = previousWasmPaths; }
          cache.restore();
        };
        if (backend.runtime === 'browser') {
          if (!wasm) throw new Error('ONNX WASM environment configuration is unavailable');
          env.useWasmCache = true;
          wasm.wasmPaths = options.wasmPaths ?? { wasm: new URL('./assets/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url).href, mjs: new URL('./assets/ort-wasm-simd-threaded.asyncify.mjs', import.meta.url).href };
        }
        options.signal?.throwIfAborted();
        const engines = new EngineCache((signal) => cache.withSignal(signal, async () => {
          const started = performance.now();
          await atStage('load', signal, () => cache.prefetch(signal));
          return VisionEngine.load(backend, options.localFilesOnly ?? false, signal, options.progressCallback, options.profilePrefix, started);
        }), options.cache?.engine ?? true, options.cache?.engineTtlMs ?? 30 * 60_000);
        return new Neko(engines, cache, restore, backend);
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
    return this.run('generate', options.signal, (signal) => this.engines.use((engine) => engine.infer({ ...options, signal }), signal));
  }
  describe(input: string, options: DescribeOptions & { format: 'markdown' }): Promise<string>;
  describe(input: string, options?: DescribeOptions & { format?: 'json' }): Promise<StructuredReport>;
  describe(input: string, options: DescribeOptions): Promise<StructuredReport | string>;
  describe(input: string, options: DescribeOptions = {}): Promise<StructuredReport | string> {
    return this.run('report', options.signal, (signal) => this.engines.use((engine) => generateReport(engine, input, { ...options, signal }), signal));
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

export function createNeko(options: NekoOptions = {}): Promise<Neko> { return Neko.create(options); }
