import { env } from '@huggingface/transformers';
import { EngineCache, type EngineCacheStatus } from './cache/engine.js';
import { installVerifiedCache } from './cache/model.js';
import { inspectBackend } from './backend/index.js';
import { VisionEngine, type InferOptions, type PrototypeResult } from './core/prototype.js';

export * from './types.js';
export * from './web/index.js';
export * from './report/index.js';
export * from './cache/manifest.js';
export * from './backend/index.js';
export type { InferOptions, PrototypeResult } from './core/prototype.js';

export interface PrototypeOptions {
  device?: 'webgpu' | 'cpu';
  cacheDir?: string;
  localFilesOnly?: boolean;
  progressCallback?: (event: unknown) => void;
  wasmPaths?: { wasm: string; mjs: string; };
  profilePrefix?: string;
}

/** First-stage real image/prompt prototype, not the full website-report API. */
export class VisionPrototype {
  private closed = false;
  private constructor(private readonly engines: EngineCache<VisionEngine>, private readonly restore: () => void, private readonly prefetchModel: (signal?: AbortSignal) => Promise<void>) {}

  static async create(options: PrototypeOptions = {}): Promise<VisionPrototype> {
    const backend = await inspectBackend(options.device ?? 'webgpu');
    const cache = await installVerifiedCache({ ...(options.cacheDir ? { cacheDir: options.cacheDir } : {}), localFilesOnly: options.localFilesOnly ?? false, ...(options.progressCallback ? { onProgress: options.progressCallback } : {}) });
    const previousWasmCache = env.useWasmCache;
    const wasm = env.backends.onnx.wasm;
    const previousWasmPaths = wasm?.wasmPaths;
    const restore = () => {
      env.useWasmCache = previousWasmCache;
      if (wasm) {
        if (previousWasmPaths === undefined) delete wasm.wasmPaths;
        else wasm.wasmPaths = previousWasmPaths;
      }
      cache.restore();
    };
    const engines = new EngineCache(async () => {
      const loadStarted = performance.now();
      await cache.prefetch();
      return VisionEngine.load(backend, options.localFilesOnly ?? false, options.progressCallback, options.profilePrefix, loadStarted);
    });
    try {
      if (backend.runtime === 'browser') {
        if (!wasm) throw new Error('ONNX WASM environment configuration is unavailable');
        env.useWasmCache = true;
        wasm.wasmPaths = options.wasmPaths ?? {
          wasm: new URL('./assets/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url).href,
          mjs: new URL('./assets/ort-wasm-simd-threaded.asyncify.mjs', import.meta.url).href,
        };
      }
      await engines.use(async () => undefined);
      return new VisionPrototype(engines, restore, cache.prefetch);
    } catch (error) { try { await engines.release(); } finally { restore(); } throw error; }
  }

  infer(options: InferOptions): Promise<PrototypeResult> {
    if (this.closed) return Promise.reject(new Error('Prototype is disposed'));
    return this.engines.use((engine) => engine.infer(options));
  }
  prefetch(signal?: AbortSignal): Promise<void> {
    if (this.closed) return Promise.reject(new Error('Prototype is disposed'));
    return this.engines.use(async () => this.prefetchModel(signal));
  }
  status(): EngineCacheStatus { return this.engines.status(); }
  async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    try { await this.engines.release(); } finally { this.restore(); }
  }
  async [Symbol.asyncDispose](): Promise<void> { await this.dispose(); }
}

export function createPrototype(options: PrototypeOptions = {}): Promise<VisionPrototype> { return VisionPrototype.create(options); }
