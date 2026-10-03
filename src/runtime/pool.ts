import type { NekoOptions } from '../index.js';
import type { InferOptions, InferenceResult } from '../core/engine.js';
import { validateInferenceOptions } from '../core/preflight.js';
import { NekoError } from '../errors.js';

export interface NekoPoolWorker { infer(options: InferOptions): Promise<InferenceResult>; dispose(): Promise<void>; }
export type NekoPoolFactory = (options: NekoOptions & { execution: 'worker' }, index: number) => Promise<NekoPoolWorker>;
export interface PoolWorkerOptions {
  options?: Omit<NekoOptions, 'execution' | 'signal'>;
  /** Application-declared reservation, not measured memory or an OS memory limit. */
  memoryBytes: number;
}
export interface NekoPoolOptions {
  workers: readonly PoolWorkerOptions[];
  maxWorkers?: number;
  maxPending?: number;
  budget: { memoryBytes: number };
  signal?: AbortSignal;
}
export interface PoolItem {
  readonly id: number;
  readonly result: Promise<InferenceResult>;
  cancel(reason?: unknown): void;
  /** Cancels this item without disposing its worker or the pool. */
  dispose(): void;
  [Symbol.dispose](): void;
}
export type PoolItemResult = { id: number; status: 'fulfilled'; value: InferenceResult } | { id: number; status: 'rejected'; error: unknown };
export interface NekoPoolStatus {
  disposed: boolean;
  pending: number;
  maxPending: number;
  admitted: number;
  rejected: number;
  workers: { index: number; state: 'idle' | 'running' | 'unavailable' | 'disposed'; requestId: number | null; memoryBytes: number }[];
  resources: { kind: 'application-declared-reservations'; memoryBytes: number; budgetMemoryBytes: number };
}
export interface NekoPool {
  submit(options: InferOptions): PoolItem;
  infer(options: InferOptions): Promise<InferenceResult>;
  /** Independent requests run concurrently across owners; this is not tensor batching. */
  inferBatch(items: readonly InferOptions[], options?: { signal?: AbortSignal }): Promise<PoolItemResult[]>;
  status(): NekoPoolStatus;
  dispose(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
}
interface Request {
  id: number;
  options: InferOptions;
  abort: AbortController;
  signal: AbortSignal;
  resolve(value: InferenceResult): void;
  reject(error: unknown): void;
  cancel(): void;
  settled: boolean;
}
interface Owner { worker: NekoPoolWorker; memoryBytes: number; active: Request | null; unavailable: boolean; operation: Promise<void> | null; }
const cancelled = (reason: unknown) => reason instanceof NekoError ? reason : new NekoError('Pool request was cancelled', 'generate', 'ABORTED', { cause: reason });
const disposed = () => new NekoError('Neko pool is disposed', 'generate', 'DISPOSED');

class WorkerPool implements NekoPool {
  private readonly pending: Request[] = [];
  private closed = false;
  private nextId = 1;
  private nextOwner = 0;
  private admitted = 0;
  private rejected = 0;
  private disposePromise?: Promise<void>;
  constructor(private readonly owners: Owner[], private readonly maxPending: number, private readonly budgetMemoryBytes: number) {}
  status(): NekoPoolStatus {
    return { disposed: this.closed, pending: this.pending.length, maxPending: this.maxPending, admitted: this.admitted, rejected: this.rejected,
      workers: this.owners.map((owner, index) => ({ index, state: this.closed ? 'disposed' : owner.unavailable ? 'unavailable' : owner.active ? 'running' : 'idle', requestId: owner.active?.id ?? null, memoryBytes: owner.memoryBytes })),
      resources: { kind: 'application-declared-reservations', memoryBytes: this.owners.reduce((sum, owner) => sum + owner.memoryBytes, 0), budgetMemoryBytes: this.budgetMemoryBytes } };
  }
  submit(options: InferOptions): PoolItem {
    const id = this.nextId++;
    const abort = new AbortController();
    const result = Promise.withResolvers<InferenceResult>();
    const item: PoolItem = { id, result: result.promise, cancel: (reason) => abort.abort(reason),
      dispose: () => abort.abort(), [Symbol.dispose]: () => abort.abort() };
    try {
      if (this.closed) throw disposed();
      validateInferenceOptions(options);
      const signal = options.signal ? AbortSignal.any([options.signal, abort.signal]) : abort.signal;
      if (signal.aborted) throw cancelled(signal.reason);
      if (this.owners.every((owner) => owner.unavailable)) throw new NekoError('All pool workers are unavailable; create a new pool explicitly', 'generate', 'WORKER_UNAVAILABLE');
      if (!this.owners.some((owner) => !owner.unavailable && !owner.active) && this.pending.length >= this.maxPending) throw new NekoError('Pool admission queue is full', 'generate', 'QUEUE_FULL');
      const request: Request = { id, options: { ...options, signal }, abort, signal, resolve: result.resolve, reject: result.reject, settled: false, cancel: () => {
        const index = this.pending.indexOf(request);
        if (index !== -1) this.pending.splice(index, 1);
        this.settle(request, false, cancelled(signal.reason));
      } };
      this.admitted++;
      signal.addEventListener('abort', request.cancel, { once: true });
      this.pending.push(request);
      this.pump();
    } catch (error) { this.rejected++; result.reject(error); }
    return item;
  }
  infer(options: InferOptions): Promise<InferenceResult> { return this.submit(options).result; }
  async inferBatch(items: readonly InferOptions[], options: { signal?: AbortSignal } = {}): Promise<PoolItemResult[]> {
    if (!Array.isArray(items)) throw new NekoError('Batch items must be an array', 'preprocess', 'INVALID_INPUT');
    if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) throw new NekoError('Batch signal must be an AbortSignal', 'preprocess', 'INVALID_INPUT');
    return Promise.all(items.map((item) => {
      const signal = options.signal ? item?.signal instanceof AbortSignal ? AbortSignal.any([item.signal, options.signal]) : options.signal : item?.signal;
      const request = this.submit(item?.signal !== undefined && !(item.signal instanceof AbortSignal) ? item : { ...item, ...(signal ? { signal } : {}) });
      return request.result.then<PoolItemResult, PoolItemResult>((value) => ({ id: request.id, status: 'fulfilled', value }), (error: unknown) => ({ id: request.id, status: 'rejected', error }));
    }));
  }
  private settle(request: Request, success: boolean, value: unknown): void {
    if (request.settled) return;
    request.settled = true;
    request.signal.removeEventListener('abort', request.cancel);
    if (success) request.resolve(value as InferenceResult); else request.reject(value);
  }
  private pump(): void {
    if (this.closed) return;
    while (this.pending.length) {
      let selected = -1;
      for (let offset = 0; offset < this.owners.length; offset++) {
        const index = (this.nextOwner + offset) % this.owners.length;
        const owner = this.owners[index]!;
        if (!owner.unavailable && !owner.active) { selected = index; break; }
      }
      if (selected === -1) {
        if (this.owners.every((owner) => owner.unavailable)) for (const request of this.pending.splice(0)) this.settle(request, false, new NekoError('All pool workers are unavailable; requests were not replayed', 'generate', 'WORKER_UNAVAILABLE'));
        return;
      }
      this.nextOwner = (selected + 1) % this.owners.length;
      const owner = this.owners[selected]!;
      const request = this.pending.shift()!;
      if (request.signal.aborted) { this.settle(request, false, cancelled(request.signal.reason)); continue; }
      owner.active = request;
      // Cancellation settles the caller, but the slot remains owned until work actually settles.
      owner.operation = Promise.resolve().then(() => {
        request.signal.throwIfAborted();
        return owner.worker.infer(request.options);
      }).then((value) => {
        owner.active = null; owner.operation = null;
        this.settle(request, true, value);
        this.pump();
      }, (error: unknown) => {
        owner.active = null; owner.operation = null;
        if (error instanceof NekoError && (error.code === 'WORKER_UNAVAILABLE' || error.code === 'DEADLINE_EXCEEDED')) owner.unavailable = true;
        this.settle(request, false, request.signal.aborted ? cancelled(request.signal.reason) : error);
        this.pump();
      });
    }
  }
  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.closed = true;
    for (const request of this.pending.splice(0)) { this.settle(request, false, disposed()); request.abort.abort(disposed()); }
    for (const owner of this.owners) owner.active?.abort.abort(disposed());
    this.disposePromise = (async () => {
      // Worker disposal terminates ownership; do not wait for uninterruptible work before requesting it.
      const results = await Promise.allSettled(this.owners.map((owner) => Promise.resolve().then(() => owner.worker.dispose())));
      await Promise.all(this.owners.map((owner) => owner.operation));
      const failures = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (failures.length) throw new AggregateError(failures.map((result) => result.reason), 'Pool worker disposal failed');
    })();
    return this.disposePromise;
  }
  [Symbol.asyncDispose](): Promise<void> { return this.dispose(); }
}

export async function createNekoPool(options: NekoPoolOptions, factory: NekoPoolFactory): Promise<NekoPool> {
  const maxWorkers = options.maxWorkers ?? 4;
  const maxPending = options.maxPending ?? 8;
  if (!Number.isSafeInteger(maxWorkers) || maxWorkers < 1 || maxWorkers > 32) throw new RangeError('maxWorkers must be between 1 and 32');
  if (!Array.isArray(options.workers) || !options.workers.length || options.workers.length > maxWorkers) throw new RangeError('workers must contain between 1 and maxWorkers owners');
  if (!Number.isSafeInteger(maxPending) || maxPending < 0 || maxPending > 10000) throw new RangeError('maxPending must be between 0 and 10000');
  const budget = options.budget?.memoryBytes;
  if (!Number.isSafeInteger(budget) || budget < 1) throw new RangeError('budget.memoryBytes must be a positive safe integer');
  let reserved = 0;
  const configs = options.workers.map((config) => {
    if (!Number.isSafeInteger(config.memoryBytes) || config.memoryBytes < 1) throw new RangeError('Each worker memoryBytes reservation must be a positive safe integer');
    if (config.options && ('execution' in config.options || 'signal' in config.options)) throw new TypeError('Worker execution and creation signal are owned by the pool');
    reserved += config.memoryBytes;
    if (!Number.isSafeInteger(reserved) || reserved > budget) throw new NekoError('Declared worker memory reservations exceed the pool budget', 'create', 'BUDGET_EXCEEDED');
    return { options: { ...config.options }, memoryBytes: config.memoryBytes };
  });
  if (typeof factory !== 'function') throw new TypeError('Pool factory must be a function');
  options.signal?.throwIfAborted();
  const creation = new AbortController();
  const signal = options.signal ? AbortSignal.any([options.signal, creation.signal]) : creation.signal;
  const results = await Promise.allSettled(configs.map(async (config, index) => {
    try {
      const worker = await factory({ ...config.options, execution: 'worker', signal }, index);
      if (!worker || typeof worker.infer !== 'function' || typeof worker.dispose !== 'function') throw new TypeError('Factory must return an isolated worker owner');
      return worker;
    } catch (cause) { creation.abort(cause); throw cause; }
  }));
  const workers = results.filter((result): result is PromiseFulfilledResult<NekoPoolWorker> => result.status === 'fulfilled').map((result) => result.value);
  const failure = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
  const duplicate = new Set(workers).size !== workers.length;
  if (failure || signal.aborted || duplicate) {
    const cleanup = await Promise.allSettled([...new Set(workers)].map((worker) => Promise.resolve().then(() => worker.dispose())));
    const error: unknown = failure?.reason ?? (duplicate ? new NekoError('Factory returned a shared owner; pool workers must be isolated', 'create', 'RUNTIME_BUSY') : new NekoError('Pool creation was cancelled', 'create', 'ABORTED', { cause: signal.reason }));
    const cleanupErrors = cleanup.filter((result): result is PromiseRejectedResult => result.status === 'rejected').map((result) => result.reason);
    if (cleanupErrors.length) throw new AggregateError([error, ...cleanupErrors], 'Pool creation and cleanup failed');
    throw error;
  }
  return new WorkerPool(workers.map((worker, index) => ({ worker, memoryBytes: configs[index]!.memoryBytes, active: null, unavailable: false, operation: null })), maxPending, budget);
}
