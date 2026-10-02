import { atStage, NekoError } from './errors.js';
import type { ErrorStage } from './errors.js';

export interface QueueStatus { running: ErrorStage | null; pending: number; maxPending: number; admitted: number; rejected: number; }
interface Request {
  stage: ErrorStage;
  signal: AbortSignal;
  started: number;
  operation: (signal: AbortSignal, queueWaitMs: number) => Promise<unknown>;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  cancel: () => void;
}
export class RequestQueue {
  private readonly pending: Request[] = [];
  private active: ErrorStage | null = null;
  private admitted = 0;
  private rejected = 0;
  private readonly idleWaiters: (() => void)[] = [];
  constructor(private readonly maxPending = 8) {
    if (!Number.isSafeInteger(maxPending) || maxPending < 0 || maxPending > 10000) throw new RangeError('queue.maxPending must be between 0 and 10000');
  }
  status(): QueueStatus { return { running: this.active, pending: this.pending.length, maxPending: this.maxPending, admitted: this.admitted, rejected: this.rejected }; }
  idle(): Promise<void> {
    if (this.active === null) return Promise.resolve();
    const { promise, resolve } = Promise.withResolvers<void>();
    this.idleWaiters.push(resolve); return promise;
  }
  enqueue<T>(stage: ErrorStage, signal: AbortSignal, operation: (signal: AbortSignal, queueWaitMs: number) => Promise<T>): Promise<T> {
    if (signal.aborted) return Promise.reject(new NekoError('Operation was cancelled', stage, 'ABORTED', { cause: signal.reason }));
    if (this.active !== null && this.pending.length >= this.maxPending) { this.rejected++; return Promise.reject(new NekoError('Request admission queue is full', stage, 'QUEUE_FULL')); }
    this.admitted++;
    const { promise, resolve, reject } = Promise.withResolvers<T>();
    const request: Request = { stage, signal, started: performance.now(), operation, resolve: (value) => resolve(value as T), reject, cancel: () => {
      const index = this.pending.indexOf(request);
      if (index === -1) return;
      this.pending.splice(index, 1);
      signal.removeEventListener('abort', request.cancel);
      reject(new NekoError('Queued operation was cancelled', stage, 'ABORTED', { cause: signal.reason }));
    } };
    if (this.active === null) this.start(request);
    else { this.pending.push(request); signal.addEventListener('abort', request.cancel, { once: true }); }
    return promise;
  }
  private start(request: Request): void {
    this.active = request.stage;
    request.signal.removeEventListener('abort', request.cancel);
    void atStage(request.stage, request.signal, () => request.operation(request.signal, performance.now() - request.started)).finally(() => {
      const next = this.pending.shift();
      if (next) this.start(next);
      else { this.active = null; for (const resolve of this.idleWaiters.splice(0)) resolve(); }
    }).then(request.resolve, request.reject);
  }
}
