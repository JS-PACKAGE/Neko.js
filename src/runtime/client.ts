import type { Neko, NekoOptions } from '../index.js';
import type { Page, StructuredReport } from '../types.js';
import { ReportError, validateDescribeOptions, type DescribeOptions, type ReportCheckpoint } from '../report/generate.js';
import { NekoError } from '../errors.js';
import { authorizeNetwork } from '../web/policy.js';
import { aborted, decode, encode, encodeFailure, failure, methodStage, type MainMessage, type MessagePort, type WorkerMessage, type WorkerMethod } from './protocol.js';
import type * as NodeWorkers from 'node:worker_threads';
import { workerBootstrapUrl } from './context.js';
import { validateInferenceOptions } from '../core/preflight.js';
import { compileStructuredSchema } from '../core/structured.js';

export type WorkerClient = Neko;
interface WorkerHandle extends MessagePort<WorkerMessage, MainMessage> {
  onFailure(listener: (cause: unknown) => void): () => void;
  terminate(): Promise<void>;
}
interface PendingRequest {
  method: WorkerMethod;
  resolve(value: unknown): void;
  reject(error: unknown): void;
  callbacks: number[];
  events: Promise<void>;
  settled: boolean;
  cancellation: AbortController;
  checkpoint?: ReportCheckpoint;
  localFailure?: NekoError;
  cleanup(): void;
}

async function spawnWorker(url: URL, signal?: AbortSignal): Promise<WorkerHandle> {
  if (typeof process !== 'undefined' && process.release?.name === 'node') {
    // Browser bundles cannot import the Node-only worker_threads module statically.
    const protocol = 'node:';
    const { Worker }: typeof NodeWorkers = await import(`${protocol}worker_threads`);
    // Shipped JavaScript needs no caller loaders; parent process flags may be invalid in Workers.
    const worker = new Worker(url, { execArgv: [] });
    return {
      post: (message) => worker.postMessage(message),
      listen(listener) { worker.on('message', listener); return () => { worker.off('message', listener); }; },
      onFailure(listener) {
        const exit = (code: number) => listener(new Error(`SDK worker exited unexpectedly (${code})`));
        worker.on('error', listener); worker.on('messageerror', listener); worker.on('exit', exit);
        return () => { worker.off('error', listener); worker.off('messageerror', listener); worker.off('exit', exit); };
      },
      async terminate() { await worker.terminate(); },
    };
  }
  // Execute vetted bytes, not a second native module fetch that could follow an unchecked redirect.
  const response = await fetch(url, { redirect: 'manual', ...(signal ? { signal } : {}) });
  if (response.type === 'opaqueredirect' || response.status >= 300 && response.status < 400) throw new NekoError('SDK worker bootstrap redirects are denied', 'create', 'POLICY_DENIED');
  if (!response.ok) throw new Error(`SDK worker bootstrap failed (${response.status})`);
  const objectUrl = URL.createObjectURL(new Blob([await response.arrayBuffer()], { type: 'text/javascript' }));
  let worker: Worker;
  try { worker = new Worker(objectUrl, { type: 'module', name: url.href }); }
  catch (cause) { URL.revokeObjectURL(objectUrl); throw cause; }
  return {
    post: (message) => worker.postMessage(message),
    listen(listener) {
      const receive = (event: MessageEvent<WorkerMessage>) => listener(event.data);
      worker.addEventListener('message', receive);
      return () => { worker.removeEventListener('message', receive); };
    },
    onFailure(listener) {
      const error = (event: ErrorEvent) => listener(event.error ?? new Error(event.message || 'SDK module worker failed'));
      const messageError = () => listener(new Error('SDK module worker message could not be decoded'));
      worker.addEventListener('error', error); worker.addEventListener('messageerror', messageError);
      return () => { worker.removeEventListener('error', error); worker.removeEventListener('messageerror', messageError); };
    },
    async terminate() { worker.terminate(); URL.revokeObjectURL(objectUrl); },
  };
}

class Connection {
  private nextRequest = 1;
  private nextCallback = 1;
  private readonly pending = new Map<number, PendingRequest>();
  private readonly callbacks = new Map<number, { fn: (...args: unknown[]) => unknown; key: string }>();
  private stopped = false;
  private closed = false;
  private disposePromise?: Promise<void>;
  private readonly unlisten: () => void;
  private readonly unfail: () => void;
  constructor(private readonly worker: WorkerHandle) {
    this.unlisten = worker.listen((message) => { this.receive(message); });
    this.unfail = worker.onFailure((cause) => { this.crash(cause); });
  }

  checkRequest(method: WorkerMethod, signal?: AbortSignal): void {
    if (this.stopped || this.closed && method !== 'dispose') throw new NekoError('Neko is disposed', methodStage(method), 'DISPOSED');
    if (signal instanceof AbortSignal && signal.aborted) throw aborted(method, signal.reason);
  }
  request<T>(method: WorkerMethod, args: unknown[], signal?: AbortSignal): Promise<T> {
    try { this.checkRequest(method, signal); }
    catch (cause) { return Promise.reject(cause); }
    const id = this.nextRequest++;
    return new Promise<T>((resolve, reject) => {
      const callbackIds: number[] = [];
      const abort = () => {
        const state = this.pending.get(id);
        if (!state || state.settled) return;
        const error = aborted(method, signal?.reason);
        if (method === 'describe') { state.localFailure ??= error; state.cancellation.abort(state.localFailure); }
        else this.settle(state, false, error);
        this.post({ type: 'abort', id, reason: encodeFailure(signal?.reason) });
      };
      const state: PendingRequest = { method, resolve: (value) => resolve(value as T), reject, callbacks: callbackIds, events: Promise.resolve(), settled: false, cancellation: new AbortController(), cleanup: () => { signal?.removeEventListener('abort', abort); } };
      this.pending.set(id, state);
      try {
        const value = encode(args, (callback, _mode, key) => {
          const callbackId = this.nextCallback++;
          this.callbacks.set(callbackId, { fn: callback, key }); callbackIds.push(callbackId);
          return callbackId;
        });
        signal?.addEventListener('abort', abort, { once: true });
        this.post({ type: 'request', id, method, args: value });
        if (signal?.aborted) abort();
      } catch (cause) { this.settle(state, false, failure(method, cause)); this.pending.delete(id); }
    });
  }

  private post(message: MainMessage): void {
    try { this.worker.post(message); }
    catch (cause) { this.crash(cause); }
  }
  private receive(message: WorkerMessage): void {
    if (message.type === 'callback') {
      const state = this.pending.get(message.requestId);
      const callback = this.callbacks.get(message.callbackId);
      if (!state || state.settled || state.localFailure || state.cancellation.signal.aborted || !callback) {
        this.post({ type: 'callback-result', id: message.id, ok: false, error: encodeFailure(aborted(state?.method ?? 'create', 'Request is no longer active')) });
        return;
      }
      // Messages are FIFO, but async callbacks need an explicit chain before result settlement.
      state.events = state.events.then(async () => {
        if (state.settled || state.localFailure || state.cancellation.signal.aborted) {
          this.post({ type: 'callback-result', id: message.id, ok: false, error: encodeFailure(aborted(state.method, 'Request is no longer active')) });
          return;
        }
        let cancel: (() => void) | undefined;
        try {
          const args = decode(message.args) as unknown[];
          if (callback.key === 'onCheckpoint') state.checkpoint = structuredClone(args[0]) as ReportCheckpoint;
          const cancellation = new Promise<never>((_, reject) => {
            cancel = () => { reject(state.cancellation.signal.reason); };
            state.cancellation.signal.addEventListener('abort', cancel, { once: true });
          });
          const value: unknown = await Promise.race([Promise.resolve().then(() => callback.fn(...args)), cancellation]);
          this.post({ type: 'callback-result', id: message.id, ok: true, value: encode(value) });
        } catch (cause) {
          // Never use the thrown value as a sentinel: `throw undefined` is a real failure.
          const active = !state.settled && !state.localFailure && !state.cancellation.signal.aborted;
          if (active) {
            const error = failure(state.method, cause);
            if (state.method === 'describe') { state.localFailure = error; state.cancellation.abort(error); }
            else this.settle(state, false, error);
          }
          this.post({ type: 'callback-result', id: message.id, ok: false, error: encodeFailure(cause) });
          if (active) this.post({ type: 'abort', id: message.requestId, reason: encodeFailure(cause) });
        } finally {
          if (cancel) state.cancellation.signal.removeEventListener('abort', cancel);
        }
      });
      return;
    }
    const state = this.pending.get(message.id);
    if (!state) return;
    let ok = message.ok;
    let value: unknown;
    try { value = decode(message.ok ? message.value : message.error); }
    catch (cause) { ok = false; value = failure(state.method, cause); }
    // A remote deadline can finish while user code is still pending on this thread.
    // Unblock the callback barrier without replacing the remote error with a local failure.
    if (!ok) state.cancellation.abort(value);
    void state.events.then(() => {
      if (!state.settled) {
        try {
          if (state.localFailure && value instanceof ReportError) state.localFailure = new ReportError(state.localFailure, value.checkpoint);
          else if (state.localFailure && value instanceof Error && 'checkpoint' in value && !Object.hasOwn(state.localFailure, 'checkpoint')) Object.defineProperty(state.localFailure, 'checkpoint', { value: value.checkpoint, enumerable: true });
          this.settle(state, ok && !state.localFailure, state.localFailure ?? value);
        }
        catch (cause) { this.settle(state, false, failure(state.method, cause)); }
      }
      this.pending.delete(message.id);
    }).catch((cause: unknown) => { this.crash(cause); });
  }
  private settle(state: PendingRequest, ok: boolean, value: unknown): void {
    if (state.settled) return;
    state.settled = true;
    state.cleanup();
    if (!ok && state.checkpoint && value instanceof Error && !Object.hasOwn(value, 'checkpoint')) Object.defineProperty(value, 'checkpoint', { value: state.checkpoint, enumerable: true });
    if (!ok) state.cancellation.abort(value);
    // Factory callbacks remain registered: model loading is lazy and policy belongs to the instance.
    if (state.method !== 'create' || !ok) for (const callback of state.callbacks) this.callbacks.delete(callback);
    if (ok) state.resolve(value); else state.reject(value);
  }
  private crash(cause: unknown): void {
    if (this.stopped) return;
    this.stopped = true; this.closed = true;
    for (const state of this.pending.values()) this.settle(state, false, state.localFailure ?? failure(state.method, cause));
    this.pending.clear(); this.callbacks.clear(); this.unlisten(); this.unfail();
    void this.worker.terminate().catch(() => undefined);
  }
  dispose(): Promise<void> {
    if (!this.disposePromise) {
      this.closed = true;
      for (const [id, state] of this.pending) {
        if (state.settled) continue;
        const reason = new Error('Neko disposed');
        this.settle(state, false, state.localFailure ?? aborted(state.method, reason));
        this.post({ type: 'abort', id, reason: encodeFailure(reason) });
      }
      this.disposePromise = (async () => {
        try { if (!this.stopped) await this.request<void>('dispose', []); }
        finally {
          this.stopped = true; this.unlisten(); this.unfail();
          this.pending.clear(); this.callbacks.clear();
          await this.worker.terminate();
        }
      })();
    }
    return this.disposePromise;
  }
}

/** The inline SDK is instantiated inside a native thread or a module Worker, never on this thread. */
export async function createWorkerClient(options: NekoOptions = {}): Promise<WorkerClient> {
  options.signal?.throwIfAborted();
  const url = workerBootstrapUrl();
  await authorizeNetwork(options.policy, url, 'worker', options.localFilesOnly ?? false);
  options.signal?.throwIfAborted();
  const connection = new Connection(await spawnWorker(url, options.signal));
  const { signal, ...configuration } = options;
  try { await connection.request('create', [{ ...configuration, execution: 'inline' }], signal); }
  catch (cause) { await connection.dispose(); throw cause; }

  function describe(input: string | Page, options: DescribeOptions & { format: 'markdown' }): Promise<string>;
  function describe(input: string | Page, options?: DescribeOptions & { format?: 'json' }): Promise<StructuredReport>;
  function describe(input: string | Page, options: DescribeOptions): Promise<StructuredReport | string>;
  async function describe(input: string | Page, options: DescribeOptions = {}): Promise<StructuredReport | string> {
    connection.checkRequest('describe', options?.signal);
    validateDescribeOptions(options);
    const { signal, ...configuration } = options;
    return connection.request('describe', [input, configuration], signal);
  }
  return {
    infer: async (options) => { connection.checkRequest('infer', options?.signal); validateInferenceOptions(options); const { signal, ...configuration } = options; return connection.request('infer', [configuration], signal); },
    inferStructured: async (options) => { connection.checkRequest('inferStructured', options?.signal); validateInferenceOptions(options); compileStructuredSchema(options.schema); const { signal, ...configuration } = options; return connection.request('inferStructured', [configuration], signal); },
    planInference: async (options) => { connection.checkRequest('planInference', options?.signal); validateInferenceOptions(options, true); if (options.schema !== undefined) compileStructuredSchema(options.schema); const { signal, ...configuration } = options; return connection.request('planInference', [configuration], signal); },
    describe,
    load: (signal) => connection.request('load', [], signal),
    warmup: (signal) => connection.request('warmup', [], signal),
    runtimeStatus: () => connection.request('runtimeStatus', []),
    queueStatus: () => connection.request('queueStatus', []),
    cache: {
      model: {
        prefetch: (signal) => connection.request('cache.model.prefetch', [], signal),
        status: (signal) => connection.request('cache.model.status', [], signal),
        clear: (signal) => connection.request('cache.model.clear', [], signal),
      },
      engine: { status: () => connection.request('cache.engine.status', []), release: () => connection.request('cache.engine.release', []) },
    },
    backend: { current: () => connection.request('backend.current', []), detect: (device, signal) => connection.request('backend.detect', [device], signal) },
    dispose: () => connection.dispose(),
    [Symbol.asyncDispose]: () => connection.dispose(),
  };
}
