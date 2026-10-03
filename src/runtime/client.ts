import type { Neko, NekoOptions } from '../index.js';
import type { Page, StructuredReport } from '../types.js';
import { ReportError, validateDescribeOptions, type DescribeOptions, type ReportCheckpoint } from '../report/generate.js';
import { NekoError } from '../errors.js';
import { authorizeNetwork } from '../web/policy.js';
import { aborted, decode, encode, encodeFailure, failure, methodStage, streamTransfers, MAX_OUTSTANDING_NOTIFICATIONS, type MainMessage, type MessagePort, type WorkerMessage, type WorkerMethod, type WorkerExecution } from './protocol.js';
import type * as NodeWorkers from 'node:worker_threads';
import { workerBootstrapUrl } from './context.js';
import { validateInferenceOptions } from '../core/preflight.js';
import { compileStructuredSchema } from '../core/structured.js';
import { createInferenceStream } from './stream.js';
import { validateHardDeadline, validateHealthOptions, type HealthOptions, type RuntimeDiagnostics, type RuntimeHealth } from './diagnostics.js';
import { getRegisteredModelProfile } from '../cache/registry.js';
import { createConversationSession } from '../core/session.js';
import { deferredReadableStream } from './readable.js';
import type { StructuredInferOptions, StructuredInferenceResult } from '../core/engine.js';
import type { SchemaValue } from '../core/structured.js';

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
  queuedCallbacks: number;
  settled: boolean;
  streaming: boolean;
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
      post: (message, transfer) => worker.postMessage(message, transfer as NodeWorkers.TransferListItem[] | undefined),
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
    post: (message, transfer) => worker.postMessage(message, transfer ?? []),
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
  private termination?: Promise<void>;
  private readonly pings = new Map<number, { resolve(value: RuntimeHealth): void; reject(error: unknown): void; cleanup(): void }>();
  execution!: WorkerExecution;
  readonly lifetime = new AbortController();
  private readonly unlisten: () => void;
  private readonly unfail: () => void;
  constructor(private readonly worker: WorkerHandle) {
    this.unlisten = worker.listen((message) => { this.receive(message); });
    this.unfail = worker.onFailure((cause) => { this.crash(cause); });
  }

  checkRequest(method: WorkerMethod, signal?: AbortSignal): void {
    if (this.closed && method !== 'dispose') throw new NekoError('Neko is disposed', methodStage(method), 'DISPOSED');
    if (this.stopped) throw new NekoError('SDK worker is unavailable; call restart() to create a new worker', methodStage(method), 'WORKER_UNAVAILABLE');
    if (signal instanceof AbortSignal && signal.aborted) throw aborted(method, signal.reason);
  }
  request<T>(method: WorkerMethod, args: unknown[], signal?: AbortSignal, hardDeadlineMs?: number): Promise<T> {
    try { this.checkRequest(method, signal); validateHardDeadline(hardDeadlineMs); }
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
      let timer: Parameters<typeof globalThis.clearTimeout>[0] | undefined;
      const state: PendingRequest = { method, resolve: (value) => resolve(value as T), reject, callbacks: callbackIds, events: Promise.resolve(), queuedCallbacks: 0, settled: false, streaming: false, cancellation: new AbortController(), cleanup: () => { signal?.removeEventListener('abort', abort); clearTimeout(timer); } };
      this.pending.set(id, state);
      try {
        const value = encode(args, (callback, _mode, key) => {
          const callbackId = this.nextCallback++;
          this.callbacks.set(callbackId, { fn: callback, key }); callbackIds.push(callbackId);
          return callbackId;
        });
        signal?.addEventListener('abort', abort, { once: true });
        if (hardDeadlineMs !== undefined) timer = setTimeout(() => { this.crash(new NekoError('Worker hard deadline exceeded; outstanding work was terminated without replay', methodStage(method), 'DEADLINE_EXCEEDED')); }, hardDeadlineMs);
        this.post({ type: 'request', id, method, args: value });
        if (signal?.aborted) abort();
      } catch (cause) { this.settle(state, false, failure(method, cause)); this.pending.delete(id); }
    });
  }

  private post(message: MainMessage): void {
    try {
      const encoded = message.type === 'request' ? message.args : message.type === 'callback-result' && message.ok ? message.value : undefined;
      this.worker.post(message, streamTransfers(encoded));
    }
    catch (cause) { this.crash(cause); }
  }
  private receive(message: WorkerMessage): void {
    if (message.type === 'stream-end') {
      const state = this.pending.get(message.id);
      if (!state) return;
      state.streaming = false; state.cancellation.abort(aborted(state.method, 'Bundle stream completed'));
      void state.events.then(() => { this.pending.delete(message.id); }, (cause: unknown) => { this.pending.delete(message.id); this.crash(cause); });
      return;
    }
    if (message.type === 'pong') {
      const ping = this.pings.get(message.id);
      if (ping) { this.pings.delete(message.id); ping.cleanup(); ping.resolve({ healthy: true, execution: message.execution, roundTripMs: null, checkedAt: Date.now() }); }
      return;
    }
    if (message.type === 'callback') {
      const state = this.pending.get(message.requestId);
      const callback = this.callbacks.get(message.callbackId);
      if (!state || state.settled && !state.streaming || state.localFailure || state.cancellation.signal.aborted || !callback) {
        this.post({ type: 'callback-result', id: message.id, ok: false, error: encodeFailure(aborted(state?.method ?? 'create', 'Request is no longer active')) });
        return;
      }
      // Messages are FIFO, but async callbacks need an explicit chain before result settlement.
      state.queuedCallbacks++;
      state.events = state.events.then(async () => {
        if (state.settled && !state.streaming || state.localFailure || state.cancellation.signal.aborted) {
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
          const active = (!state.settled || state.streaming) && !state.localFailure && !state.cancellation.signal.aborted;
          if (active) {
            const error = failure(state.method, cause);
            if (state.method === 'describe' || state.streaming) { state.localFailure = error; state.cancellation.abort(error); }
            else this.settle(state, false, error);
          }
          this.post({ type: 'callback-result', id: message.id, ok: false, error: encodeFailure(cause) });
          if (active) this.post({ type: 'abort', id: message.requestId, reason: encodeFailure(state.localFailure ?? cause) });
        } finally {
          if (cancel) state.cancellation.signal.removeEventListener('abort', cancel);
        }
      }).finally(() => { state.queuedCallbacks--; });
      return;
    }
    const state = this.pending.get(message.id);
    if (!state) return;
    let ok = message.ok;
    let value: unknown;
    try { value = decode(message.ok ? message.value : message.error); }
    catch (cause) { ok = false; value = failure(state.method, cause); }
    if (ok && state.method === 'create') this.execution = value as WorkerExecution;
    if (ok && state.method === 'cache.model.exportBundle' && value instanceof ReadableStream) state.streaming = true;
    // A remote deadline can finish while user code is still pending on this thread.
    // Unblock the callback barrier without replacing the remote error with a local failure.
    if (!ok) state.cancellation.abort(value);
    void state.events.then(() => {
      if (!state.settled) {
        try {
          if (state.localFailure && value instanceof ReportError) state.localFailure = new ReportError(state.localFailure, value.checkpoint, value.partial);
          else if (state.localFailure && value instanceof Error) {
            for (const key of ['checkpoint', 'partial'] as const) if (Object.hasOwn(value, key) && !Object.hasOwn(state.localFailure, key)) Object.defineProperty(state.localFailure, key, { value: Object.getOwnPropertyDescriptor(value, key)?.value, enumerable: true });
          }
          this.settle(state, ok && !state.localFailure, state.localFailure ?? value);
        }
        catch (cause) { this.settle(state, false, failure(state.method, cause)); }
      }
      if (!state.streaming) this.pending.delete(message.id);
    }).catch((cause: unknown) => { this.crash(cause); });
  }
  private settle(state: PendingRequest, ok: boolean, value: unknown): void {
    if (state.settled) return;
    state.settled = true;
    state.cleanup();
    if (!ok && state.checkpoint && value instanceof Error && !Object.hasOwn(value, 'checkpoint')) {
      if (value instanceof NekoError) value = new ReportError(value, state.checkpoint);
      else Object.defineProperty(value, 'checkpoint', { value: state.checkpoint, enumerable: true });
    }
    if (!ok) state.cancellation.abort(value);
    // Factory callbacks remain registered: model loading is lazy and policy belongs to the instance.
    if (state.method !== 'create' || !ok) for (const callback of state.callbacks) this.callbacks.delete(callback);
    if (ok) state.resolve(value); else state.reject(value);
  }
  private crash(cause: unknown): void {
    if (this.stopped) return;
    this.stopped = true;
    const reason = cause instanceof NekoError ? cause : new NekoError('SDK worker failed; call restart() to recreate it without replay', 'generate', 'WORKER_UNAVAILABLE');
    this.lifetime.abort(reason);
    for (const state of this.pending.values()) { state.cancellation.abort(reason); this.settle(state, false, reason); }
    for (const ping of this.pings.values()) { ping.cleanup(); ping.resolve({ healthy: false, execution: this.execution, roundTripMs: null, checkedAt: Date.now(), reason: 'unavailable' }); }
    this.pending.clear(); this.callbacks.clear(); this.pings.clear(); this.unlisten(); this.unfail();
    this.termination = this.worker.terminate();
    void this.termination.catch(() => undefined);
  }
  async terminate(reason: NekoError): Promise<void> { this.crash(reason); await this.termination; }
  async health(options: HealthOptions = {}): Promise<RuntimeHealth> {
    validateHealthOptions(options);
    if (this.closed) return Promise.reject(new NekoError('Neko is disposed', 'create', 'DISPOSED'));
    if (options.signal?.aborted) return Promise.reject(aborted('create', options.signal.reason));
    if (this.stopped) return Promise.resolve({ healthy: false, execution: this.execution, roundTripMs: null, checkedAt: Date.now(), reason: 'unavailable' });
    const started = performance.now();
    const id = this.nextRequest++;
    return new Promise<RuntimeHealth>((resolve, reject) => {
      const abort = () => { const ping = this.pings.get(id); if (!ping) return; this.pings.delete(id); ping.cleanup(); reject(aborted('create', options.signal?.reason)); };
      const timer = setTimeout(() => {
        const ping = this.pings.get(id);
        if (!ping) return;
        this.pings.delete(id); ping.cleanup();
        resolve({ healthy: false, execution: this.execution, roundTripMs: null, checkedAt: Date.now(), reason: 'timeout' });
      }, options.timeoutMs ?? 5_000);
      this.pings.set(id, { resolve: (value) => resolve({ ...value, roundTripMs: value.healthy ? performance.now() - started : null }), reject, cleanup: () => { clearTimeout(timer); options.signal?.removeEventListener('abort', abort); } });
      options.signal?.addEventListener('abort', abort, { once: true });
      this.post({ type: 'ping', id });
      if (options.signal?.aborted) abort();
    });
  }
  async diagnostics(): Promise<RuntimeDiagnostics> {
    const diagnostic = await this.request<RuntimeDiagnostics>('diagnostics', []);
    let pendingRequests = 0; let outstandingCallbacks = 0;
    for (const state of this.pending.values()) { if (!state.settled || state.streaming) pendingRequests++; outstandingCallbacks += state.queuedCallbacks; }
    return { ...diagnostic, transport: { state: 'ready', pendingRequests, outstandingCallbacks: Math.max(outstandingCallbacks, diagnostic.transport?.outstandingCallbacks ?? 0), maxOutstandingNotifications: MAX_OUTSTANDING_NOTIFICATIONS } };
  }
  dispose(): Promise<void> {
    if (!this.disposePromise) {
      this.closed = true;
      this.lifetime.abort(new NekoError('Neko is disposed', 'dispose', 'DISPOSED'));
      for (const [id, state] of this.pending) {
        if (state.settled) continue;
        const reason = new NekoError('Neko disposed', methodStage(state.method), 'DISPOSED');
        this.settle(state, false, state.localFailure ?? aborted(state.method, reason));
        this.post({ type: 'abort', id, reason: encodeFailure(reason) });
      }
      for (const ping of this.pings.values()) { ping.cleanup(); ping.reject(new NekoError('Neko is disposed', 'create', 'DISPOSED')); }
      this.pings.clear();
      this.disposePromise = (async () => {
        try { if (!this.stopped) await this.request<void>('dispose', []); }
        finally {
          this.stopped = true; this.unlisten(); this.unfail();
          this.pending.clear(); this.callbacks.clear();
          await (this.termination ??= this.worker.terminate());
        }
      })();
    }
    return this.disposePromise;
  }
}

/** The inline SDK is instantiated inside a native thread or a module Worker, never on this thread. */
export async function createWorkerClient(options: NekoOptions = {}): Promise<WorkerClient> {
  const { signal: creationSignal, ...configuration } = options;
  const lifetime = new AbortController();
  const url = workerBootstrapUrl();
  const initialize = async (signal?: AbortSignal): Promise<Connection> => {
    const active = AbortSignal.any([lifetime.signal, ...(signal ? [signal] : [])]);
    active.throwIfAborted();
    await authorizeNetwork(configuration.policy, url, 'worker', configuration.localFilesOnly ?? false, undefined, active);
    active.throwIfAborted();
    const next = new Connection(await spawnWorker(url, active));
    try { await next.request('create', [{ ...configuration, execution: 'inline' }], active); return next; }
    catch (cause) { await next.dispose(); throw cause; }
  };
  let connection = await initialize(creationSignal);
  let disposed = false;
  let restartPromise: Promise<void> | undefined;
  let disposePromise: Promise<void> | undefined;

  function dispose(): Promise<void> {
    if (!disposePromise) {
      disposed = true; lifetime.abort(new NekoError('Neko is disposed', 'dispose', 'DISPOSED'));
      const current = connection.dispose();
      disposePromise = Promise.all([current, restartPromise?.catch(() => undefined)]).then(() => undefined);
    }
    return disposePromise;
  }

  function describe(input: string | Page, options: DescribeOptions & { format: 'markdown' }): Promise<string>;
  function describe(input: string | Page, options?: DescribeOptions & { format?: 'json' }): Promise<StructuredReport>;
  function describe(input: string | Page, options: DescribeOptions): Promise<StructuredReport | string>;
  async function describe(input: string | Page, options: DescribeOptions = {}): Promise<StructuredReport | string> {
    connection.checkRequest('describe', options?.signal);
    validateDescribeOptions(options);
    const { signal, hardDeadlineMs, ...configuration } = options;
    return connection.request('describe', [input, configuration], signal, hardDeadlineMs);
  }
  const client: WorkerClient = {
    infer: async (options) => {
      connection.checkRequest('infer', options?.signal); validateInferenceOptions(options);
      const { signal, hardDeadlineMs, ...configuration } = options;
      return connection.request('infer', [configuration], signal, hardDeadlineMs);
    },
    inferStream: (options) => {
      connection.checkRequest('infer', options?.signal); validateInferenceOptions(options); validateHardDeadline(options.hardDeadlineMs);
      return createInferenceStream(options, (request) => client.infer(request));
    },
    inferStructured: async <const S>(options: StructuredInferOptions<S>): Promise<StructuredInferenceResult<SchemaValue<S>>> => {
      connection.checkRequest('inferStructured', options?.signal); validateInferenceOptions(options); compileStructuredSchema(options.schema, options.structuredMode);
      const { signal, hardDeadlineMs, ...configuration } = options;
      return connection.request('inferStructured', [configuration], signal, hardDeadlineMs);
    },
    planInference: async (options) => {
      connection.checkRequest('planInference', options?.signal); validateInferenceOptions(options, true);
      if (options.schema !== undefined) compileStructuredSchema(options.schema, options.structuredMode);
      const { signal, hardDeadlineMs, ...configuration } = options;
      return connection.request('planInference', [configuration], signal, hardDeadlineMs);
    },
    describe,
    planReport: async (input, options = {}) => {
      connection.checkRequest('planReport', options?.signal); validateDescribeOptions(options);
      const { signal, hardDeadlineMs, ...configuration } = options;
      return connection.request('planReport', [input, configuration], signal, hardDeadlineMs);
    },
    ask: async (input, question, options = {}) => {
      connection.checkRequest('ask', options?.signal);
      if (typeof question !== 'string' || !question.trim()) throw new NekoError('question must be non-empty text', 'preprocess', 'INVALID_INPUT');
      if (typeof options !== 'object' || options === null || Array.isArray(options)) throw new NekoError('ask options must be a record', 'preprocess', 'INVALID_INPUT');
      const { signal, hardDeadlineMs, ...configuration } = options;
      return connection.request('ask', [input, question, configuration], signal, hardDeadlineMs);
    },
    session: (options) => {
      connection.checkRequest('infer');
      const profile = getRegisteredModelProfile(configuration.modelProfile, configuration.model);
      return createConversationSession(client, { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype }, options);
    },
    load: (signal) => connection.request('load', [], signal),
    warmup: (signal) => connection.request('warmup', [], signal),
    runtimeStatus: () => connection.request('runtimeStatus', []),
    queueStatus: () => connection.request('queueStatus', []),
    diagnostics: () => connection.diagnostics(),
    health: (options) => connection.health(options),
    restart: () => {
      if (disposed) return Promise.reject(new NekoError('Neko is disposed', 'create', 'DISPOSED'));
      if (!restartPromise) {
        restartPromise = (async () => {
          await connection.terminate(new NekoError('Worker was explicitly restarted; outstanding work was rejected without replay', 'generate', 'WORKER_UNAVAILABLE'));
          const next = await initialize();
          if (disposed) { await next.dispose(); throw new NekoError('Neko is disposed', 'create', 'DISPOSED'); }
          connection = next;
        })().finally(() => { restartPromise = undefined; });
      }
      return restartPromise;
    },
    cache: {
      model: {
        prefetch: (signal) => connection.request('cache.model.prefetch', [], signal),
        status: (signal) => connection.request('cache.model.status', [], signal),
        clear: (signal) => connection.request('cache.model.clear', [], signal),
        exportBundle: (signal) => {
          const active = connection;
          active.checkRequest('cache.model.exportBundle', signal);
          const cancellation = AbortSignal.any([active.lifetime.signal, ...(signal ? [signal] : [])]);
          return deferredReadableStream(active.request<ReadableStream<Uint8Array>>('cache.model.exportBundle', [], cancellation), cancellation);
        },
        importBundle: (source, signal) => connection.request('cache.model.importBundle', [source], signal),
        diagnostics: (signal) => connection.request('cache.model.diagnostics', [], signal),
      },
      engine: { status: () => connection.request('cache.engine.status', []), release: () => connection.request('cache.engine.release', []) },
    },
    backend: { current: () => connection.request('backend.current', []), detect: (device, signal) => connection.request('backend.detect', [device], signal) },
    dispose,
    [Symbol.asyncDispose]: dispose,
  };
  return client;
}
