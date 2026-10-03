import type { Neko, NekoOptions } from '../index.js';
import { NekoError } from '../errors.js';
import { activeRequestSignal } from './context.js';
import { aborted, decode, encode, encodeFailure, failure, type CallbackMode, type MainMessage, type MessagePort, type RequestMessage, type WorkerExecution, type WorkerMessage } from './protocol.js';

interface Operation {
  controller: AbortController;
  callbacks: Set<Promise<unknown>>;
  finished: boolean;
}
interface PendingCallback {
  requestId: number;
  resolve(value: unknown): void;
  reject(error: unknown): void;
}

/** All admission still goes through the inline SDK's scheduler; this transport has no second queue. */
export function installWorkerServer(port: MessagePort<MainMessage, WorkerMessage>, create: (options: NekoOptions) => Promise<Neko>, execution: WorkerExecution): void {
  const operations = new Map<number, Operation>();
  const operationSignals = new Map<AbortSignal, number>();
  const callbacks = new Map<number, PendingCallback>();
  let nextCallback = 1;
  let initialized: Promise<Neko> | undefined;
  let owner: Neko | undefined;
  let creatingSignal: AbortSignal | undefined;

  const callbackProxy = (origin: number, callbackId: number, mode: CallbackMode, persistent: boolean) => (...args: unknown[]): unknown => {
    // Instance callbacks belong to the request actually executing in the inline scheduler,
    // not an older request whose final text notification is still awaiting its parent ACK.
    let requestId = origin;
    if (persistent) {
      const signal = owner ? activeRequestSignal(owner) : creatingSignal;
      requestId = signal ? operationSignals.get(signal) ?? -1 : -1;
    }
    const operation = operations.get(requestId);
    if (!operation || operation.finished) throw new NekoError('Worker callback has no active request', 'generate', 'ABORTED');
    operation.controller.signal.throwIfAborted();
    const id = nextCallback++;
    const result = new Promise<unknown>((resolve, reject) => { callbacks.set(id, { requestId, resolve, reject }); });
    operation.callbacks.add(result);
    void result.then(() => { operation.callbacks.delete(result); }, (cause: unknown) => {
      operation.callbacks.delete(result);
      if (mode === 'notify') operation.controller.abort(cause);
    });
    try { port.post({ type: 'callback', id, requestId, callbackId, args: encode(args) }); }
    catch (cause) { callbacks.get(id)?.reject(cause); callbacks.delete(id); }
    // TextStreamer is synchronous; pending acknowledgements form the final result barrier.
    return mode === 'await' ? result : undefined;
  };

  const dispatch = async (message: RequestMessage): Promise<void> => {
    const operation: Operation = { controller: new AbortController(), callbacks: new Set(), finished: false };
    operations.set(message.id, operation);
    operationSignals.set(operation.controller.signal, message.id);
    const cancelCallbacks = () => {
      for (const [id, callback] of callbacks) if (callback.requestId === message.id) {
        callback.reject(aborted(message.method, operation.controller.signal.reason)); callbacks.delete(id);
      }
    };
    operation.controller.signal.addEventListener('abort', cancelCallbacks, { once: true });
    try {
      const args = decode(message.args, (id, mode) => callbackProxy(message.id, id, mode, message.method === 'create')) as unknown[];
      let value: unknown;
      if (message.method === 'create') {
        if (initialized) throw new NekoError('Worker is already initialized', 'create', 'RUNTIME_BUSY');
        const options = args[0] as NekoOptions;
        creatingSignal = operation.controller.signal;
        initialized = create({ ...options, execution: 'inline', signal: operation.controller.signal });
        owner = await initialized;
        value = execution;
      } else if (message.method === 'dispose') {
        for (const [id, pending] of operations) if (id !== message.id) pending.controller.abort(new Error('Neko disposed'));
        await initialized?.catch(() => undefined);
        await owner?.dispose();
      } else {
        if (!initialized) throw new NekoError('Worker is not initialized', 'create', 'DISPOSED');
        const instance = await initialized;
        const signal = operation.controller.signal;
        switch (message.method) {
          case 'infer': value = await instance.infer({ ...(args[0] as Parameters<Neko['infer']>[0]), signal }); break;
          case 'inferStructured': value = await instance.inferStructured({ ...(args[0] as Parameters<Neko['inferStructured']>[0]), signal }); break;
          case 'planInference': value = await instance.planInference({ ...(args[0] as Parameters<Neko['planInference']>[0]), signal }); break;
          case 'describe': value = await instance.describe(args[0] as Parameters<Neko['describe']>[0], { ...(args[1] as Parameters<Neko['describe']>[1]), signal }); break;
          case 'load': value = await instance.load(signal); break;
          case 'warmup': value = await instance.warmup(signal); break;
          case 'runtimeStatus': value = await instance.runtimeStatus(); break;
          case 'queueStatus': value = await instance.queueStatus(); break;
          case 'cache.model.prefetch': value = await instance.cache.model.prefetch(signal); break;
          case 'cache.model.status': value = await instance.cache.model.status(signal); break;
          case 'cache.model.clear': value = await instance.cache.model.clear(signal); break;
          case 'cache.engine.status': value = await instance.cache.engine.status(); break;
          case 'cache.engine.release': value = await instance.cache.engine.release(); break;
          case 'backend.current': value = await instance.backend.current(); break;
          case 'backend.detect': value = await instance.backend.detect(args[0] as Parameters<Neko['backend']['detect']>[0], signal); break;
        }
        if (value !== null && typeof value === 'object') {
          if (message.method === 'describe' && 'metadata' in value) value = { ...value, metadata: { ...(value.metadata as Record<string, unknown>), execution } };
          else if (message.method === 'infer' || message.method === 'inferStructured' || message.method === 'planInference' || message.method === 'load' || message.method === 'warmup' || message.method === 'runtimeStatus') value = { ...value, execution };
        }
      }
      operation.finished = true;
      // Notifications may be sent immediately before a synchronous final result. Do not let
      // worker success overtake a parent callback that has not acknowledged (or has thrown).
      await Promise.all(operation.callbacks);
      operation.controller.signal.throwIfAborted();
      port.post({ type: 'result', id: message.id, ok: true, value: encode(value) });
    } catch (cause) {
      operation.finished = true;
      port.post({ type: 'result', id: message.id, ok: false, error: encodeFailure(failure(message.method, cause)) });
    } finally {
      operation.controller.signal.removeEventListener('abort', cancelCallbacks);
      operations.delete(message.id);
      operationSignals.delete(operation.controller.signal);
      if (message.method === 'create') creatingSignal = undefined;
      for (const [id, callback] of callbacks) if (callback.requestId === message.id) { callback.reject(aborted(message.method, 'Request completed')); callbacks.delete(id); }
      // Catch discarded callbacks too: a runtime error may happen before notification ACKs arrive.
      for (const callback of operation.callbacks) void callback.catch(() => undefined);
    }
  };

  port.listen((message) => {
    if (message.type === 'request') { void dispatch(message); return; }
    if (message.type === 'abort') { operations.get(message.id)?.controller.abort(decode(message.reason)); return; }
    const callback = callbacks.get(message.id);
    if (!callback) return;
    callbacks.delete(message.id);
    if (message.ok) callback.resolve(decode(message.value)); else callback.reject(decode(message.error));
  });
}
