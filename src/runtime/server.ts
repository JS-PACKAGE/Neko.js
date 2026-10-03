import type { Neko, NekoOptions } from '../index.js';
import { NekoError } from '../errors.js';
import { activeRequestSignal, setWorkerExecution } from './context.js';
import { aborted, decode, encode, encodeFailure, encodedCharacters, failure, MAX_NOTIFICATION_CHARACTERS, MAX_OUTSTANDING_NOTIFICATIONS, streamTransfers, type CallbackMode, type MainMessage, type MessagePort, type RequestMessage, type WorkerExecution, type WorkerMessage } from './protocol.js';

interface Operation {
  controller: AbortController;
  callbacks: Set<Promise<unknown>>;
  finished: boolean;
}
interface PendingCallback {
  requestId: number;
  mode: CallbackMode;
  characters: number;
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
  let eventCallbackId: number | undefined;
  let outstandingNotifications = 0;
  let notificationCharacters = 0;

  const removeCallback = (id: number): PendingCallback | undefined => {
    const callback = callbacks.get(id);
    if (!callback) return undefined;
    callbacks.delete(id);
    if (callback.mode === 'notify') { outstandingNotifications--; notificationCharacters -= callback.characters; }
    return callback;
  };

  const callbackProxy = (origin: number, callbackId: number, mode: CallbackMode, persistent: boolean) => (...args: unknown[]): unknown => {
    // Instance callbacks belong to the request actually executing in the inline scheduler,
    // not an older request whose final text notification is still awaiting its parent ACK.
    let requestId = origin;
    if (persistent) {
      const signal = owner ? activeRequestSignal(owner) : creatingSignal;
      requestId = signal ? operationSignals.get(signal) ?? -1 : -1;
    }
    if (persistent && callbackId === eventCallbackId) {
      // Lifecycle observers never hold an ACK barrier or abort an operation.
      try { port.post({ type: 'callback', id: nextCallback++, requestId, callbackId, args: encode(args) }); }
      catch { /* Observability is best-effort, including transport failures. */ }
      return undefined;
    }
    const operation = operations.get(requestId);
    if (!operation || operation.finished) throw new NekoError('Worker callback has no active request', 'generate', 'ABORTED');
    operation.controller.signal.throwIfAborted();
    const encoded = encode(args);
    const characters = mode === 'notify' ? encodedCharacters(encoded) : 0;
    if (mode === 'notify' && (outstandingNotifications >= MAX_OUTSTANDING_NOTIFICATIONS || notificationCharacters + characters > MAX_NOTIFICATION_CHARACTERS)) {
      const error = new NekoError('Worker notification consumer exceeded its bounded acknowledgement window', 'generate', 'STREAM_OVERFLOW');
      operation.controller.abort(error);
      throw error;
    }
    const id = nextCallback++;
    const result = new Promise<unknown>((resolve, reject) => { callbacks.set(id, { requestId, mode, characters, resolve, reject }); });
    if (mode === 'notify') { outstandingNotifications++; notificationCharacters += characters; }
    operation.callbacks.add(result);
    void result.then(() => { operation.callbacks.delete(result); }, (cause: unknown) => {
      operation.callbacks.delete(result);
      if (mode === 'notify') operation.controller.abort(cause);
    });
    try { port.post({ type: 'callback', id, requestId, callbackId, args: encoded }, streamTransfers(encoded)); }
    catch (cause) { removeCallback(id)?.reject(cause); }
    // TextStreamer is synchronous; pending acknowledgements form the final result barrier.
    return mode === 'await' ? result : undefined;
  };

  const dispatch = async (message: RequestMessage): Promise<void> => {
    const operation: Operation = { controller: new AbortController(), callbacks: new Set(), finished: false };
    operations.set(message.id, operation);
    operationSignals.set(operation.controller.signal, message.id);
    const cancelCallbacks = () => {
      for (const [id, callback] of callbacks) if (callback.requestId === message.id) {
        removeCallback(id)?.reject(aborted(message.method, operation.controller.signal.reason));
      }
    };
    operation.controller.signal.addEventListener('abort', cancelCallbacks, { once: true });
    let streaming = false;
    let cleaned = false;
    let importedStream: ReadableStream<Uint8Array> | undefined;
    const cleanup = () => {
      if (cleaned) return;
      cleaned = true;
      operation.controller.signal.removeEventListener('abort', cancelCallbacks);
      operations.delete(message.id);
      operationSignals.delete(operation.controller.signal);
      if (message.method === 'create') creatingSignal = undefined;
      for (const [id, callback] of callbacks) if (callback.requestId === message.id) removeCallback(id)?.reject(aborted(message.method, 'Request completed'));
      for (const callback of operation.callbacks) void callback.catch(() => undefined);
    };
    try {
      if (message.method === 'create' && message.args !== null && typeof message.args === 'object' && message.args.kind === 'array') {
        const configuration = message.args.items[0];
        if (configuration !== null && typeof configuration === 'object' && configuration.kind === 'record') {
          const event = configuration.entries.find(([key]) => key === 'onEvent')?.[1];
          if (event !== null && typeof event === 'object' && event.kind === 'callback') eventCallbackId = event.id;
        }
      }
      const args = decode(message.args, (id, mode) => callbackProxy(message.id, id, mode, message.method === 'create')) as unknown[];
      if (message.method === 'cache.model.importBundle' && args[0] instanceof ReadableStream) importedStream = args[0] as ReadableStream<Uint8Array>;
      let value: unknown;
      if (message.method === 'create') {
        if (initialized) throw new NekoError('Worker is already initialized', 'create', 'RUNTIME_BUSY');
        const options = args[0] as NekoOptions;
        creatingSignal = operation.controller.signal;
        initialized = create({ ...options, execution: 'inline', signal: operation.controller.signal });
        owner = await initialized;
        setWorkerExecution(owner, execution);
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
          case 'planReport': value = await instance.planReport(args[0] as Parameters<Neko['planReport']>[0], { ...(args[1] as Parameters<Neko['planReport']>[1]), signal }); break;
          case 'ask': value = await instance.ask(args[0] as Parameters<Neko['ask']>[0], args[1] as string, { ...(args[2] as Parameters<Neko['ask']>[2]), signal }); break;
          case 'releaseGenerationState': value = await instance.releaseGenerationState(args[0] as Parameters<Neko['releaseGenerationState']>[0]); break;
          case 'reuseCacheInfo': value = await instance.reuseCacheInfo(); break;
          case 'clearReuseCaches': value = await instance.clearReuseCaches(); break;
          case 'load': value = await instance.load(signal); break;
          case 'warmup': value = await instance.warmup(signal); break;
          case 'runtimeStatus': value = await instance.runtimeStatus(); break;
          case 'queueStatus': value = await instance.queueStatus(); break;
          case 'diagnostics': value = { ...await instance.diagnostics(), transport: { state: 'ready', pendingRequests: operations.size - 1, outstandingCallbacks: callbacks.size, maxOutstandingNotifications: MAX_OUTSTANDING_NOTIFICATIONS } }; break;
          case 'cache.model.prefetch': value = await instance.cache.model.prefetch(signal); break;
          case 'cache.model.status': value = await instance.cache.model.status(signal); break;
          case 'cache.model.clear': value = await instance.cache.model.clear(signal); break;
          case 'cache.model.exportBundle': value = instance.cache.model.exportBundle(signal); break;
          case 'cache.model.importBundle': value = await instance.cache.model.importBundle(args[0] as Parameters<Neko['cache']['model']['importBundle']>[0], signal); break;
          case 'cache.model.diagnostics': value = await instance.cache.model.diagnostics(signal); break;
          case 'cache.engine.status': value = await instance.cache.engine.status(); break;
          case 'cache.engine.release': value = await instance.cache.engine.release(); break;
          case 'backend.current': value = await instance.backend.current(); break;
          case 'backend.detect': value = await instance.backend.detect(args[0] as Parameters<Neko['backend']['detect']>[0], signal); break;
        }
        if (value !== null && typeof value === 'object') {
          if (message.method === 'infer' || message.method === 'inferStructured' || message.method === 'planInference' || message.method === 'load' || message.method === 'warmup' || message.method === 'runtimeStatus') value = { ...value, execution };
        }
      }
      if (message.method === 'cache.model.exportBundle' && value instanceof ReadableStream) {
        const reader = (value as ReadableStream<Uint8Array>).getReader();
        let ended = false;
        let controller: ReadableStreamDefaultController<Uint8Array>;
        const finish = () => {
          operation.finished = true;
          operation.controller.signal.removeEventListener('abort', abort);
          reader.releaseLock(); cleanup();
          port.post({ type: 'stream-end', id: message.id });
        };
        const abort = () => {
          if (ended) return;
          ended = true;
          const reason = failure(message.method, operation.controller.signal.reason);
          controller.error(reason);
          void reader.cancel(reason).catch(() => undefined).finally(finish);
        };
        value = new ReadableStream<Uint8Array>({
          start(output) { controller = output; operation.controller.signal.addEventListener('abort', abort, { once: true }); },
          async pull(output) {
            if (ended) return;
            try {
              const item = await reader.read();
              if (ended) return;
              await Promise.all(operation.callbacks);
              operation.controller.signal.throwIfAborted();
              if (item.done) { ended = true; finish(); output.close(); }
              else output.enqueue(item.value);
            } catch (cause) {
              if (ended) return;
              ended = true; output.error(failure(message.method, cause));
              await reader.cancel(cause).catch(() => undefined); finish();
            }
          },
          async cancel(reason) {
            if (ended) return;
            ended = true; operation.controller.abort(reason);
            await reader.cancel(reason).catch(() => undefined); finish();
          },
        }, { highWaterMark: 0 });
        streaming = true;
      }
      operation.finished = !streaming;
      // Notifications may be sent immediately before a synchronous final result. Do not let
      // worker success overtake a parent callback that has not acknowledged (or has thrown).
      await Promise.all(operation.callbacks);
      operation.controller.signal.throwIfAborted();
      const encoded = encode(value);
      port.post({ type: 'result', id: message.id, ok: true, value: encoded }, streamTransfers(encoded));
    } catch (cause) {
      if (streaming) { streaming = false; operation.controller.abort(cause); }
      operation.finished = true;
      if (importedStream && !importedStream.locked) void importedStream.cancel(cause).catch(() => undefined);
      port.post({ type: 'result', id: message.id, ok: false, error: encodeFailure(failure(message.method, cause)) });
    } finally {
      if (!streaming) cleanup();
    }
  };

  port.listen((message) => {
    if (message.type === 'ping') { port.post({ type: 'pong', id: message.id, execution }); return; }
    if (message.type === 'request') { void dispatch(message); return; }
    if (message.type === 'abort') { operations.get(message.id)?.controller.abort(decode(message.reason)); return; }
    const callback = callbacks.get(message.id);
    if (!callback) return;
    removeCallback(message.id);
    if (message.ok) callback.resolve(decode(message.value)); else callback.reject(decode(message.error));
  });
}
