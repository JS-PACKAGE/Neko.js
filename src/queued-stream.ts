import { atStage, awaitUser, NekoError } from './errors.js';

/** Holds one admission slot until a lazily consumed resource stream closes or is cancelled. */
export function queuedReadable(
  run: (signal: AbortSignal, operation: (active: AbortSignal) => Promise<void>) => Promise<void>,
  open: (signal: AbortSignal) => ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): ReadableStream<Uint8Array> {
  const cancellation = new AbortController();
  const requestSignal = AbortSignal.any([cancellation.signal, ...(signal ? [signal] : [])]);
  let activeSignal = requestSignal;
  const ready = Promise.withResolvers<ReadableStreamDefaultReader<Uint8Array>>();
  const finished = Promise.withResolvers<void>();
  // Cancellation may arrive before the consumer asks for its first chunk.
  void ready.promise.catch(() => undefined);
  void finished.promise.catch(() => undefined);
  let operation: Promise<void> | undefined;
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let closed = false;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      operation ??= run(requestSignal, async (active) => {
        activeSignal = active;
        try {
          active.throwIfAborted();
          reader = open(active).getReader();
          ready.resolve(reader);
          await awaitUser(() => finished.promise, active, 'cache');
        } finally {
          if (reader) { try { await reader.cancel(); } finally { reader.releaseLock(); } }
        }
      }).catch((cause: unknown) => {
        ready.reject(cause); finished.reject(cause);
        if (!closed) { closed = true; controller.error(cause); }
      });
      try {
        const source = await ready.promise;
        const next = await atStage('cache', activeSignal, () => source.read());
        if (closed) return;
        if (next.done) { closed = true; finished.resolve(); controller.close(); }
        else controller.enqueue(next.value);
      } catch (cause) {
        finished.reject(cause);
        if (!closed) { closed = true; controller.error(cause); }
      }
    },
    async cancel(reason: unknown) {
      closed = true;
      cancellation.abort(new NekoError('Resource stream was cancelled', 'cache', 'ABORTED', { cause: reason }));
      finished.resolve();
      if (reader) await reader.cancel(reason);
      await operation;
    },
  }, { highWaterMark: 0 });
}
