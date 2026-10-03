import { NekoError } from '../errors.js';

/** Adapts a transferred stream promise without pre-reading or collecting its contents. */
export function deferredReadableStream(source: Promise<ReadableStream<Uint8Array>>, signal?: AbortSignal): ReadableStream<Uint8Array> {
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let ended = false;
  let cancelReason: unknown;
  let output: ReadableStreamDefaultController<Uint8Array>;
  const cancelReader = async (reason: unknown) => {
    const current = reader; reader = undefined;
    if (!current) return;
    try { await current.cancel(reason); } finally { current.releaseLock(); }
  };
  const abort = () => {
    if (ended) return;
    ended = true; cancelReason = signal?.reason;
    signal?.removeEventListener('abort', abort);
    output.error(signal?.reason instanceof NekoError ? signal.reason : new NekoError('Operation was cancelled', 'cache', 'ABORTED', { cause: signal?.reason }));
    void cancelReader(cancelReason).catch(() => undefined);
  };
  const ready = source.then(async (stream) => {
    reader = stream.getReader();
    if (ended) await cancelReader(cancelReason);
  });
  return new ReadableStream<Uint8Array>({
    start(controller) {
      output = controller;
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) abort();
      return ready.catch((cause: unknown) => { ended = true; signal?.removeEventListener('abort', abort); throw cause; });
    },
    async pull(controller) {
      await ready;
      if (ended) return;
      try {
        const current = reader!;
        const item = await current.read();
        if (ended) return;
        if (item.done) { ended = true; signal?.removeEventListener('abort', abort); current.releaseLock(); reader = undefined; controller.close(); }
        else controller.enqueue(item.value);
      } catch (cause) {
        ended = true; signal?.removeEventListener('abort', abort);
        reader?.releaseLock(); reader = undefined; controller.error(cause);
      }
    },
    async cancel(reason) {
      ended = true; cancelReason = reason;
      signal?.removeEventListener('abort', abort);
      await ready;
      await cancelReader(reason);
    },
  }, { highWaterMark: 0 });
}
