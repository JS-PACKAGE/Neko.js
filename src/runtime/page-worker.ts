/**
 * Short-lived classic `blob:` workers that keep large, pure computations off a page's main thread during inline execution.
 * Their source is a constant from this bundle; they fetch nothing. Workers, SDK workers and Node run the work in place instead.
 */
export function onPageMainThread(): boolean {
  return typeof Worker === 'function' && typeof document !== 'undefined' && typeof URL.createObjectURL === 'function';
}

/** Thrown before any input is posted, so the caller still owns its transferables and can run the work itself. */
export class PageWorkerUnavailable extends Error {}

/**
 * Runs `source`, which must define `self.onmessage` for one request and answer with one message, after it signals that it loaded
 * (a CSP that forbids blob workers fails that handshake, raising PageWorkerUnavailable). Transferables move only after the handshake.
 */
export async function runPageWorker<T>(source: string, message: unknown, transfer: Transferable[]): Promise<T> {
  const url = URL.createObjectURL(new Blob([`'use strict';\n${source}\nself.postMessage('ready');`], { type: 'text/javascript' }));
  let worker: Worker | undefined;
  try {
    try { worker = new Worker(url); } catch (cause) { throw new PageWorkerUnavailable('Page worker could not be created', { cause }); }
    const ready = Promise.withResolvers<void>();
    worker.onmessage = () => { ready.resolve(); };
    worker.onerror = () => { ready.reject(new PageWorkerUnavailable('Page worker failed to start')); };
    await ready.promise;
    const reply = Promise.withResolvers<T>();
    worker.onmessage = (event: MessageEvent<T>) => { reply.resolve(event.data); };
    worker.onerror = () => { reply.reject(new Error('Page worker failed')); };
    worker.onmessageerror = () => { reply.reject(new Error('Page worker reply could not be deserialized')); };
    worker.postMessage(message, transfer);
    return await reply.promise;
  } finally { worker?.terminate(); URL.revokeObjectURL(url); }
}

/** Lets the host run other tasks and render. A message task, unlike nested timers, is never clamped to 4 ms. */
export function yieldToPage(): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  const channel = new MessageChannel();
  channel.port1.onmessage = () => { channel.port1.close(); resolve(); };
  channel.port2.postMessage(undefined);
  return promise;
}
