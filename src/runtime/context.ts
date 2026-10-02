const activeRequests = new WeakMap<object, AbortSignal>();

/** Bound only while the inline scheduler actually executes an admitted request. */
export function setActiveRequest(instance: object, signal: AbortSignal | undefined): void {
  if (signal === undefined) activeRequests.delete(instance);
  else activeRequests.set(instance, signal);
}
export function activeRequestSignal(instance: object): AbortSignal | undefined {
  return activeRequests.get(instance);
}

/** Keep authorization and startup on the same worker identity in source and bundled layouts. */
export function workerBootstrapUrl(): URL {
  return new URL('./worker.js', import.meta.url);
}
