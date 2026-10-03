import type { ExecutionInfo } from '../types.js';

const activeRequests = new WeakMap<object, AbortSignal>();
const workerExecutions = new WeakMap<object, ExecutionInfo>();

/** Worker identity is owned by the transport, never taken from caller options. */
export function setWorkerExecution(instance: object, execution: ExecutionInfo): void {
  workerExecutions.set(instance, execution);
}
export function workerExecution(instance: object): ExecutionInfo | undefined {
  return workerExecutions.get(instance);
}

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
