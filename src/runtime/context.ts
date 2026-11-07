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

let callerBase: string | undefined;
/** Set once in a browser SDK worker from its creating page, so caller-relative URLs resolve as they would inline. */
export function setCallerBaseUrl(href: string): void { callerBase = new URL(href).href; }
/** Base for caller-relative URLs: the creating page in a browser SDK worker (not the worker's module URL), else this realm's location. */
export function callerBaseUrl(): string | undefined { return callerBase ?? (typeof location === 'object' ? location.href : undefined); }
