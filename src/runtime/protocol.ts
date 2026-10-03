import { NekoError, type ErrorCode, type ErrorStage } from '../errors.js';
import { ReportError, type ReportCheckpoint } from '../report/generate.js';

export interface WorkerExecution {
  mode: 'worker';
  runtime: 'node' | 'browser';
  workerId: string;
  threadId?: number;
}
export type WorkerMethod = 'create' | 'infer' | 'inferStructured' | 'planInference' | 'describe' | 'load' | 'warmup' | 'runtimeStatus' | 'queueStatus' | 'cache.model.prefetch' | 'cache.model.status' | 'cache.model.clear' | 'cache.engine.status' | 'cache.engine.release' | 'backend.current' | 'backend.detect' | 'dispose';
export type CallbackMode = 'notify' | 'await';
export type Encoded = null | undefined | string | number | boolean | bigint
  | { kind: 'array'; items: Encoded[] }
  | { kind: 'record'; entries: [string, Encoded][] }
  | { kind: 'url'; href: string }
  | { kind: 'native'; value: unknown }
  | { kind: 'callback'; id: number; mode: CallbackMode }
  | { kind: 'error'; name: string; message: string; stack?: string; stage?: ErrorStage; code?: ErrorCode; cause?: Encoded; checkpoint?: Encoded; reportError?: true };
export type RequestMessage = { type: 'request'; id: number; method: WorkerMethod; args: Encoded };
export type MainMessage = RequestMessage | { type: 'abort'; id: number; reason: Encoded }
  | { type: 'callback-result'; id: number; ok: true; value: Encoded }
  | { type: 'callback-result'; id: number; ok: false; error: Encoded };
export type WorkerMessage = { type: 'result'; id: number; ok: true; value: Encoded }
  | { type: 'result'; id: number; ok: false; error: Encoded }
  | { type: 'callback'; id: number; requestId: number; callbackId: number; args: Encoded };
export interface MessagePort<Incoming, Outgoing> {
  post(message: Outgoing): void;
  listen(listener: (message: Incoming) => void): () => void;
}

/** Records are tagged as records too: caller JSON cannot impersonate an RPC callback. */
export function encode(value: unknown, callback?: (fn: (...args: unknown[]) => unknown, mode: CallbackMode, key: string) => number, key = '', ancestors = new Set<object>()): Encoded {
  if (value === null || value === undefined || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') return value;
  if (typeof value === 'function') {
    if (!callback) throw new TypeError('Functions are not supported in worker results');
    const mode = key === 'onToken' || key === 'progressCallback' ? 'notify' : 'await';
    return { kind: 'callback', id: callback(value as (...args: unknown[]) => unknown, mode, key), mode };
  }
  if (typeof value !== 'object') throw new TypeError('Worker values must be structured-cloneable');
  if (value instanceof URL) return { kind: 'url', href: value.href };
  if (value instanceof AbortSignal) throw new TypeError('AbortSignal must be supplied as the request signal');
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer || value instanceof Blob || value instanceof Date) return { kind: 'native', value };
  if (ancestors.has(value)) throw new TypeError('Cyclic worker values are not supported');
  ancestors.add(value);
  try {
    if (value instanceof Error) return {
      kind: 'error', name: value.name, message: value.message,
      ...(value.stack === undefined ? {} : { stack: value.stack }),
      ...(value instanceof NekoError ? { stage: value.stage, code: value.code } : {}),
      ...(value instanceof ReportError ? { reportError: true as const } : {}),
      ...(Object.hasOwn(value, 'cause') ? { cause: encode(value.cause, callback, 'cause', ancestors) } : {}),
      ...('checkpoint' in value ? { checkpoint: encode(value.checkpoint, callback, 'checkpoint', ancestors) } : {}),
    };
    if (Array.isArray(value)) return { kind: 'array', items: value.map((item) => encode(item, callback, key, ancestors)) };
    const prototype: unknown = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) throw new TypeError('Worker values must be plain records, URLs, blobs, or decoded image bytes');
    return { kind: 'record', entries: Object.entries(value).map(([name, item]) => [name, encode(item, callback, name, ancestors)]) };
  } finally { ancestors.delete(value); }
}

export function decode(value: Encoded, callback?: (id: number, mode: CallbackMode) => (...args: unknown[]) => unknown): unknown {
  if (value === null || typeof value !== 'object') return value;
  switch (value.kind) {
    case 'native': return value.value;
    case 'url': return new URL(value.href);
    case 'array': return value.items.map((item) => decode(item, callback));
    case 'record': return Object.fromEntries(value.entries.map(([key, item]) => [key, decode(item, callback)]));
    case 'callback': {
      if (!callback) throw new TypeError('Unexpected worker callback reference');
      return callback(value.id, value.mode);
    }
    case 'error': {
      const options = Object.hasOwn(value, 'cause') ? { cause: decode(value.cause, callback) } : undefined;
      let error = value.stage && value.code ? new NekoError(value.message, value.stage, value.code, options) : new Error(value.message, options);
      if (Object.hasOwn(value, 'checkpoint')) {
        const checkpoint = decode(value.checkpoint, callback);
        if (value.reportError && error instanceof NekoError) error = new ReportError(error, checkpoint as ReportCheckpoint);
        else Object.defineProperty(error, 'checkpoint', { value: checkpoint, enumerable: true });
      }
      error.name = value.name;
      if (value.stack !== undefined) error.stack = value.stack;
      return error;
    }
  }
}

export function methodStage(method: WorkerMethod): ErrorStage {
  if (method === 'create') return 'create';
  if (method === 'describe') return 'report';
  if (method === 'planInference') return 'preprocess';
  if (method === 'load' || method === 'warmup') return 'load';
  if (method.startsWith('cache.')) return 'cache';
  if (method.startsWith('backend.')) return 'backend';
  if (method === 'dispose') return 'dispose';
  return 'generate';
}
export function aborted(method: WorkerMethod, reason: unknown): NekoError {
  return new NekoError('Operation was cancelled', methodStage(method), 'ABORTED', { cause: reason });
}
export function failure(method: WorkerMethod, cause: unknown): NekoError {
  if (cause instanceof NekoError) return cause;
  return new NekoError(cause instanceof Error ? cause.message : String(cause), methodStage(method), cause instanceof TypeError || cause instanceof RangeError ? 'INVALID_INPUT' : 'OPERATION_FAILED', { cause });
}

export function encodeFailure(error: unknown): Encoded {
  try { return encode(error); }
  catch { return encode(new Error(error instanceof Error ? error.message : 'Worker operation threw a non-cloneable value')); }
}
