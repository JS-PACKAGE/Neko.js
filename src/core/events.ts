import type { ModelIdentity } from './engine.js';
import type { ErrorCode } from '../errors.js';
import type { ExecutionInfo } from '../types.js';

export type NekoOperation = 'infer' | 'inferStructured' | 'planInference' | 'ask' | 'describe' | 'load' | 'warmup';

/** Content-free, best-effort lifecycle notifications; duration includes admission wait. */
export type NekoEvent = {
  type: 'request';
  operation: NekoOperation;
  phase: 'start' | 'end';
  requestId: string;
  queueWaitMs?: number;
  durationMs?: number;
  outcome?: 'ok' | 'error' | 'aborted';
  errorCode?: ErrorCode;
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number };
  model?: ModelIdentity;
  execution?: ExecutionInfo;
} | { type: 'engine'; phase: 'loaded'; loadMs: number };

const prefix = globalThis.crypto.randomUUID();
let nextRequest = 0;
export function eventRequestId(): string { return `${prefix}:${++nextRequest}`; }

/** Do not await observers or expose their failures to request execution. */
export function notifyEvent(callback: ((event: NekoEvent) => void) | undefined, event: NekoEvent): void {
  try { if (callback) void Promise.resolve(callback(event)).catch(() => undefined); }
  catch { /* Observability must not affect SDK behavior. */ }
}
