import type { BackendDevice, BackendInfo } from '../backend/index.js';
import type { EngineCacheStatus } from '../cache/engine.js';
import type { RuntimeReadiness } from '../core/engine.js';
import type { QueueStatus } from '../core-queue.js';
import { NekoError, type ErrorStage } from '../errors.js';
import type { ExecutionInfo } from '../types.js';

export interface HealthOptions { signal?: AbortSignal; timeoutMs?: number; }
export interface RuntimeHealth {
  healthy: boolean;
  execution: ExecutionInfo;
  roundTripMs: number | null;
  checkedAt: number;
  reason?: 'timeout' | 'unavailable';
}
export interface RuntimeTransportDiagnostics {
  state: 'ready' | 'unavailable';
  pendingRequests: number;
  outstandingCallbacks: number;
  maxOutstandingNotifications: number;
}
export interface RuntimeDiagnostics {
  schemaVersion: 1;
  execution: ExecutionInfo;
  readiness: { loaded: boolean; textReady: boolean; visionReady: boolean };
  backend: { runtime: 'node' | 'browser'; device: BackendDevice; executionProviders: string[]; supported: boolean; providerEvidence: 'loaded-session-configuration' | 'configured' } | null;
  engine: EngineCacheStatus;
  queue: QueueStatus;
  transport?: RuntimeTransportDiagnostics;
}

/** Explicit allowlisting prevents future readiness/backend fields from disclosing inputs or paths. */
export function createRuntimeDiagnostics(input: {
  execution: ExecutionInfo;
  readiness: RuntimeReadiness | null;
  backend: BackendInfo | null;
  engine: EngineCacheStatus;
  queue: QueueStatus;
  transport?: RuntimeTransportDiagnostics;
}): RuntimeDiagnostics {
  const { execution, readiness, backend, engine, queue, transport } = input;
  return {
    schemaVersion: 1,
    execution: {
      mode: execution.mode, runtime: execution.runtime,
      ...(execution.workerId === undefined ? {} : { workerId: execution.workerId }),
      ...(execution.threadId === undefined ? {} : { threadId: execution.threadId }),
    },
    readiness: { loaded: engine.loaded, textReady: readiness?.textReady ?? false, visionReady: readiness?.visionReady ?? false },
    backend: backend ? { runtime: backend.runtime, device: backend.device, executionProviders: [...backend.executionProviders], supported: backend.supported, providerEvidence: readiness ? 'loaded-session-configuration' : 'configured' } : null,
    engine: { loaded: engine.loaded, sessions: engine.sessions, memory: engine.memory, hits: engine.hits, loads: engine.loads },
    queue: { running: queue.running, pending: queue.pending, maxPending: queue.maxPending, admitted: queue.admitted, rejected: queue.rejected },
    ...(transport ? { transport: { state: transport.state, pendingRequests: transport.pendingRequests, outstandingCallbacks: transport.outstandingCallbacks, maxOutstandingNotifications: transport.maxOutstandingNotifications } } : {}),
  };
}

export function validateHealthOptions(options: HealthOptions): void {
  if (typeof options !== 'object' || options === null || Array.isArray(options)) throw new NekoError('health options must be a record', 'create', 'INVALID_INPUT');
  if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) throw new NekoError('signal must be an AbortSignal', 'create', 'INVALID_INPUT');
  if (options.timeoutMs !== undefined && (!Number.isSafeInteger(options.timeoutMs) || options.timeoutMs < 1 || options.timeoutMs > 2_147_483_647)) throw new NekoError('timeoutMs must be between 1 and 2147483647', 'create', 'INVALID_INPUT');
}
export function validateHardDeadline(value: number | undefined): void {
  if (value !== undefined && (!Number.isSafeInteger(value) || value < 1 || value > 2_147_483_647)) throw new NekoError('hardDeadlineMs must be between 1 and 2147483647', 'preprocess', 'INVALID_INPUT');
}
export function assertHardDeadlineUnsupported(value: number | undefined, stage: ErrorStage = 'generate'): void {
  validateHardDeadline(value);
  if (value !== undefined) throw new NekoError('Hard deadlines require worker execution; inline native work cannot be forcibly interrupted', stage, 'UNSUPPORTED_BACKEND');
}
