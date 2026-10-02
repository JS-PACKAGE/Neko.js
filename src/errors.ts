export type ErrorStage = 'create' | 'backend' | 'cache' | 'load' | 'extract' | 'image' | 'preprocess' | 'generate' | 'report' | 'dispose';
export type ErrorCode = 'ABORTED' | 'INVALID_INPUT' | 'UNSUPPORTED_BACKEND' | 'RUNTIME_BUSY' | 'DISPOSED' | 'CONTEXT_LIMIT' | 'MODEL_OUTPUT' | 'INCOMPLETE_GENERATION' | 'LANGUAGE_MISMATCH' | 'OPERATION_FAILED';

export const ERROR_STAGES: Record<ErrorStage, true> = { create: true, backend: true, cache: true, load: true, extract: true, image: true, preprocess: true, generate: true, report: true, dispose: true };
export const ERROR_CODES: Record<ErrorCode, true> = { ABORTED: true, INVALID_INPUT: true, UNSUPPORTED_BACKEND: true, RUNTIME_BUSY: true, DISPOSED: true, CONTEXT_LIMIT: true, MODEL_OUTPUT: true, INCOMPLETE_GENERATION: true, LANGUAGE_MISMATCH: true, OPERATION_FAILED: true };

export class NekoError extends Error {
  override name = 'NekoError';
  constructor(message: string, readonly stage: ErrorStage, readonly code: ErrorCode, options?: ErrorOptions) { super(message, options); }
}

export async function atStage<T>(stage: ErrorStage, signal: AbortSignal | undefined, operation: () => Promise<T>, disposeOnAbort?: (value: T) => Promise<void>): Promise<T> {
  try {
    signal?.throwIfAborted();
    const value = await operation();
    if (signal?.aborted) { await disposeOnAbort?.(value); signal.throwIfAborted(); }
    return value;
  }
  catch (cause) {
    if (cause instanceof NekoError) throw cause;
    if (signal?.aborted) throw new NekoError('Operation was cancelled', stage, 'ABORTED', { cause: signal.reason });
    throw new NekoError(cause instanceof Error ? cause.message : String(cause), stage, cause instanceof TypeError || cause instanceof RangeError ? 'INVALID_INPUT' : 'OPERATION_FAILED', { cause });
  }
}
