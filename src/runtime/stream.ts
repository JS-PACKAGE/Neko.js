import type { InferOptions, InferenceResult } from '../core/engine.js';
import { NekoError } from '../errors.js';

export interface InferStreamOptions extends InferOptions {
  maxBufferedEvents?: number;
  maxBufferedCharacters?: number;
}
export type InferenceStreamEvent = { type: 'token'; text: string } | { type: 'result'; result: InferenceResult };

/** Synchronous model callbacks cannot await consumers; overflow cancels rather than buffering forever. */
export function createInferenceStream(options: InferStreamOptions, infer: (options: InferOptions) => Promise<InferenceResult>): AsyncIterableIterator<InferenceStreamEvent> {
  const maxEvents = options.maxBufferedEvents ?? 64;
  const maxCharacters = options.maxBufferedCharacters ?? 1_048_576;
  if (!Number.isSafeInteger(maxEvents) || maxEvents < 1 || maxEvents > 10_000) throw new NekoError('maxBufferedEvents must be between 1 and 10000', 'preprocess', 'INVALID_INPUT');
  if (!Number.isSafeInteger(maxCharacters) || maxCharacters < 1 || maxCharacters > 16_777_216) throw new NekoError('maxBufferedCharacters must be between 1 and 16777216', 'preprocess', 'INVALID_INPUT');
  const controller = new AbortController();
  const { signal: callerSignal, onToken, ...configuration } = options;
  delete configuration.maxBufferedEvents; delete configuration.maxBufferedCharacters;
  const signal = AbortSignal.any([controller.signal, ...(callerSignal ? [callerSignal] : [])]);
  const queue: (InferenceStreamEvent | undefined)[] = [];
  let head = 0;
  let buffered = 0;
  let characters = 0;
  let started = false;
  let complete = false;
  let returned = false;
  let failed = false;
  let error: unknown;
  let waiter: { resolve(value: IteratorResult<InferenceStreamEvent>): void; reject(error: unknown): void } | undefined;

  const stopListening = () => { signal.removeEventListener('abort', aborted); };
  const fail = (cause: unknown) => {
    if (returned || failed) return;
    failed = true; error = cause; complete = true;
    queue.length = 0; buffered = 0; characters = 0;
    stopListening();
    const pending = waiter; waiter = undefined; pending?.reject(cause);
  };
  const aborted = () => {
    fail(signal.reason instanceof NekoError ? signal.reason : new NekoError('Operation was cancelled', 'generate', 'ABORTED', { cause: signal.reason }));
  };
  signal.addEventListener('abort', aborted, { once: true });
  if (signal.aborted) aborted();

  const push = (event: InferenceStreamEvent) => {
    if (returned || failed) return;
    const pending = waiter;
    if (pending) { waiter = undefined; pending.resolve({ done: false, value: event }); return; }
    const size = event.type === 'token' ? event.text.length : event.result.text.length;
    if (buffered >= maxEvents || characters + size > maxCharacters) {
      const cause = new NekoError('Inference stream consumer exceeded its bounded buffer', 'generate', 'STREAM_OVERFLOW');
      fail(cause); controller.abort(cause); throw cause;
    }
    queue[(head + buffered) % maxEvents] = event; buffered++; characters += size;
  };
  const start = () => {
    if (started || returned || failed) return;
    started = true;
    void Promise.resolve().then(() => infer({ ...configuration, signal, onToken: (text) => {
      if (returned || failed || signal.aborted) signal.throwIfAborted();
      push({ type: 'token', text });
      onToken?.(text);
    } })).then((result) => {
      if (returned || failed) return;
      push({ type: 'result', result });
      complete = true; stopListening();
    }).catch((cause: unknown) => { fail(cause); if (!controller.signal.aborted) controller.abort(cause); });
  };
  const close = () => {
    returned = true; complete = true; queue.length = 0; buffered = 0; characters = 0;
    stopListening();
    const pending = waiter; waiter = undefined; pending?.resolve({ done: true, value: undefined });
    controller.abort(new NekoError('Inference stream consumer stopped', 'generate', 'ABORTED'));
  };
  return {
    [Symbol.asyncIterator]() { return this; },
    next(): Promise<IteratorResult<InferenceStreamEvent>> {
      if (returned) return Promise.resolve({ done: true, value: undefined });
      if (failed) return Promise.reject(error);
      if (buffered > 0) {
        const event = queue[head]!;
        queue[head] = undefined; head = (head + 1) % maxEvents; buffered--;
        characters -= event.type === 'token' ? event.text.length : event.result.text.length;
        return Promise.resolve({ done: false, value: event });
      }
      if (complete) return Promise.resolve({ done: true, value: undefined });
      if (waiter) return Promise.reject(new NekoError('Inference streams support one pending next() call', 'generate', 'INVALID_INPUT'));
      const result = new Promise<IteratorResult<InferenceStreamEvent>>((resolve, reject) => { waiter = { resolve, reject }; });
      start();
      return result;
    },
    return(): Promise<IteratorResult<InferenceStreamEvent>> { close(); return Promise.resolve({ done: true, value: undefined }); },
    throw(cause?: unknown): Promise<IteratorResult<InferenceStreamEvent>> { close(); return Promise.reject(cause); },
  };
}
