import { NekoError, type ErrorStage } from '../errors.js';

export interface StreamBufferOptions {
  maxBufferedEvents?: number;
  maxBufferedCharacters?: number;
}
export interface StreamSink<Event> {
  readonly signal: AbortSignal;
  emit(event: Event): void;
  /** A terminal event may be delivered before its producer releases transactional ownership. */
  finish(event: Event): void;
}
interface BoundedStreamOptions<Event> extends StreamBufferOptions {
  signal?: AbortSignal;
  stage?: ErrorStage;
  characters(event: Event): number;
  onConsumed?: () => void;
  onCancel?: () => void;
}

/** Count retained string contents without allocating a serialized copy of a result or snapshot. */
export function streamCharacters(value: unknown): number {
  if (typeof value === 'string') return value.length;
  if (!value || typeof value !== 'object') return 0;
  let characters = 0;
  for (const key in value) {
    if (Object.hasOwn(value, key)) characters += streamCharacters((value as Record<string, unknown>)[key]);
  }
  return characters;
}

/** Callback producers cannot await consumers: overflow fails closed and cancels the real operation. */
export function createBoundedStream<Event>(options: BoundedStreamOptions<Event>, produce: (sink: StreamSink<Event>) => Promise<void>): AsyncIterableIterator<Event> {
  const maxEvents = options.maxBufferedEvents ?? 64;
  const maxCharacters = options.maxBufferedCharacters ?? 1_048_576;
  if (!Number.isSafeInteger(maxEvents) || maxEvents < 1 || maxEvents > 10_000) throw new NekoError('maxBufferedEvents must be between 1 and 10000', 'preprocess', 'INVALID_INPUT');
  if (!Number.isSafeInteger(maxCharacters) || maxCharacters < 1 || maxCharacters > 16_777_216) throw new NekoError('maxBufferedCharacters must be between 1 and 16777216', 'preprocess', 'INVALID_INPUT');
  if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) throw new NekoError('signal must be an AbortSignal', 'preprocess', 'INVALID_INPUT');
  const stage = options.stage ?? 'generate';
  const controller = new AbortController();
  const signal = options.signal ? AbortSignal.any([controller.signal, options.signal]) : controller.signal;
  const queue: ({ event: Event; characters: number } | undefined)[] = [];
  let head = 0;
  let buffered = 0;
  let characters = 0;
  let complete = false;
  let returned = false;
  let consumed = false;
  let finishing = false;
  let failed = false;
  let error: unknown;
  let producer: Promise<void> | undefined;
  let waiter: { resolve(value: IteratorResult<Event>): void; reject(error: unknown): void } | undefined;
  const done = (): IteratorResult<Event> => ({ done: true, value: undefined });
  const stopListening = () => { signal.removeEventListener('abort', aborted); };
  const clear = () => { queue.length = 0; head = 0; buffered = 0; characters = 0; };
  const fail = (cause: unknown) => {
    if (returned || consumed || failed) return;
    failed = true; error = cause; complete = true;
    clear(); stopListening(); options.onCancel?.();
    const pending = waiter; waiter = undefined; pending?.reject(cause);
  };
  const aborted = () => {
    fail(signal.reason instanceof NekoError ? signal.reason : new NekoError('Operation was cancelled', stage, 'ABORTED', { cause: signal.reason }));
  };
  signal.addEventListener('abort', aborted, { once: true });
  if (signal.aborted) aborted();
  const emit = (event: Event) => {
    if (signal.aborted) signal.throwIfAborted();
    if (returned || consumed || failed) throw error ?? new NekoError('Stream consumer stopped', stage, 'ABORTED');
    if (complete) throw new NekoError('Stream producer emitted after its terminal result', stage, 'INVALID_INPUT');
    const pending = waiter;
    if (pending) { waiter = undefined; pending.resolve({ done: false, value: event }); return; }
    const size = options.characters(event);
    if (buffered >= maxEvents || characters + size > maxCharacters) {
      const cause = new NekoError('Stream consumer exceeded its bounded buffer', stage, 'STREAM_OVERFLOW');
      fail(cause); controller.abort(cause); throw cause;
    }
    queue[(head + buffered) % maxEvents] = { event, characters: size }; buffered++; characters += size;
  };
  const start = () => {
    if (producer || returned || failed) return;
    producer = Promise.resolve().then(() => {
      signal.throwIfAborted();
      return produce({ signal, emit, finish: (event) => { emit(event); complete = true; } });
    }).then(() => {
      if (!complete && !returned && !failed) throw new NekoError('Stream producer completed without a terminal result', stage, 'MODEL_OUTPUT');
    }).catch((cause: unknown) => {
      fail(cause); if (!controller.signal.aborted) controller.abort(cause);
    });
  };
  const close = () => {
    if (returned || consumed) return;
    returned = true; complete = true; clear(); stopListening(); options.onCancel?.();
    const pending = waiter; waiter = undefined; pending?.resolve(done());
    controller.abort(new NekoError('Stream consumer stopped', stage, 'ABORTED'));
  };
  return {
    [Symbol.asyncIterator]() { return this; },
    next(): Promise<IteratorResult<Event>> {
      if (returned) return Promise.resolve(done());
      if (failed) return Promise.reject(error);
      if (waiter || finishing) return Promise.reject(new NekoError('Streams support one pending next() call', stage, 'INVALID_INPUT'));
      if (consumed) return Promise.resolve(done());
      if (buffered > 0) {
        const item = queue[head]!;
        queue[head] = undefined; head = (head + 1) % maxEvents; buffered--; characters -= item.characters;
        return Promise.resolve({ done: false, value: item.event });
      }
      if (complete) {
        finishing = true;
        try {
          options.onConsumed?.();
          // Exhaustion is one synchronous decision: later cancellation cannot reject an already committed turn.
          consumed = true; stopListening();
        } catch (cause) { fail(cause); controller.abort(cause); }
        return Promise.resolve(producer).then(() => {
          finishing = false;
          if (failed) throw error;
          if (returned) return done();
          return done();
        });
      }
      const result = new Promise<IteratorResult<Event>>((resolve, reject) => { waiter = { resolve, reject }; });
      start(); return result;
    },
    return(): Promise<IteratorResult<Event>> { close(); return Promise.resolve(done()); },
    throw(cause?: unknown): Promise<IteratorResult<Event>> { close(); return Promise.reject(cause); },
  };
}
