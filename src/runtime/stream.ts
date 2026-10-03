import type { InferOptions, InferenceResult } from '../core/engine.js';
import { NekoError } from '../errors.js';
import { createBoundedStream, streamCharacters, type StreamBufferOptions } from './bounded-stream.js';

export interface InferStreamOptions extends InferOptions, StreamBufferOptions {}
export type InferenceStreamEvent = { type: 'token'; text: string } | { type: 'result'; result: InferenceResult; usage: InferenceResult['usage'] };

/** Synchronous model callbacks cannot await consumers; overflow cancels rather than buffering forever. */
export function createInferenceStream(options: InferStreamOptions, infer: (options: InferOptions) => Promise<InferenceResult>): AsyncIterableIterator<InferenceStreamEvent> {
  if (options.onToken !== undefined && typeof options.onToken !== 'function') throw new NekoError('onToken must be a function', 'preprocess', 'INVALID_INPUT');
  const { onToken, ...configuration } = options;
  delete configuration.maxBufferedEvents; delete configuration.maxBufferedCharacters;
  return createBoundedStream<InferenceStreamEvent>({
    ...options, characters: (event) => event.type === 'token' ? event.text.length : streamCharacters(event.result),
  }, async (sink) => {
    const result = await infer({ ...configuration, signal: sink.signal, onToken: (text) => {
      sink.emit({ type: 'token', text }); onToken?.(text);
    } });
    sink.signal.throwIfAborted();
    sink.finish({ type: 'result', result, usage: result.usage });
  });
}
