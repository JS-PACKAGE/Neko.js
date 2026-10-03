import type { InferenceResult, StructuredInferOptions, StructuredInferenceResult } from '../core/engine.js';
import type { SchemaValue } from '../core/structured.js';
import type { DescribeOptions, ReportEvent, ReportPhase } from '../report/generate.js';
import { validateDescribeOptions } from '../report/generate.js';
import { renderMarkdown } from '../report/markdown.js';
import type { AskOptions } from '../web/query.js';
import type { DocumentAnswer, Page, StructuredReport } from '../types.js';
import type { DocumentIndex, DocumentIndexSnapshot } from '../documents/index.js';
import type { AskDocumentsOptions, DocumentsAnswer } from '../documents/query.js';
import { NekoError } from '../errors.js';
import { createBoundedStream, streamCharacters, type StreamBufferOptions } from './bounded-stream.js';

export type { StreamBufferOptions } from './bounded-stream.js';
export type StructuredInferStreamOptions<S = unknown> = StructuredInferOptions<S> & StreamBufferOptions;
/** Provisional JSON deltas are unvalidated; only the terminal result is safe to use as a schema value. */
export type StructuredInferenceStreamEvent<T = unknown> =
  | { type: 'provisional'; text: string }
  | { type: 'result'; result: StructuredInferenceResult<T>; usage: InferenceResult['usage'] };

export function createStructuredInferenceStream<const S>(options: StructuredInferStreamOptions<S>, infer: (options: StructuredInferOptions<S>) => Promise<StructuredInferenceResult<SchemaValue<S>>>): AsyncIterableIterator<StructuredInferenceStreamEvent<SchemaValue<S>>> {
  if (options.onToken !== undefined && typeof options.onToken !== 'function') throw new NekoError('onToken must be a function', 'preprocess', 'INVALID_INPUT');
  const { onToken, ...configuration } = options;
  delete configuration.maxBufferedEvents; delete configuration.maxBufferedCharacters;
  return createBoundedStream<StructuredInferenceStreamEvent<SchemaValue<S>>>({
    ...options, characters: (event) => event.type === 'provisional' ? event.text.length : streamCharacters(event.result),
  }, async (sink) => {
    const result = await infer({ ...configuration, signal: sink.signal, onToken: (text) => {
      sink.emit({ type: 'provisional', text }); onToken?.(text);
    } });
    sink.signal.throwIfAborted();
    sink.finish({ type: 'result', result, usage: result.usage });
  });
}

export type DescribeStreamOptions = DescribeOptions & StreamBufferOptions;
export type DescribeStreamEvent<Result = StructuredReport | string> =
  | { type: 'provisional'; text: string; phase: ReportPhase }
  | { type: 'stage'; event: ReportEvent }
  | { type: 'result'; result: Result; usage: InferenceResult['usage'] };
export type DescribeStreamHost = (input: string | Page, options: DescribeOptions) => Promise<StructuredReport | string>;

export function createDescribeStream(input: string | Page, options: DescribeStreamOptions & { format: 'markdown' }, describe: DescribeStreamHost): AsyncIterableIterator<DescribeStreamEvent<string>>;
export function createDescribeStream(input: string | Page, options: DescribeStreamOptions & { format?: 'json' }, describe: DescribeStreamHost): AsyncIterableIterator<DescribeStreamEvent<StructuredReport>>;
export function createDescribeStream(input: string | Page, options: DescribeStreamOptions, describe: DescribeStreamHost): AsyncIterableIterator<DescribeStreamEvent>;
export function createDescribeStream(input: string | Page, options: DescribeStreamOptions, describe: DescribeStreamHost): AsyncIterableIterator<DescribeStreamEvent> {
  validateDescribeOptions(options);
  const { onToken, onEvent, format, ...configuration } = options;
  delete configuration.maxBufferedEvents; delete configuration.maxBufferedCharacters;
  return createBoundedStream<DescribeStreamEvent>({
    ...options, stage: 'report',
    characters: (event) => event.type === 'provisional' ? event.text.length : streamCharacters(event.type === 'stage' ? event.event : event.result),
  }, async (sink) => {
    // Markdown carries no accounting. Request the same validated report once and render it locally.
    const report = await describe(input, {
      ...configuration, signal: sink.signal, format: 'json',
      onToken: (text, phase) => { sink.emit({ type: 'provisional', text, phase }); onToken?.(text, phase); },
      onEvent: async (event) => { sink.emit({ type: 'stage', event: structuredClone(event) }); await onEvent?.(event); },
    });
    sink.signal.throwIfAborted();
    if (typeof report === 'string') throw new NekoError('Describe stream host must return a validated JSON report', 'report', 'MODEL_OUTPUT');
    sink.finish({ type: 'result', result: format === 'markdown' ? renderMarkdown(report) : report, usage: report.metadata.usage });
  });
}

export type AskStreamOptions = AskOptions & StreamBufferOptions;
export type AskStreamEvent =
  | { type: 'provisional'; text: string }
  | { type: 'result'; result: DocumentAnswer; usage: DocumentAnswer['usage'] };
export type AskStreamHost = (input: string | Page, question: string, options: AskOptions) => Promise<DocumentAnswer>;

export function createAskStream(input: string | Page, question: string, options: AskStreamOptions, ask: AskStreamHost): AsyncIterableIterator<AskStreamEvent> {
  if (options.onToken !== undefined && typeof options.onToken !== 'function') throw new NekoError('onToken must be a function', 'preprocess', 'INVALID_INPUT');
  const { onToken, ...configuration } = options;
  delete configuration.maxBufferedEvents; delete configuration.maxBufferedCharacters;
  return createBoundedStream<AskStreamEvent>({
    ...options, characters: (event) => event.type === 'provisional' ? event.text.length : streamCharacters(event.result),
  }, async (sink) => {
    const result = await ask(input, question, { ...configuration, signal: sink.signal, onToken: (text) => {
      sink.emit({ type: 'provisional', text }); onToken?.(text);
    } });
    sink.signal.throwIfAborted();
    sink.finish({ type: 'result', result, usage: result.usage });
  });
}

export type AskDocumentsStreamOptions = AskDocumentsOptions & StreamBufferOptions;
export type AskDocumentsStreamEvent =
  | { type: 'provisional'; text: string }
  | { type: 'result'; result: DocumentsAnswer; usage: DocumentsAnswer['usage'] };
export type AskDocumentsStreamHost = (index: DocumentIndex | DocumentIndexSnapshot, question: string, options: AskDocumentsOptions) => Promise<DocumentsAnswer>;

export function createAskDocumentsStream(index: DocumentIndex | DocumentIndexSnapshot, question: string, options: AskDocumentsStreamOptions, ask: AskDocumentsStreamHost): AsyncIterableIterator<AskDocumentsStreamEvent> {
  if (options.onToken !== undefined && typeof options.onToken !== 'function') throw new NekoError('onToken must be a function', 'preprocess', 'INVALID_INPUT');
  const { onToken, ...configuration } = options;
  delete configuration.maxBufferedEvents; delete configuration.maxBufferedCharacters;
  return createBoundedStream<AskDocumentsStreamEvent>({
    ...options, characters: (event) => event.type === 'provisional' ? event.text.length : streamCharacters(event.result),
  }, async (sink) => {
    const result = await ask(index, question, { ...configuration, signal: sink.signal, onToken: (text) => {
      sink.emit({ type: 'provisional', text }); onToken?.(text);
    } });
    sink.signal.throwIfAborted();
    sink.finish({ type: 'result', result, usage: result.usage });
  });
}
