import assert from 'node:assert/strict';
import test from 'node:test';
import { createAskDocumentsStream, createAskStream, createDescribeStream, createStructuredInferenceStream } from '../../src/runtime/operation-stream.js';
import { compileStructuredSchema, validateStructuredValue } from '../../src/core/structured.js';
import type { InferenceResult } from '../../src/core/engine.js';
import { NekoError } from '../../src/errors.js';
import { getRegisteredModelProfile } from '../../src/cache/registry.js';
import { generateExtractiveReport } from '../../src/report/extractive.js';
import { renderMarkdown } from '../../src/report/markdown.js';
import { askDocument } from '../../src/web/query.js';
import { DocumentIndex } from '../../src/documents/index.js';
import { askDocuments, type DocumentQueryHost } from '../../src/documents/query.js';
import type { Page, StructuredReport } from '../../src/types.js';

const profile = getRegisteredModelProfile();
const model = { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype };
const usage = { inputTokens: 17, outputTokens: 8, totalTokens: 25 };
const source: Page = { url: 'about:blank', paragraphs: [{ id: 'p1', text: 'The sky is blue.', source: { kind: 'html' } }], images: [] };
const schema = { type: 'object', additionalProperties: false, required: ['answer'], properties: { answer: { type: 'string' } } } as const;
const compiled = compileStructuredSchema(schema, 'validation-only');

// These deterministic hosts exercise callback contracts, not model output quality or performance.
function inference(text: string): InferenceResult {
  return { text, usage, model, finishReason: 'stop',
    backend: { runtime: 'node', device: 'cpu', executionProviders: ['cpu'], gpuMemoryBytes: null, adapterEvidence: null, supported: true, capabilityEvidence: 'model-runtime-compatibility', sessions: [], providerEvidence: 'loaded-session-configuration' },
    timings: { loadMs: 0, preprocessMs: 0, firstTokenMs: null, generationMs: 0, totalMs: 0 }, memory: { jsHeapBytes: null, gpuBytes: null } };
}

async function extractiveReport(): Promise<StructuredReport> {
  const result = await generateExtractiveReport(source, { mode: 'extractive', sourceLanguage: 'en' }, { model, execution: { mode: 'inline', runtime: 'node' } });
  assert.ok(typeof result !== 'string'); return result;
}

test('structured streams distinguish provisional deltas from one schema-validated result and actual usage', async () => {
  const text = '{"answer":"blue"}';
  let calls = 0; let callback = '';
  const stream = createStructuredInferenceStream({ prompt: 'color', schema, onToken: (delta) => { callback += delta; } }, async (options) => {
    calls++; options.onToken?.('{"answer":'); options.onToken?.('"blue"}');
    const value = validateStructuredValue(compiled, JSON.parse(text));
    return { ...inference(text), value, structured: { mode: 'json-boundary-runtime-validation', dialect: 'draft-07' } };
  });
  assert.equal(calls, 0);
  const events = [];
  for await (const event of stream) events.push(event);
  assert.deepEqual(events.map(({ type }) => type), ['provisional', 'provisional', 'result']);
  const final = events.at(-1)!; assert.equal(final.type, 'result');
  if (final.type !== 'result') throw new Error('Missing final result');
  assert.deepEqual(final.result.value, { answer: 'blue' }); assert.deepEqual(final.usage, usage);
  assert.equal(callback, text); assert.equal(calls, 1);
});

test('structured validation failures never turn provisional tokens into a result', async () => {
  const stream = createStructuredInferenceStream({ prompt: 'color', schema }, async (options) => {
    options.onToken?.('{"answer":42}');
    const value = validateStructuredValue(compiled, { answer: 42 });
    return { ...inference('{"answer":42}'), value, structured: { mode: 'json-boundary-runtime-validation', dialect: 'draft-07' } };
  });
  assert.equal((await stream.next()).value?.type, 'provisional');
  await assert.rejects(stream.next(), (error: unknown) => error instanceof NekoError && error.code === 'STRUCTURED_OUTPUT');
});

test('describe streams preserve real stages and render markdown from one accounted validated report', async () => {
  let calls = 0; let retained: StructuredReport | undefined;
  const stream = createDescribeStream(source, { mode: 'extractive', sourceLanguage: 'en', format: 'markdown' }, async (input, options) => {
    calls++; assert.equal(options.format, 'json'); assert.equal(input, source);
    const result = await generateExtractiveReport(source, options, { model, execution: { mode: 'inline', runtime: 'node' } });
    assert.ok(typeof result !== 'string'); retained = result; return result;
  });
  const events = [];
  for await (const event of stream) events.push(event);
  assert.equal(calls, 1);
  assert.deepEqual(events.filter(({ type }) => type === 'stage').map((event) => event.type === 'stage' ? event.event.id : ''), ['section:0', 'summary', 'conclusion']);
  const final = events.at(-1)!; assert.equal(final.type, 'result');
  if (final.type !== 'result' || !retained) throw new Error('Missing report');
  assert.equal(final.result, renderMarkdown(retained)); assert.deepEqual(final.usage, { inputTokens: 0, outputTokens: 0, totalTokens: 0 });
});

test('describe stage overflow cancels the host and callback failures remain observable', async () => {
  const report = await extractiveReport(); let signal: AbortSignal | undefined;
  const stream = createDescribeStream(source, { maxBufferedEvents: 1 }, async (_input, options) => {
    signal = options.signal;
    for (let index = 0; index < 3; index++) await options.onEvent?.({ type: 'stage-start', id: `section:${index}`, phase: 'section', usage, elapsedMs: index });
    return report;
  });
  assert.equal((await stream.next()).value?.type, 'stage');
  // Let the producer fill the single slot while the consumer deliberately waits.
  await new Promise<void>((resolve) => { setImmediate(resolve); });
  await assert.rejects(stream.next(), (error: unknown) => error instanceof NekoError && error.code === 'STREAM_OVERFLOW');
  assert.equal(signal?.aborted, true);
  const failed = createDescribeStream(source, { onToken: () => { throw undefined; } }, async (_input, options) => {
    options.onToken?.('unvalidated', 'summary'); return report;
  });
  assert.equal((await failed.next()).value?.type, 'provisional');
  await assert.rejects(failed.next(), (error: unknown) => error === undefined);
});

test('ask streams expose provisional JSON but only commit exact source-backed answers and actual usage', async () => {
  const stream = createAskStream(source, 'What color is the sky?', {}, (input, question, options) => askDocument(input, question, async (request) => {
    request.onToken?.('{"status":"answered",'); request.onToken?.('"claims":[]}');
    return { value: { status: 'answered', claims: [{ text: 'The sky is blue.', paragraphIds: ['p1'] }] }, usage, model };
  }, options));
  const events = [];
  for await (const event of stream) events.push(event);
  assert.deepEqual(events.map(({ type }) => type), ['provisional', 'provisional', 'result']);
  const final = events.at(-1)!;
  if (final.type !== 'result') throw new Error('Missing answer');
  assert.equal(final.result.answer, source.paragraphs[0]!.text); assert.deepEqual(final.usage, usage);
  assert.equal(final.result.claims[0]!.citations[0]!.quote, source.paragraphs[0]!.text);
});

test('ask cancellation after a buffered final answer still fails instead of emitting a stale result', async () => {
  const abort = new AbortController();
  const completed = Promise.withResolvers<void>(); let signal: AbortSignal | undefined;
  const stream = createAskStream(source, 'sky', { signal: abort.signal }, (input, question, options) => {
    signal = options.signal;
    return askDocument(input, question, async (request) => {
      request.onToken?.('provisional');
      return { value: { status: 'insufficient-evidence', claims: [] }, usage };
    }, options).then((result) => { completed.resolve(); return result; });
  });
  assert.equal((await stream.next()).value?.type, 'provisional');
  await completed.promise; abort.abort('stopped');
  await assert.rejects(stream.next(), (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
  assert.equal(signal?.aborted, true);
});

test('document retrieval streams preserve exact citations, coverage and actual inference usage', async () => {
  const index = await DocumentIndex.create([{ id: 'sky', text: source.paragraphs[0]!.text }]);
  const host: DocumentQueryHost = {
    async planInference(options) { return { inputTokens: 17, maxNewTokens: options.maxNewTokens!, contextLimit: 4096, availableOutputTokens: 4079, fits: true, model }; },
    async inferStructured(options) {
      options.onToken?.('{"status":"answered",');
      const chunkId = index.search('sky').hits[0]!.chunk.id;
      return { value: { status: 'answered', claims: [{ text: source.paragraphs[0]!.text, chunkIds: [chunkId] }] }, usage, model };
    },
  };
  const snapshot = index.exportSnapshot();
  const stream = createAskDocumentsStream(snapshot, 'sky', {}, (input, question, options) => askDocuments(host, input, question, options));
  const events = [];
  for await (const event of stream) events.push(event);
  assert.deepEqual(events.map(({ type }) => type), ['provisional', 'result']);
  const final = events.at(-1)!;
  if (final.type !== 'result') throw new Error('Missing document answer');
  assert.deepEqual(final.usage, usage); assert.equal(final.result.indexId, index.id);
  assert.equal(final.result.retrieval.selectedChunkIds.length, 1);
  assert.equal(index.validateCitation(final.result.claims[0]!.citations[0]!), true);
  assert.equal(index.exportSnapshot(), snapshot);
});

test('unread structured streams and early returns never start or retain active producers', async () => {
  let calls = 0;
  const release = Promise.withResolvers<void>(); let signal: AbortSignal | undefined;
  const stream = createStructuredInferenceStream({ prompt: 'color', schema }, async (options) => {
    calls++; signal = options.signal; options.onToken?.('partial'); await release.promise;
    return { ...inference('{"answer":"blue"}'), value: { answer: 'blue' }, structured: { mode: 'json-boundary-runtime-validation', dialect: 'draft-07' } };
  });
  const unread = createStructuredInferenceStream({ prompt: 'color', schema }, async () => {
    calls++; return { ...inference('{"answer":"blue"}'), value: { answer: 'blue' }, structured: { mode: 'json-boundary-runtime-validation', dialect: 'draft-07' } };
  });
  await unread.return!(); assert.equal(calls, 0);
  try {
    assert.equal((await stream.next()).value?.type, 'provisional');
    await stream.return!(); assert.equal(signal?.aborted, true);
    assert.deepEqual(await stream.next(), { done: true, value: undefined });
  } finally { release.resolve(); }
});
