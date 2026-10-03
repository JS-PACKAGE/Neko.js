import assert from 'node:assert/strict';
import test from 'node:test';
import { createInferenceStream } from '../../src/runtime/stream.js';
import { deferredReadableStream } from '../../src/runtime/readable.js';
import { NekoError } from '../../src/errors.js';
import type { InferenceResult } from '../../src/core/engine.js';

// Only the callback/iterator boundary is under test; these are not model-quality fixtures.
function result(text: string): InferenceResult {
  return { text, finishReason: 'stop', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 }, model: { id: 'test', revision: 'test', profile: 'default', dtype: { embed_tokens: 'q4', decoder_model_merged: 'q4', vision_encoder: 'fp16' } }, backend: { runtime: 'node', device: 'cpu', executionProviders: ['cpu'], gpuMemoryBytes: null, adapterEvidence: null, supported: true, capabilityEvidence: 'model-runtime-compatibility', sessions: [], providerEvidence: 'loaded-session-configuration' }, timings: { loadMs: 0, preprocessMs: 0, firstTokenMs: 0, generationMs: 0, totalMs: 0 }, memory: { jsHeapBytes: null, gpuBytes: null } };
}

test('inference stream starts lazily, preserves token order, and emits the final result once', async () => {
  let calls = 0;
  const stream = createInferenceStream({ prompt: 'exact' }, async (options) => {
    calls++; assert.equal(options.prompt, 'exact');
    options.onToken?.('A'); options.onToken?.('B');
    return result('AB');
  });
  assert.equal(calls, 0);
  const events = [];
  for await (const event of stream) events.push(event);
  assert.deepEqual(events.map((event) => event.type === 'token' ? event.text : event.result.text), ['A', 'B', 'AB']);
  assert.equal(calls, 1);
  assert.deepEqual(await stream.next(), { done: true, value: undefined });
});

test('breaking iteration aborts active inference without waiting for native completion', async () => {
  const entered = Promise.withResolvers<AbortSignal>();
  const release = Promise.withResolvers<void>();
  const stream = createInferenceStream({ prompt: 'exact' }, async (options) => {
    entered.resolve(options.signal!); options.onToken?.('A'); await release.promise; return result('A');
  });
  try {
    for await (const event of stream) { assert.equal(event.type, 'token'); break; }
    const signal = await entered.promise;
    assert.equal(signal.aborted, true);
    assert.ok(signal.reason instanceof NekoError && signal.reason.code === 'ABORTED');
  } finally { release.resolve(); }
});

test('synchronous producer overflow explicitly fails and aborts instead of pretending to await consumers', async () => {
  let signal: AbortSignal | undefined;
  const stream = createInferenceStream({ prompt: 'exact', maxBufferedEvents: 1 }, async (options) => {
    signal = options.signal;
    options.onToken?.('A'); options.onToken?.('B'); options.onToken?.('C');
    return result('ABC');
  });
  assert.deepEqual(await stream.next(), { done: false, value: { type: 'token', text: 'A' } });
  await assert.rejects(stream.next(), (error: unknown) => error instanceof NekoError && error.code === 'STREAM_OVERFLOW');
  assert.equal(signal?.aborted, true);
});

test('character bound and caller cancellation cannot leave a pending consumer hanging', async () => {
  const stream = createInferenceStream({ prompt: 'exact', maxBufferedCharacters: 2 }, async (options) => {
    options.onToken?.('A'); options.onToken?.('too large'); return result('Atoo large');
  });
  await stream.next();
  await assert.rejects(stream.next(), (error: unknown) => error instanceof NekoError && error.code === 'STREAM_OVERFLOW');
  const abort = new AbortController();
  const canceled = createInferenceStream({ prompt: 'exact', signal: abort.signal }, () => new Promise<InferenceResult>(() => undefined));
  const next = canceled.next(); abort.abort('caller stopped');
  await assert.rejects(next, (error: unknown) => error instanceof NekoError && error.code === 'ABORTED' && error.cause === 'caller stopped');
});

test('deferred transferred stream cancellation before arrival cancels its source once', async () => {
  const source = Promise.withResolvers<ReadableStream<Uint8Array>>();
  let canceled = 0;
  const stream = deferredReadableStream(source.promise);
  const cancellation = stream.cancel('consumer stopped');
  source.resolve(new ReadableStream<Uint8Array>({ cancel(reason) { canceled++; assert.equal(reason, 'consumer stopped'); } }, { highWaterMark: 0 }));
  await cancellation;
  assert.equal(canceled, 1);
});
