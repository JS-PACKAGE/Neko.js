import assert from 'node:assert/strict';
import test, { type TestContext } from 'node:test';
import { mkdtemp, realpath, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNeko, NekoError, type NekoEvent, type NekoOptions } from '../src/index.js';
import { EngineCache } from '../src/cache/engine.js';
import { getRegisteredModelProfile } from '../src/cache/registry.js';
import type { InferOptions, InferenceResult, RuntimeReadiness, VisionEngine } from '../src/core/engine.js';
import { encode } from '../src/runtime/protocol.js';

const profile = getRegisteredModelProfile();
const result: InferenceResult = {
  text: 'private model output', finishReason: 'stop', usage: { inputTokens: 4, outputTokens: 2, totalTokens: 6 },
  model: { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype },
  backend: { runtime: 'node', device: 'cpu', executionProviders: ['cpu'], gpuMemoryBytes: null, adapterEvidence: null, supported: true, capabilityEvidence: 'model-runtime-compatibility', sessions: [], providerEvidence: 'loaded-session-configuration' },
  timings: { loadMs: 0, preprocessMs: 0, firstTokenMs: null, generationMs: 0, totalMs: 0 }, memory: { jsHeapBytes: null, gpuBytes: null },
};
const readiness: RuntimeReadiness = { loaded: true, textReady: true, visionReady: false, model: result.model, backend: result.backend, memory: { jsHeapBytes: null, gpuBytes: null } };

// Exercise LocalNeko's actual scheduler with an inert engine, never model IO.
async function fixture(context: TestContext, onEvent: NekoOptions['onEvent'], infer: (options: InferOptions) => Promise<InferenceResult> = async () => result) {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-events-'));
  const engine = {
    infer, readiness: () => readiness, warmup: async () => readiness,
    inferStructured: async () => ({ ...result, value: { answer: 'private structured output' }, structured: { mode: 'tokenizer-constrained-runtime-validation', dialect: 'draft-07' } }),
    planInference: async () => ({ inputTokens: 4, maxNewTokens: 128, contextLimit: 4096, availableOutputTokens: 4092, fits: true, model: result.model }),
  } as unknown as VisionEngine;
  context.mock.method(EngineCache.prototype, 'use', async (operation: (engine: VisionEngine) => Promise<unknown>) => operation(engine));
  const neko = await createNeko({ device: 'cpu', localFilesOnly: true, cacheDir: directory, ...(onEvent ? { onEvent } : {}) });
  context.after(async () => { await neko.dispose(); await rm(directory, { recursive: true, force: true }); });
  return neko;
}

function requests(events: NekoEvent[]) { return events.filter((event) => event.type === 'request'); }

test('successful inference emits ordered content-free lifecycle metadata and opaque distinct ids', async (context) => {
  const events: NekoEvent[] = [];
  const neko = await fixture(context, (event) => { events.push(event); });
  const actual = await neko.infer({ prompt: 'private prompt text' });
  assert.equal(actual.text, result.text);
  const lifecycle = requests(events);
  assert.deepEqual(lifecycle.map((event) => [event.operation, event.phase, event.outcome]), [['infer', 'start', undefined], ['infer', 'end', 'ok']]);
  assert.equal(lifecycle[0]!.requestId, lifecycle[1]!.requestId);
  assert.deepEqual(lifecycle[1]!.usage, result.usage);
  assert.deepEqual(lifecycle[1]!.model, result.model);
  assert.ok(lifecycle[1]!.queueWaitMs! >= 0); assert.ok(lifecycle[1]!.durationMs! >= 0);
  const serialized = JSON.stringify(events);
  for (const secret of ['private prompt text', 'private model output', directorySentinel]) assert.equal(serialized.includes(secret), false);
  await neko.warmup(); await neko.load();
  assert.deepEqual(requests(events).slice(2).map((event) => [event.operation, event.phase]), [['warmup', 'start'], ['warmup', 'end'], ['load', 'start'], ['load', 'end']]);
  assert.equal(new Set(requests(events).filter((event) => event.phase === 'start').map((event) => event.requestId)).size, 3);
  assert.ok(events.some((event) => event.type === 'engine' && event.loadMs >= 0));
});
const directorySentinel = 'neko-events-';

test('failure end reports only the error code and preserves the original SDK error', async (context) => {
  const events: NekoEvent[] = []; const failure = new NekoError('private failure details', 'generate', 'MODEL_OUTPUT');
  const neko = await fixture(context, (event) => { events.push(event); }, async () => { throw failure; });
  await assert.rejects(neko.infer({ prompt: 'private prompt text' }), (error) => error === failure);
  const end = requests(events).at(-1)!;
  assert.equal(end.outcome, 'error'); assert.equal(end.errorCode, 'MODEL_OUTPUT');
  assert.equal(JSON.stringify(events).includes(failure.message), false);
});

test('queued cancellation emits aborted end without entering the engine', async (context) => {
  const events: NekoEvent[] = []; const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>(); let calls = 0;
  const neko = await fixture(context, (event) => { events.push(event); }, async () => { calls++; entered.resolve(); await release.promise; return result; });
  const active = neko.infer({ prompt: 'active' }); await entered.promise;
  const controller = new AbortController(); const queued = neko.infer({ prompt: 'cancelled', signal: controller.signal });
  controller.abort('private abort reason');
  await assert.rejects(queued, (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
  const end = requests(events).find((event) => event.outcome === 'aborted')!;
  assert.equal(end.errorCode, 'ABORTED'); assert.equal(calls, 1);
  assert.equal(JSON.stringify(events).includes('private abort reason'), false);
  release.resolve(); await active;
});

test('throwing, rejecting and never-settling observers cannot affect successful results', async (context) => {
  let mode = 0;
  const neko = await fixture(context, () => {
    if (mode === 0) throw new Error('observer failed');
    if (mode === 1) return Promise.reject(new Error('observer rejected'));
    return new Promise<void>(() => {});
  });
  for (mode = 0; mode < 3; mode++) assert.equal((await neko.infer({ prompt: 'test' })).text, result.text);
});

test('structured inference, planning and document operations retain their public operation names', async (context) => {
  const events: NekoEvent[] = [];
  const neko = await fixture(context, (event) => { events.push(event); });
  const structured = await neko.inferStructured({ prompt: 'private schema prompt', schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'], additionalProperties: false } });
  assert.equal(structured.value.answer, 'private structured output');
  await neko.planInference({ prompt: 'private image prompt', image: { data: new Uint8Array([1, 2, 3]), width: 1, height: 1, channels: 3 } });
  await assert.rejects(neko.ask({} as never, 'private question'));
  await assert.rejects(neko.describe({} as never));
  assert.deepEqual(requests(events).filter((event) => event.phase === 'end').map((event) => [event.operation, event.outcome]), [
    ['inferStructured', 'ok'], ['planInference', 'ok'], ['ask', 'error'], ['describe', 'error'],
  ]);
  const serialized = JSON.stringify(events);
  for (const secret of ['private schema prompt', 'private structured output', 'private image prompt', 'private question', 'additionalProperties']) assert.equal(serialized.includes(secret), false);
});

test('createNeko rejects non-function observers before acquiring a runtime', async () => {
  for (const onEvent of [true, null, {}, 'callback']) await assert.rejects(createNeko({ onEvent: onEvent as never }), (error: unknown) => error instanceof NekoError && error.code === 'INVALID_INPUT' && error.cause instanceof TypeError);
});

test('worker configuration encodes onEvent as one-way notify', () => {
  const modes: string[] = [];
  encode([{ onEvent() {} }], (_callback, mode, key) => { modes.push(`${key}:${mode}`); return 1; });
  assert.deepEqual(modes, ['onEvent:notify']);
});

// No model load is required: this worker operation fails on inert page validation.
test('worker forwards lifecycle events and observer rejection does not replace the operation error', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-events-worker-'));
  const events: NekoEvent[] = [];
  const neko = await createNeko({ device: 'cpu', localFilesOnly: true, execution: 'worker', cacheDir: directory,
    onEvent: async (event) => { events.push(event); throw new Error('private observer error'); },
  });
  try {
    await assert.rejects(neko.ask({} as never, 'private question'), (error: unknown) => error instanceof NekoError && error.code === 'INVALID_INPUT');
    const lifecycle = requests(events);
    assert.deepEqual(lifecycle.map((event) => [event.operation, event.phase, event.outcome]), [['ask', 'start', undefined], ['ask', 'end', 'error']]);
    assert.equal(lifecycle[1]!.errorCode, 'INVALID_INPUT'); assert.equal(lifecycle[1]!.execution?.mode, 'worker');
    assert.equal(JSON.stringify(events).includes('private question'), false);
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});
