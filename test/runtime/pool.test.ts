import assert from 'node:assert/strict';
import test from 'node:test';
import { createNekoPool, type NekoPoolFactory, type NekoPoolOptions } from '../../src/runtime/pool.js';
import type { InferenceResult } from '../../src/core/engine.js';
import { getRegisteredModelProfile } from '../../src/cache/registry.js';
import { NekoError } from '../../src/errors.js';

const profile = getRegisteredModelProfile();
const result: InferenceResult = {
  text: 'fixture', finishReason: 'stop', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
  model: { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype },
  backend: { runtime: 'node', device: 'cpu', executionProviders: ['cpu'], gpuMemoryBytes: null, adapterEvidence: null, supported: true, capabilityEvidence: 'model-runtime-compatibility', sessions: [], providerEvidence: 'loaded-session-configuration' },
  timings: { loadMs: 0, preprocessMs: 0, firstTokenMs: null, generationMs: 0, totalMs: 0 }, memory: { jsHeapBytes: null, gpuBytes: null },
};
const options: NekoPoolOptions = { workers: [{ memoryBytes: 100 }, { memoryBytes: 100 }], budget: { memoryBytes: 200 }, maxPending: 8 };
const code = (expected: string) => (error: unknown): boolean => error instanceof NekoError && error.code === expected;

// Controlled owners exercise real scheduling and ownership, not tensor/model performance.
test('pool batch occupies independent workers concurrently and admits pending items in FIFO order', async () => {
  const starts: string[] = []; const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>();
  let active = 0; let peak = 0;
  const pool = await createNekoPool(options, async (workerOptions, index) => {
    assert.equal(workerOptions.execution, 'worker');
    return { async infer(input) {
      starts.push(`${index}:${input.prompt}`); peak = Math.max(peak, ++active);
      if (starts.length === 2) entered.resolve();
      if (starts.length <= 2) await release.promise;
      active--; return { ...result, text: input.prompt! };
    }, async dispose() {} };
  });
  const batch = pool.inferBatch(['a', 'b', 'c', 'd'].map((prompt) => ({ prompt })));
  await entered.promise;
  assert.equal(peak, 2); assert.deepEqual(starts, ['0:a', '1:b']); assert.equal(pool.status().pending, 2);
  release.resolve(); const output = await batch;
  assert.deepEqual(starts, ['0:a', '1:b', '0:c', '1:d']);
  assert.deepEqual(output.map((item) => item.status === 'fulfilled' ? item.value.text : 'failed'), ['a', 'b', 'c', 'd']);
  assert.deepEqual(pool.status().resources, { kind: 'application-declared-reservations', memoryBytes: 200, budgetMemoryBytes: 200 });
  await pool.dispose();
});

test('queued cancellation frees capped admission; active cancellation retains its owner until settlement', async () => {
  const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>(); const starts: string[] = [];
  const pool = await createNekoPool({ ...options, workers: [options.workers[0]!], maxPending: 1 }, async () => ({
    async infer(input) { starts.push(input.prompt!); if (starts.length === 1) { entered.resolve(); await release.promise; } return result; }, async dispose() { release.resolve(); },
  }));
  const first = pool.submit({ prompt: 'first' }); await entered.promise;
  const queued = pool.submit({ prompt: 'cancelled' });
  await assert.rejects(pool.infer({ prompt: 'overflow' }), code('QUEUE_FULL'));
  queued.dispose(); await assert.rejects(queued.result, code('ABORTED'));
  const next = pool.infer({ prompt: 'next' });
  first.cancel(); await assert.rejects(first.result, code('ABORTED'));
  assert.equal(pool.status().workers[0]!.state, 'running'); assert.deepEqual(starts, ['first']);
  release.resolve(); await next; assert.deepEqual(starts, ['first', 'next']);
  await pool.dispose();
});

test('zero pending cap permits sequential inference after result settlement', async () => {
  const pool = await createNekoPool({ ...options, workers: [options.workers[0]!], maxPending: 0 }, async () => ({ async infer() { return result; }, async dispose() {} }));
  await pool.infer({ prompt: 'one' }); await pool.infer({ prompt: 'two' });
  assert.equal(pool.status().admitted, 2); await pool.dispose();
});

test('ordinary item failures are isolated and worker termination never replays requests', async () => {
  let calls = 0;
  const pool = await createNekoPool({ ...options, workers: [options.workers[0]!] }, async () => ({
    async infer(input) { calls++; if (input.prompt === 'error') throw new Error('fixture failure'); if (input.prompt === 'crash') throw new NekoError('fixture crash', 'generate', 'WORKER_UNAVAILABLE'); return result; }, async dispose() {},
  }));
  const output = await pool.inferBatch(['error', 'ok', 'crash', 'not replayed'].map((prompt) => ({ prompt })));
  assert.deepEqual(output.map((item) => item.status), ['rejected', 'fulfilled', 'rejected', 'rejected']);
  assert.equal(calls, 3); assert.equal(pool.status().workers[0]!.state, 'unavailable');
  await assert.rejects(pool.infer({ prompt: 'later' }), code('WORKER_UNAVAILABLE')); await pool.dispose();
});

test('disposal cancels all items and requests worker disposal before waiting on active inference', async () => {
  const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>(); let disposals = 0;
  const pool = await createNekoPool({ ...options, workers: [options.workers[0]!] }, async () => ({
    async infer() { entered.resolve(); await release.promise; return result; }, async dispose() { disposals++; release.resolve(); },
  }));
  const first = pool.infer({ prompt: 'active' }); await entered.promise;
  const second = pool.infer({ prompt: 'queued' });
  const rejected = Promise.all([assert.rejects(first, code('DISPOSED')), assert.rejects(second, code('DISPOSED'))]);
  const disposal = pool.dispose(); assert.strictEqual(pool.dispose(), disposal); await disposal; await rejected;
  assert.equal(disposals, 1); await assert.rejects(pool.infer({ prompt: 'closed' }), code('DISPOSED'));
});

test('creation rejects over-budget reservations before invoking factory and cleans partially created owners', async () => {
  let calls = 0; let disposed = 0;
  const factory: NekoPoolFactory = async (_input, index) => { calls++; if (index === 1) throw new Error('creation failed'); return { async infer() { return result; }, async dispose() { disposed++; } }; };
  await assert.rejects(createNekoPool({ ...options, budget: { memoryBytes: 199 } }, factory), code('BUDGET_EXCEEDED')); assert.equal(calls, 0);
  await assert.rejects(createNekoPool(options, factory), /creation failed/); assert.equal(calls, 2); assert.equal(disposed, 1);
});

test('factory shared owners are rejected and disposed only once', async () => {
  let disposed = 0;
  const owner = { async infer() { return result; }, async dispose() { disposed++; } };
  await assert.rejects(createNekoPool(options, async () => owner), code('RUNTIME_BUSY')); assert.equal(disposed, 1);
});

test('batch and item abort signals produce separate cancellation results', async () => {
  const signal = new AbortController(); signal.abort();
  const pool = await createNekoPool(options, async () => ({ async infer() { return result; }, async dispose() {} }));
  const output = await pool.inferBatch([{ prompt: 'cancelled', signal: signal.signal }, { prompt: 'ok' }]);
  assert.equal(output[0]!.status, 'rejected'); assert.equal(output[1]!.status, 'fulfilled');
  const all = await pool.inferBatch([{ prompt: 'batch-cancelled' }], { signal: signal.signal }); assert.equal(all[0]!.status, 'rejected');
  await pool.dispose();
});
