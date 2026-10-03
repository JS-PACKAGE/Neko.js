import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, realpath, rm, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';
import type * as SDK from '../../src/index.js';

// The worker's package-local asset policy is exercised against the shipped bundle layout.
const { createNeko, NekoError, MODEL_ID, MODEL_REVISION }: typeof SDK = await import(new URL('../../node/index.js', import.meta.url).href);

async function cacheDirectory(): Promise<string> {
  return mkdtemp(join(await realpath(tmpdir()), 'neko-worker-test-'));
}

test('native workers have independent runtime ownership and disposal releases each owner', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const owners: SDK.Neko[] = [];
  try {
    const inline = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu' }); owners.push(inline);
    const first = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu', execution: 'worker' }); owners.push(first);
    const second = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu', execution: 'worker' }); owners.push(second);
    await assert.rejects(createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu' }), (error: unknown) => error instanceof NekoError && error.code === 'RUNTIME_BUSY');
    assert.equal((await first.backend.current())?.runtime, 'node');
    assert.equal((await second.backend.current())?.runtime, 'node');
    assert.equal(await first.runtimeStatus(), null);
    assert.equal((await first.cache.model.status()).downloaded, false);
    const disposal = first.dispose();
    assert.equal(first.dispose(), disposal);
    await disposal;
    await assert.rejects(first.infer({ prompt: 'disposed' }), (error: unknown) => error instanceof NekoError && error.code === 'DISPOSED');
    for (const diagnostic of [first.backend.current(), first.runtimeStatus(), first.queueStatus(), first.cache.engine.status()]) {
      await assert.rejects(diagnostic, (error: unknown) => error instanceof NekoError && error.code === 'DISPOSED');
    }
    for (const request of [
      () => first.infer({ prompt: '' }),
      () => first.inferStructured({ prompt: '', schema: { type: 'invalid' } }),
      () => first.planInference({ prompt: '', schema: { type: 'invalid' } }),
      () => first.describe('', { language: 'not_a_language' }),
    ]) await assert.rejects(request(), (error: unknown) => error instanceof NekoError && error.code === 'DISPOSED');
    assert.equal((await second.cache.engine.status()).loaded, false);
    assert.equal((await inline.backend.current())?.runtime, 'node');
  } finally {
    try { await Promise.all(owners.map((owner) => owner.dispose())); }
    finally { await rm(directory, { recursive: true, force: true }); }
  }
});

test('worker policy receives URLs on the parent and preserves a denial that throws undefined', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const decisions: string[] = [];
  const neko = await createNeko({ cacheDir: directory, device: 'cpu', execution: 'worker', policy: {
    network: (url, kind) => {
      assert.ok(url instanceof URL);
      decisions.push(kind);
      if (kind === 'model') throw undefined;
    },
  } });
  try {
    await assert.rejects(neko.cache.model.prefetch(), (error: unknown) => error instanceof NekoError && error.code === 'OPERATION_FAILED' && Object.hasOwn(error, 'cause') && error.cause === undefined);
    assert.deepEqual(decisions, ['worker', 'model']);
    assert.equal((await neko.cache.model.status()).bytes, 0);
    await assert.rejects(neko.planInference({ prompt: 'Count the actual chat tokens.' }), (error: unknown) => error instanceof NekoError && error.code === 'OPERATION_FAILED' && Object.hasOwn(error, 'cause') && error.cause === undefined);
    assert.deepEqual(decisions, ['worker', 'model', 'model']);
    assert.equal((await neko.cache.engine.status()).loaded, false);
    assert.equal((await neko.queueStatus()).running, null);
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('worker cancels queued and active requests, and disposes while a policy callback never resolves', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  let entered!: () => void;
  const policyEntered = new Promise<void>((resolve) => { entered = resolve; });
  const neko = await createNeko({ cacheDir: directory, device: 'cpu', execution: 'worker', queue: { maxPending: 1 }, policy: {
    network: (_url, kind) => {
      if (kind !== 'model') return;
      entered();
      return new Promise<void>(() => undefined);
    },
  } });
  try {
    const activeAbort = new AbortController();
    const activeReason = { cancelled: 'active' };
    const active = neko.cache.model.prefetch(activeAbort.signal);
    const activeRejection = assert.rejects(active, (error: unknown) => error instanceof NekoError && error.code === 'ABORTED' && error.cause === activeReason);
    await policyEntered;
    const queuedAbort = new AbortController();
    const queuedReason = { cancelled: 'queued' };
    const queued = neko.cache.model.status(queuedAbort.signal);
    const queuedRejection = assert.rejects(queued, (error: unknown) => error instanceof NekoError && error.code === 'ABORTED' && error.cause === queuedReason);
    const status = await neko.queueStatus();
    assert.equal(status.running, 'cache');
    assert.equal(status.pending, 1);
    await assert.rejects(neko.cache.model.status(), (error: unknown) => error instanceof NekoError && error.code === 'QUEUE_FULL');
    queuedAbort.abort(queuedReason);
    await queuedRejection;
    assert.equal((await neko.queueStatus()).pending, 0);
    activeAbort.abort(activeReason);
    await activeRejection;
    await neko.dispose();
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('a queued policy decision is not held behind an earlier request notification acknowledgement', { timeout: 30_000 }, async (context) => {
  const directory = await cacheDirectory();
  const root = join(directory, MODEL_ID, MODEL_REVISION);
  await mkdir(root, { recursive: true });
  await writeFile(join(root, 'generation_config.json'), JSON.stringify({
    bos_token_id: 248044, do_sample: true, eos_token_id: [248046, 248044], pad_token_id: 248044,
    temperature: 0.6, top_k: 20, top_p: 0.95, transformers_version: '5.3.0.dev0', trust_remote_code: false,
  }, null, 2) + '\n');
  let release!: () => void;
  const acknowledgement = new Promise<void>((resolve) => { release = resolve; });
  let reached!: () => void;
  const policyReached = new Promise<void>((resolve) => { reached = resolve; });
  const denial = new Error('second request denied');
  const neko = await createNeko({ cacheDir: directory, device: 'cpu', execution: 'worker',
    progressCallback: async () => { await acknowledgement; },
    policy: { network: (_url, kind) => { if (kind === 'model') { reached(); throw denial; } } },
  });
  let cancel!: () => void;
  const canceled = new Promise<never>((_, reject) => { cancel = () => { reject(context.signal.reason); }; });
  context.signal.addEventListener('abort', cancel, { once: true });
  const work: Promise<unknown>[] = [];
  try {
    let completed = false;
    const first = neko.cache.model.status().then((status) => { completed = true; return status; });
    const rejected = assert.rejects(neko.cache.model.prefetch(), (error: unknown) => error instanceof NekoError && error.cause === denial);
    work.push(first, rejected);
    await Promise.race([policyReached, canceled]);
    assert.equal(completed, false);
    release();
    const [status] = await Promise.all([first, rejected]);
    assert.equal(status.bytes, 248);
  } finally {
    context.signal.removeEventListener('abort', cancel); release();
    await neko.dispose();
    await Promise.all(work).catch(() => undefined);
    await rm(directory, { recursive: true, force: true });
  }
});

test('a report duration budget cancels a never-settling parent predicate and releases queued work without a caller signal', { timeout: 30_000 }, async (context) => {
  const directory = await cacheDirectory();
  const entered = Promise.withResolvers<void>();
  const canceled = Promise.withResolvers<never>();
  const cancel = () => { canceled.reject(context.signal.reason); };
  const neko = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu', execution: 'worker', queue: { maxPending: 1 } });
  context.signal.addEventListener('abort', cancel, { once: true });
  const page: SDK.Page = { url: 'https://example.test/', paragraphs: [{ id: 'p1', text: 'The exhibit opens Monday.', source: { kind: 'html' } }], images: [] };
  const work: Promise<unknown>[] = [];
  try {
    const report = neko.describe(page, {
      budget: { maxDurationMs: 1_000 },
      sources: { paragraph: () => { entered.resolve(); return new Promise<boolean>(() => undefined); } },
    });
    const rejected = assert.rejects(report, (error: unknown) => error instanceof NekoError && error.code === 'BUDGET_EXCEEDED' && error.stage === 'report');
    work.push(rejected);
    await Promise.race([entered.promise, canceled.promise]);
    const queued = neko.cache.model.status();
    work.push(queued);
    const queuedStatus = await neko.queueStatus();
    assert.equal(queuedStatus.running, 'report');
    assert.equal(queuedStatus.pending, 1);
    const [, status] = await Promise.race([Promise.all([rejected, queued]), canceled.promise]);
    assert.equal(status.bytes, 0);
    assert.equal(status.downloaded, false);
    const releasedStatus = await neko.queueStatus();
    assert.equal(releasedStatus.running, null);
    assert.equal(releasedStatus.pending, 0);
  } finally {
    context.signal.removeEventListener('abort', cancel);
    await neko.dispose();
    await Promise.all(work).catch(() => undefined);
    await rm(directory, { recursive: true, force: true });
  }
});

test('a report predicate that throws undefined retains its callback cause and releases worker admission', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const neko = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu', execution: 'worker' });
  const page: SDK.Page = { url: 'https://example.test/', paragraphs: [{ id: 'p1', text: 'The exhibit opens Monday.', source: { kind: 'html' } }], images: [] };
  try {
    await assert.rejects(neko.describe(page, { budget: { maxDurationMs: 1_000 }, sources: { paragraph: () => { throw undefined; } } }), (error: unknown) => error instanceof NekoError && error.code === 'OPERATION_FAILED' && Object.hasOwn(error, 'cause') && error.cause === undefined);
    assert.equal((await neko.cache.model.status()).bytes, 0);
    assert.equal((await neko.queueStatus()).running, null);
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('worker creation isolates parent-only Node execution flags', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const sdk = new URL('../../node/index.js', import.meta.url).href;
  const script = `
    import { createNeko } from ${JSON.stringify(sdk)};
    const neko = await createNeko({ cacheDir: ${JSON.stringify(directory)}, localFilesOnly: true, device: 'cpu', execution: 'worker' });
    try { console.log(JSON.stringify({ backend: await neko.backend.current(), cache: await neko.cache.model.status() })); }
    finally { await neko.dispose(); }
  `;
  try {
    const { stdout } = await promisify(execFile)(process.execPath, ['--stack-trace-limit=10', '--input-type=module', '-e', script], { timeout: 10_000 });
    const status = JSON.parse(stdout) as { backend: { runtime: string }; cache: { bytes: number } };
    assert.equal(status.backend.runtime, 'node');
    assert.equal(status.cache.bytes, 0);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('worker cold invalid inference and report options reject before model or source acquisition', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const decisions: string[] = [];
  const neko = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu', execution: 'worker', policy: {
    network: (_url, kind) => { decisions.push(kind); return kind === 'worker'; },
  } });
  try {
    for (const request of [
      () => neko.infer({ prompt: '' }),
      () => neko.infer({ prompt: 'preserved', maxNewTokens: 0 }),
      () => neko.infer({ prompt: 'preserved', validateDestination: false as never }),
      () => neko.inferStructured({ prompt: 'preserved', schema: { $ref: 'https://example.test/schema' } }),
      () => neko.planInference({ prompt: 'preserved', contextWindowTokens: 0 }),
      () => neko.describe('https://example.test/', { language: 'not_a_language' }),
    ]) {
      await assert.rejects(request(), (error: unknown) => error instanceof NekoError && error.stage === 'preprocess' && ['INVALID_INPUT', 'SCHEMA_INVALID'].includes(error.code));
    }
    assert.equal((await neko.cache.engine.status()).loaded, false);
    assert.equal((await neko.cache.model.status()).bytes, 0);
    assert.deepEqual(decisions, ['worker']);
    const controller = new AbortController(); controller.abort('cancel planning');
    await assert.rejects(neko.planInference({ prompt: '', signal: controller.signal }), (error: unknown) => error instanceof NekoError && error.code === 'ABORTED' && error.stage === 'preprocess');
    assert.equal((await neko.queueStatus()).running, null);
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('worker heartbeat and allowlisted diagnostics reveal no input, output, policy or cache path', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const neko = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu', execution: 'worker' });
  try {
    const health = await neko.health();
    assert.equal(health.healthy, true);
    assert.equal(health.execution.mode, 'worker');
    assert.ok(health.roundTripMs !== null && health.roundTripMs >= 0);
    const diagnostic = await neko.diagnostics();
    assert.equal(diagnostic.schemaVersion, 1);
    assert.equal(diagnostic.execution.workerId, health.execution.workerId);
    assert.equal(diagnostic.readiness.loaded, false);
    assert.equal(diagnostic.transport?.state, 'ready');
    const serialized = JSON.stringify(diagnostic);
    for (const privateField of ['prompt', 'messages', 'text', 'cacheDir', 'policy', 'profilePrefix']) assert.equal(Object.hasOwn(diagnostic, privateField), false);
    assert.equal(serialized.includes(directory), false);
    const abort = new AbortController(); abort.abort('health cancelled');
    await assert.rejects(neko.health({ signal: abort.signal }), (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('explicit worker restart rejects outstanding work and creates a fresh worker without replay', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const entered = Promise.withResolvers<void>();
  let decisions = 0;
  const neko = await createNeko({ cacheDir: directory, device: 'cpu', execution: 'worker', policy: {
    network: (_url, kind) => { if (kind === 'model') { decisions++; entered.resolve(); return new Promise<void>(() => undefined); } },
  } });
  try {
    const before = await neko.health();
    const active = neko.infer({ prompt: 'never replay this exact request' });
    const rejected = assert.rejects(active, (error: unknown) => error instanceof NekoError && error.code === 'WORKER_UNAVAILABLE');
    await entered.promise;
    const queued = neko.cache.model.status();
    const queuedRejected = assert.rejects(queued, (error: unknown) => error instanceof NekoError && error.code === 'WORKER_UNAVAILABLE');
    await neko.restart();
    await Promise.all([rejected, queuedRejected]);
    const after = await neko.health();
    assert.equal(after.healthy, true);
    assert.notEqual(after.execution.workerId, before.execution.workerId);
    assert.equal(decisions, 1);
    assert.equal(await neko.runtimeStatus(), null);
    assert.equal((await neko.cache.model.status()).bytes, 0);
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('a hard worker deadline terminates its realm and rejects all outstanding requests until explicit restart', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const entered = Promise.withResolvers<void>();
  const neko = await createNeko({ cacheDir: directory, device: 'cpu', execution: 'worker', policy: {
    network: (_url, kind) => { if (kind === 'model') { entered.resolve(); return new Promise<void>(() => undefined); } },
  } });
  try {
    const before = await neko.health();
    const active = neko.infer({ prompt: 'bounded native request', hardDeadlineMs: 500 });
    const rejected = assert.rejects(active, (error: unknown) => error instanceof NekoError && error.code === 'DEADLINE_EXCEEDED');
    await entered.promise;
    const queued = neko.cache.model.status();
    const queuedRejected = assert.rejects(queued, (error: unknown) => error instanceof NekoError && error.code === 'DEADLINE_EXCEEDED');
    await Promise.all([rejected, queuedRejected]);
    assert.equal((await neko.health()).reason, 'unavailable');
    await assert.rejects(neko.queueStatus(), (error: unknown) => error instanceof NekoError && error.code === 'WORKER_UNAVAILABLE');
    await neko.restart();
    assert.notEqual((await neko.health()).execution.workerId, before.execution.workerId);
    assert.equal((await neko.cache.model.status()).bytes, 0);
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('worker bundle transfer propagates missing assets and consumer source errors without buffering a model', { timeout: 30_000 }, async () => {
  const directory = await cacheDirectory();
  const neko = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu', execution: 'worker' });
  try {
    const reader = neko.cache.model.exportBundle().getReader();
    await assert.rejects(reader.read(), /installed asset/);
    reader.releaseLock();
    const source = new ReadableStream<Uint8Array>({ start(controller) { controller.error(new Error('bundle source failed')); } });
    await assert.rejects(neko.cache.model.importBundle(source), /bundle source failed/);
    assert.equal((await neko.cache.model.status()).bytes, 0);
    assert.equal((await neko.health()).healthy, true);
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});
