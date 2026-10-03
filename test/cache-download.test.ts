import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { lstat, mkdtemp, readFile, readdir, realpath, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import type { ServerResponse } from 'node:http';
import test from 'node:test';
import { env } from '@huggingface/transformers';
import { downloadResumable } from '../src/cache/download.js';
import type { DownloadProgress, ResumableDownloadOptions } from '../src/cache/download.js';
import { nodeDownloadStage } from '../src/cache/node-download.js';
import { withNodeInstallLock } from '../src/cache/lock.js';
import { installVerifiedCache } from '../src/cache/model.js';
import type { VerifiedCacheInstallation } from '../src/cache/model.js';
import { MODEL_ID, MODEL_REVISION } from '../src/cache/manifest.js';

const largeFixture = Buffer.alloc(2 * 1024 * 1024 + 37);
for (let index = 0; index < largeFixture.length; index++) largeFixture[index] = index % 251;
const sha256 = createHash('sha256').update(largeFixture).digest('hex');

async function localFixture() {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-resume-'));
  const destination = join(directory, 'verified.bin');
  const requests: { range: string | undefined; ifRange: string | undefined; path: string | undefined }[] = [];
  let mode: 'interrupt' | 'range' | 'changed' | 'invalid-range' | 'corrupt' = 'interrupt';
  let etag = '"fixture-v1"';
  let controller = new AbortController();
  const server = createServer((request, response) => {
    requests.push({ range: request.headers.range, ifRange: typeof request.headers['if-range'] === 'string' ? request.headers['if-range'] : undefined, path: request.url });
    const offset = request.headers.range ? Number(/^bytes=(\d+)-$/.exec(request.headers.range)?.[1]) : 0;
    const ranged = !!offset && mode !== 'changed';
    const bytes = mode === 'corrupt' ? Buffer.alloc(largeFixture.length, 255) : largeFixture;
    response.writeHead(ranged ? 206 : 200, {
      'content-length': String(bytes.length - (ranged ? offset : 0)), etag,
      ...(ranged ? { 'content-range': `bytes ${mode === 'invalid-range' ? offset + 1 : offset}-${bytes.length - 1}/${bytes.length}` } : {}),
    });
    if (mode === 'interrupt') response.write(bytes.subarray(0, 256 * 1024));
    else response.end(bytes.subarray(ranged ? offset : 0));
  });
  const listening = Promise.withResolvers<void>();
  server.once('error', listening.reject);
  server.listen(0, '127.0.0.1', () => listening.resolve());
  await listening.promise;
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const source = `http://127.0.0.1:${address.port}/model.bin`;
  const validate = async () => { const info = await lstat(directory); assert.ok(info.isDirectory() && !info.isSymbolicLink()); };
  const validateFile = async (path: string) => {
    try { const info = await lstat(path); assert.ok(info.isFile() && !info.isSymbolicLink()); }
    catch (error) { if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'ENOENT') throw error; }
  };
  const stage = await nodeDownloadStage({ directory, key: 'fixture', validate, validateFile, async destination() { return destination; } });
  const events: DownloadProgress[] = [];
  const options: ResumableDownloadOptions = {
    source, size: largeFixture.length, sha256, stage, signal: controller.signal,
    async fetch(headers, signal) { return { response: await fetch(options.source, { headers, ...(signal ? { signal } : {}) }), destination: options.source }; },
    onProgress(event) {
      events.push(event);
      if (mode === 'interrupt' && event.phase === 'download' && event.loaded >= 256 * 1024) controller.abort(new Error('fixture interrupted'));
    },
  };
  return {
    directory, destination, requests, stage, events, options,
    setMode(value: typeof mode) { mode = value; controller = new AbortController(); options.signal = controller.signal; },
    setEtag(value: string) { etag = value; },
    async close() {
      server.closeAllConnections();
      const closed = Promise.withResolvers<void>();
      server.close((error) => error ? closed.reject(error) : closed.resolve());
      await closed.promise;
      await rm(directory, { recursive: true, force: true });
    },
  };
}

test('interrupted real HTTP large downloads resume with Range and If-Range and a full pinned SHA-256', async () => {
  const fixture = await localFixture();
  try {
    await writeFile(join(fixture.directory, 'unrelated.txt'), 'preserve');
    await assert.rejects(downloadResumable(fixture.options));
    const partial = await fixture.stage.inspect();
    assert.ok(partial.size > 0 && partial.size < largeFixture.length);
    await assert.rejects(readFile(fixture.destination), { code: 'ENOENT' });
    fixture.setMode('range');
    await downloadResumable(fixture.options);
    assert.equal(fixture.requests[1]?.range, `bytes=${partial.size}-`);
    assert.equal(fixture.requests[1]?.ifRange, '"fixture-v1"');
    const installed = await readFile(fixture.destination);
    assert.equal(createHash('sha256').update(installed).digest('hex'), sha256);
    assert.deepEqual(installed, largeFixture);
    assert.ok(fixture.events.some((event) => event.phase === 'resume' && event.resumedFrom === partial.size));
    assert.equal((await fixture.stage.inspect()).size, 0);
    assert.equal(await readFile(join(fixture.directory, 'unrelated.txt'), 'utf8'), 'preserve');
  } finally { await fixture.close(); }
});

test('changed validator consumes one full 200 response and source changes never append old bytes', async () => {
  const fixture = await localFixture();
  try {
    await assert.rejects(downloadResumable(fixture.options));
    fixture.setMode('changed'); fixture.setEtag('"fixture-v2"');
    await downloadResumable(fixture.options);
    assert.ok(fixture.events.some((event) => event.resetReason === 'validator-changed'));
    assert.equal(fixture.requests.length, 2);
    assert.deepEqual(await readFile(fixture.destination), largeFixture);
    fixture.setMode('interrupt');
    await assert.rejects(downloadResumable(fixture.options));
    fixture.options.source += '?new-source';
    fixture.setMode('range');
    await downloadResumable(fixture.options);
    assert.equal(fixture.requests.at(-1)?.range, undefined);
    assert.ok(fixture.events.some((event) => event.resetReason === 'source-changed'));
  } finally { await fixture.close(); }
});

test('invalid ranges and final hash failures discard only owned staging, without retry or replacing installed bytes', async () => {
  const fixture = await localFixture();
  try {
    await writeFile(fixture.destination, 'valid old install');
    await writeFile(join(fixture.directory, 'unrelated.txt'), 'preserve');
    await assert.rejects(downloadResumable(fixture.options));
    fixture.setMode('invalid-range');
    await assert.rejects(downloadResumable(fixture.options), /invalid resume response/);
    assert.equal(fixture.requests.length, 2);
    assert.equal((await fixture.stage.inspect()).size, 0);
    fixture.setMode('corrupt');
    await assert.rejects(downloadResumable(fixture.options), /SHA-256/);
    assert.equal((await fixture.stage.inspect()).size, 0);
    assert.equal(await readFile(fixture.destination, 'utf8'), 'valid old install');
    assert.equal(await readFile(join(fixture.directory, 'unrelated.txt'), 'utf8'), 'preserve');
  } finally { await fixture.close(); }
});

test('two independent processes coordinate one real HTTP installation without corrupting or duplicating downloads', async () => {
  const fixture = await localFixture();
  fixture.setMode('range');
  try {
    const owner = fileURLToPath(new URL('./fixtures/cache-download-owner.js', import.meta.url));
    const run = () => {
      const ready = Promise.withResolvers<void>();
      const result = Promise.withResolvers<void>();
      const child = spawn(process.execPath, [owner, fixture.directory, fixture.options.source, String(largeFixture.length), sha256], { stdio: ['ignore', 'ignore', 'pipe', 'ipc'] });
      let errors = '';
      child.stderr!.on('data', (bytes: Buffer) => { errors += bytes.toString(); });
      child.once('error', result.reject);
      child.once('message', () => ready.resolve());
      child.once('exit', (code) => { if (code === 0) result.resolve(); else result.reject(new Error(`Owner failed (${code}): ${errors}`)); });
      return { child, done: result.promise, ready: ready.promise };
    };
    const owners = [run(), run()];
    await Promise.all(owners.map((owner) => owner.ready));
    for (const owner of owners) owner.child.send('start');
    await Promise.all(owners.map((owner) => owner.done));
    assert.equal(fixture.requests.length, 1);
    assert.deepEqual(await readFile(fixture.destination), largeFixture);
    assert.deepEqual(await readdir(fixture.directory), ['verified.bin']);
  } finally { await fixture.close(); }
});

const generationFixture = JSON.stringify({
  bos_token_id: 248044, do_sample: true, eos_token_id: [248046, 248044], pad_token_id: 248044,
  temperature: 0.6, top_k: 20, top_p: 0.95, transformers_version: '5.3.0.dev0', trust_remote_code: false,
}, null, 2) + '\n';

test('verified installation preserves interrupted staging across owners and clears only its own pinned staging', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-install-resume-'));
  const requests: (string | undefined)[] = [];
  let interrupt = true;
  const server = createServer((request, response) => {
    requests.push(request.headers.range);
    const offset = Number(/^bytes=(\d+)-$/.exec(request.headers.range ?? '')?.[1] ?? 0);
    response.writeHead(offset ? 206 : 200, {
      'content-length': String(Buffer.byteLength(generationFixture) - offset), etag: '"generation"',
      ...(offset ? { 'content-range': `bytes ${offset}-${Buffer.byteLength(generationFixture) - 1}/${Buffer.byteLength(generationFixture)}` } : {}),
    });
    if (interrupt) response.write(generationFixture.slice(0, 64));
    else response.end(generationFixture.slice(offset));
  });
  const listening = Promise.withResolvers<void>();
  server.listen(0, '127.0.0.1', () => listening.resolve()); await listening.promise;
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const baseUrl = `http://127.0.0.1:${address.port}/`;
  const originalFetch = env.fetch;
  env.fetch = globalThis.fetch;
  let installation: VerifiedCacheInstallation | undefined;
  try {
    await writeFile(join(directory, '.neko-stage-unrelated.part'), 'preserve');
    const controller = new AbortController();
    installation = await installVerifiedCache({ cacheDir: directory, modelSource: { baseUrl }, onProgress(event) { if (event.phase === 'download' && event.loaded >= 64) controller.abort(new Error('interrupt fixture')); } });
    await assert.rejects(installation.prefetch(controller.signal, ['generation_config.json']), /interrupt fixture/);
    installation.restore();
    interrupt = false;
    installation = await installVerifiedCache({ cacheDir: directory, modelSource: { baseUrl } });
    await installation.prefetch(undefined, ['generation_config.json']);
    assert.deepEqual(requests, [undefined, 'bytes=64-']);
    assert.equal(await readFile(join(directory, MODEL_ID, MODEL_REVISION, 'generation_config.json'), 'utf8'), generationFixture);
    await installation.clear();
    assert.equal(await readFile(join(directory, '.neko-stage-unrelated.part'), 'utf8'), 'preserve');
  } finally {
    installation?.restore(); env.fetch = originalFetch;
    server.closeAllConnections();
    const closed = Promise.withResolvers<void>(); server.close(() => closed.resolve()); await closed.promise;
    await rm(directory, { recursive: true, force: true });
  }
});

test('hard-terminated worker lock owners are recoverable while the same Node process remains alive', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-terminated-lock-'));
  const worker = new Worker(new URL('./fixtures/cache-lock-worker.js', import.meta.url), { workerData: { directory } });
  try {
    const locked = Promise.withResolvers<void>();
    worker.once('message', () => locked.resolve());
    worker.once('error', locked.reject);
    await locked.promise;
    await worker.terminate();
    let acquired = false;
    await withNodeInstallLock({
      directory, key: 'terminated-worker',
      async validate() { const info = await lstat(directory); assert.ok(info.isDirectory() && !info.isSymbolicLink()); },
      async validateFile(path) {
        try { const info = await lstat(path); assert.ok(info.isFile() && !info.isSymbolicLink()); }
        catch (error) { if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'ENOENT') throw error; }
      },
    }, async () => { acquired = true; });
    assert.equal(acquired, true);
    assert.deepEqual(await readdir(directory), []);
  } finally { await worker.terminate(); await rm(directory, { recursive: true, force: true }); }
});

test('prefetch bounds concurrent HTTP requests and cancels siblings without automatic retries', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-bounded-download-'));
  let requests = 0;
  const held: ServerResponse[] = [];
  const server = createServer((_request, response) => {
    requests++;
    held.push(response);
    if (held.length === 2) for (const pending of held) pending.writeHead(503).end();
  });
  const listening = Promise.withResolvers<void>();
  server.listen(0, '127.0.0.1', () => listening.resolve()); await listening.promise;
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const originalFetch = env.fetch; env.fetch = globalThis.fetch;
  const installation = await installVerifiedCache({ cacheDir: directory, modelSource: { baseUrl: `http://127.0.0.1:${address.port}/` }, downloadConcurrency: 2 });
  try {
    await assert.rejects(installation.prefetch(undefined, ['config.json', 'generation_config.json', 'tokenizer.json']), /HTTP 503/);
    assert.equal(requests, 2);
  } finally {
    installation.restore(); env.fetch = originalFetch; server.closeAllConnections();
    const closed = Promise.withResolvers<void>(); server.close(() => closed.resolve()); await closed.promise;
    await rm(directory, { recursive: true, force: true });
  }
});

test('parallel Node signal scopes do not exchange cancellation and nested scopes stay reentrant', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-signal-scopes-'));
  const installation = await installVerifiedCache({ cacheDir: directory, localFilesOnly: true });
  const first = new AbortController();
  const second = new AbortController();
  const entered = Promise.withResolvers<void>();
  const resume = Promise.withResolvers<void>();
  try {
    const pending = installation.withSignal(first.signal, async () => {
      entered.resolve(); await resume.promise;
      await assert.rejects(env.fetch('https://huggingface.co/unpinned'), (error: unknown) => error === first.signal.reason);
    });
    await entered.promise;
    await installation.withSignal(second.signal, async () => {
      first.abort(new Error('first scope canceled'));
      await installation.withSignal(second.signal, async () => {
        await assert.rejects(env.fetch('https://huggingface.co/unpinned'), /Unpinned resource URL/);
      });
    });
    resume.resolve(); await pending;
  } finally { resume.resolve(); installation.restore(); await rm(directory, { recursive: true, force: true }); }
});

