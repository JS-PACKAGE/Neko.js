import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, realpath, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { AutoConfig, env } from '@huggingface/transformers';
import { installVerifiedCache, ModelIntegrityError } from '../src/cache/model.js';
import { MODEL_FILES, MODEL_ID, MODEL_REVISION, modelFileUrl } from '../src/cache/manifest.js';
import { EngineCache } from '../src/cache/engine.js';
import { NekoError } from '../src/errors.js';

test('corrupt native cache content cannot reach the Transformers config loader or trigger a replacement download', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-onnx-integrity-'));
  const root = join(directory, MODEL_ID, MODEL_REVISION);
  const originalFetch = env.fetch;
  let fetched = false;
  env.fetch = async () => { fetched = true; throw new Error('unexpected download'); };
  await mkdir(root, { recursive: true });
  await writeFile(join(root, 'config.json'), 'x'.repeat(MODEL_FILES['config.json'].size));
  await writeFile(join(directory, 'old-model.gguf'), 'preserve');
  const installation = await installVerifiedCache({ cacheDir: directory });
  try {
    await assert.rejects(AutoConfig.from_pretrained(MODEL_ID, { revision: MODEL_REVISION }), ModelIntegrityError);
    assert.equal(fetched, false);
    assert.equal(await readFile(join(directory, 'old-model.gguf'), 'utf8'), 'preserve');
  } finally { installation.restore(); env.fetch = originalFetch; await rm(directory, { recursive: true, force: true }); }
});

test('bounded native cache writes reject oversized and hash-mismatched resources without promoting an entry', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-onnx-bounds-'));
  const installation = await installVerifiedCache({ cacheDir: directory });
  const file = 'generation_config.json';
  const spec = MODEL_FILES[file];
  try {
    for (const size of [spec.size + 1, spec.size]) {
      await assert.rejects(env.customCache!.put(modelFileUrl(file), new Response('x'.repeat(size))), ModelIntegrityError);
      assert.deepEqual(await readdir(directory), []);
    }
  } finally { installation.restore(); await rm(directory, { recursive: true, force: true }); }
});

test('native cache rejects symlink ancestors and model entries without reading or modifying their targets', async () => {
  const root = await mkdtemp(join(await realpath(tmpdir()), 'neko-onnx-links-'));
  try {
    const target = join(root, 'target');
    await writeFile(target, 'user data');
    await symlink(root, join(root, 'linked'), 'dir');
    await assert.rejects(installVerifiedCache({ cacheDir: join(root, 'linked', 'cache') }), ModelIntegrityError);
    const directory = join(root, 'cache');
    const entryRoot = join(directory, MODEL_ID, MODEL_REVISION);
    await mkdir(entryRoot, { recursive: true });
    await symlink(target, join(entryRoot, 'config.json'));
    const installation = await installVerifiedCache({ cacheDir: directory, localFilesOnly: true });
    try {
      await assert.rejects(AutoConfig.from_pretrained(MODEL_ID, { revision: MODEL_REVISION, local_files_only: true }), ModelIntegrityError);
      assert.equal(await readFile(target, 'utf8'), 'user data');
      assert.equal((await stat(directory)).mode & 0o777, 0o700);
    } finally { installation.restore(); installation.restore(); }
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('metadata ranges and HEAD require verified cached bytes without promoting partial responses or fetching offline misses', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-onnx-metadata-'));
  const root = join(directory, MODEL_ID, MODEL_REVISION);
  const file = 'generation_config.json';
  const entry = join(root, file);
  const fixture = JSON.stringify({
    bos_token_id: 248044, do_sample: true, eos_token_id: [248046, 248044], pad_token_id: 248044,
    temperature: 0.6, top_k: 20, top_p: 0.95, transformers_version: '5.3.0.dev0', trust_remote_code: false,
  }, null, 2) + '\n';
  const originalFetch = env.fetch;
  let networkRequests = 0;
  env.fetch = async () => { networkRequests++; throw new Error('unexpected network'); };
  await mkdir(root, { recursive: true });
  await writeFile(entry, fixture);
  const installation = await installVerifiedCache({ cacheDir: directory, localFilesOnly: true });
  const request = modelFileUrl(file);
  try {
    const range = await env.fetch(request, { headers: { Range: 'bytes=0-0' } }) as Response;
    const partial = range.clone();
    assert.equal(range.status, 206);
    assert.equal(range.headers.get('content-range'), 'bytes 0-0/248');
    assert.equal(range.headers.get('content-length'), '1');
    assert.equal(await range.text(), '{');
    await assert.rejects(env.customCache!.put(request, partial), ModelIntegrityError);
    assert.equal(await readFile(entry, 'utf8'), fixture);
    const head = await env.fetch(request, { method: 'HEAD' }) as Response;
    assert.equal(head.status, 200);
    assert.equal(head.headers.get('content-length'), '248');
    assert.equal(head.body, null);
    await writeFile(entry, 'x'.repeat(MODEL_FILES[file].size));
    await assert.rejects(env.fetch(request, { headers: { Range: 'bytes=0-0' } }), ModelIntegrityError);
    await assert.rejects(env.fetch(request, { method: 'HEAD' }), ModelIntegrityError);
    await rm(entry);
    await assert.rejects(env.fetch(request, { headers: { Range: 'bytes=0-0' } }), Error);
    assert.equal(networkRequests, 0);
  } finally { installation.restore(); env.fetch = originalFetch; await rm(directory, { recursive: true, force: true }); }
});

test('opaque browser redirects fail closed with a typed policy error and never promote a cache entry', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-opaque-redirect-'));
  const originalFetch = env.fetch;
  let requests = 0;
  env.fetch = async () => {
    requests++;
    return { type: 'opaqueredirect', status: 0, body: null } as Response;
  };
  const installation = await installVerifiedCache({ cacheDir: directory });
  try {
    await assert.rejects(env.fetch(modelFileUrl('onnx/embed_tokens_q4.onnx')), (error: unknown) => {
      assert.ok(error instanceof NekoError);
      assert.equal(error.code, 'POLICY_DENIED');
      assert.equal(error.stage, 'cache');
      return true;
    });
    assert.equal(requests, 1);
    assert.deepEqual(await readdir(directory), []);
  } finally { installation.restore(); env.fetch = originalFetch; await rm(directory, { recursive: true, force: true }); }
});

test('invalid explicitly provided model sources reject instead of falling back to remote model downloads', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-model-source-input-'));
  try {
    for (const source of [false, null, [], {}, { baseUrl: 42 }, { baseUrl: '/models/' }, { baseUrl: 'https://example.com/models/?token=secret' }, { baseUrl: 'https://user:password@example.com/models/' }]) {
      await assert.rejects(installVerifiedCache({ cacheDir: directory, modelSource: source } as Parameters<typeof installVerifiedCache>[0]), TypeError);
    }
    const installation = await installVerifiedCache({ cacheDir: directory, localFilesOnly: true });
    installation.restore();
    assert.deepEqual(await readdir(directory), []);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('an explicit model mirror preserves pinned integrity and canonical cache keys without trusting sibling assets or redirect hops', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-model-mirror-'));
  const originalFetch = env.fetch;
  const file = 'generation_config.json';
  const canonical = modelFileUrl(file);
  const mirror = 'http://127.0.0.1:8787/models/pinned/';
  const fixture = JSON.stringify({
    bos_token_id: 248044, do_sample: true, eos_token_id: [248046, 248044], pad_token_id: 248044,
    temperature: 0.6, top_k: 20, top_p: 0.95, transformers_version: '5.3.0.dev0', trust_remote_code: false,
  }, null, 2) + '\n';
  let mode: 'verified' | 'corrupt' | 'redirect' = 'verified';
  const requests: string[] = [];
  env.fetch = async (url) => {
    requests.push(String(url));
    if (mode === 'redirect') return new Response(null, { status: 302, headers: { location: `${mirror}unregistered.bin` } });
    return new Response(mode === 'corrupt' ? 'x'.repeat(MODEL_FILES[file].size) : fixture);
  };
  const installation = await installVerifiedCache({ cacheDir: directory, modelSource: { baseUrl: mirror }, profile: 'all-q4' });
  try {
    const downloaded = await env.fetch(canonical) as Response;
    await env.customCache!.put(canonical, downloaded);
    assert.equal(await readFile(join(directory, MODEL_ID, MODEL_REVISION, file), 'utf8'), fixture);
    assert.equal(await (await env.customCache!.match(canonical) as Response).text(), fixture);
    mode = 'corrupt';
    await assert.rejects(env.fetch(canonical), ModelIntegrityError);
    mode = 'redirect';
    await assert.rejects(env.fetch(canonical), (error: unknown) => {
      assert.ok(error instanceof Error && 'code' in error);
      assert.equal(error.code, 'POLICY_DENIED');
      return true;
    });
    await assert.rejects(env.fetch(modelFileUrl('onnx/vision_encoder_fp16.onnx')), ModelIntegrityError);
    assert.deepEqual(requests, [`${mirror}${file}`, `${mirror}${file}`, `${mirror}${file}`]);
    assert.equal(await readFile(join(directory, MODEL_ID, MODEL_REVISION, file), 'utf8'), fixture);
  } finally { installation.restore(); env.fetch = originalFetch; await rm(directory, { recursive: true, force: true }); }
});

test('a never-settling model network approval is bounded by the active operation signal before fetching', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-model-approval-'));
  const originalFetch = env.fetch;
  let fetched = false;
  let beginApproval: () => void = () => {};
  const started = new Promise<void>((resolve) => { beginApproval = resolve; });
  env.fetch = async () => { fetched = true; throw new Error('unexpected network'); };
  const installation = await installVerifiedCache({
    cacheDir: directory,
    policy: { network: async () => { beginApproval(); await new Promise<void>(() => {}); } },
  });
  const controller = new AbortController();
  const reason = new NekoError('Duration budget expired', 'generate', 'BUDGET_EXCEEDED');
  try {
    const pending = installation.withSignal(controller.signal, async () => {
      await env.fetch(modelFileUrl('onnx/embed_tokens_q4.onnx'));
    });
    await started;
    controller.abort(reason);
    await assert.rejects(pending, (error: unknown) => error === reason);
    assert.equal(fetched, false);
  } finally { installation.restore(); env.fetch = originalFetch; await rm(directory, { recursive: true, force: true }); }
});


test('release waits for in-flight work and queued requests reuse one engine', async () => {
  let releaseOperation: () => void = () => {};
  let began: () => void = () => {};
  const started = new Promise<void>((resolve) => { began = resolve; });
  const blocked = new Promise<void>((resolve) => { releaseOperation = resolve; });
  let disposed = false;
  const engine = { memory: 42, dispose: async () => { disposed = true; } };
  const cache = new EngineCache(async () => engine);
  const first = cache.use(async () => { began(); await blocked; assert.equal(disposed, false); return 'first'; });
  await started;
  const second = cache.use(async () => { assert.equal(disposed, false); return 'second'; });
  const release = cache.release();
  assert.equal(disposed, false);
  releaseOperation();
  assert.deepEqual(await Promise.all([first, second]), ['first', 'second']);
  await release;
  assert.equal(disposed, true);
  assert.deepEqual(cache.status(), { loaded: false, sessions: 0, memory: 0, hits: 1, loads: 1 });
});

test('disabled engine caching disposes at each request boundary', async () => {
  let disposed = 0;
  const cache = new EngineCache(async () => ({ memory: 1, dispose: async () => { disposed++; } }), false);
  await cache.use(async () => 'a');
  await cache.use(async () => 'b');
  assert.equal(disposed, 2);
  assert.equal(cache.status().loads, 2);
  assert.equal(cache.status().hits, 0);
  await cache.release();
});

test('idle TTL renews on use and expires only after the renewed boundary', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  let disposals = 0;
  const cache = new EngineCache(async () => ({ memory: 1, dispose: async () => { disposals++; } }), true, 100);
  await cache.use(async () => 'first');
  context.mock.timers.tick(90);
  await cache.use(async () => 'second');
  context.mock.timers.tick(20);
  await Promise.resolve();
  assert.equal(disposals, 0);
  assert.equal(cache.status().loaded, true);
  context.mock.timers.tick(80);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(disposals, 1);
  assert.equal(cache.status().loaded, false);
  await cache.release();
});
