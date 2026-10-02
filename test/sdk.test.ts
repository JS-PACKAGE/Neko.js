import assert from 'node:assert/strict';
import { mkdtemp, realpath, rm, writeFile, mkdir, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createNeko, MODEL_ID, MODEL_REVISION, NekoError } from '../src/index.js';

test('lazy SDK owns one runtime, scoped clearing preserves unrelated files, and concurrent disposal is awaited', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-sdk-'));
  const neko = await createNeko({ cacheDir: directory, localFilesOnly: true, device: 'cpu' });
  try {
    assert.equal(neko.cache.engine.status().loaded, false);
    const status = await neko.cache.model.status();
    assert.equal(status.downloaded, false); assert.equal(status.verified, false); assert.equal(status.bytes, 0); assert.equal(status.path, directory);
    await assert.rejects(createNeko({ cacheDir: directory, device: 'cpu' }), (error: unknown) => error instanceof NekoError && error.code === 'RUNTIME_BUSY');
    const namespace = join(directory, MODEL_ID, MODEL_REVISION);
    await mkdir(namespace, { recursive: true });
    await writeFile(join(namespace, 'notes.txt'), 'preserve namespace neighbor');
    await writeFile(join(directory, 'user-data.txt'), 'preserve root neighbor');
    await writeFile(join(namespace, 'config.json'), 'corrupt pinned entry');
    await neko.cache.model.clear();
    assert.equal(await readFile(join(namespace, 'notes.txt'), 'utf8'), 'preserve namespace neighbor');
    assert.equal(await readFile(join(directory, 'user-data.txt'), 'utf8'), 'preserve root neighbor');
    const first = neko.dispose(); const second = neko.dispose(); assert.equal(first, second); await second;
    await assert.rejects(neko.infer({ prompt: 'ignored' }), (error: unknown) => error instanceof NekoError && error.code === 'DISPOSED');
    const next = await createNeko({ cacheDir: directory, device: 'cpu' }); await next.dispose();
  } finally { await neko.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('cancelling a factory after runtime installation releases ownership before rejecting', async () => {
  const controller = new AbortController();
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-abort-create-'));
  try {
    await assert.rejects(createNeko({
      cacheDir: directory, device: 'cpu', signal: controller.signal,
      cache: { get engineTtlMs() { controller.abort('cancelled during factory configuration'); return 0; } },
    }), (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
    const next = await createNeko({ cacheDir: directory, device: 'cpu' }); await next.dispose();
  } finally { await rm(directory, { recursive: true, force: true }); }
});
