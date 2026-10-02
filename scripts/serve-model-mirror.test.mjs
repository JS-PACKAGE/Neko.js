import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { MODEL_ID, MODEL_REVISION, MODEL_FILES } from '../dist/src/cache/manifest.js';
import { createModelMirror } from './serve-model-mirror.mjs';

test('model mirror refuses same-sized corrupt pinned assets before opening a listening socket', async () => {
  const root = await mkdtemp(join(await realpath(tmpdir()), 'neko-mirror-integrity-'));
  const directory = join(root, MODEL_ID, MODEL_REVISION, 'onnx');
  const name = 'onnx/embed_tokens_q4.onnx';
  await mkdir(directory, { recursive: true });
  const entry = join(root, MODEL_ID, MODEL_REVISION, name);
  const corrupt = Buffer.alloc(MODEL_FILES[name].size, 0x78);
  await writeFile(entry, corrupt);
  try {
    await assert.rejects(createModelMirror({ cacheDir: root }), Error);
    assert.deepEqual(await readFile(entry), corrupt);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('model mirror rejects symlink cache roots and entries without serving or modifying their targets', async () => {
  const root = await mkdtemp(join(await realpath(tmpdir()), 'neko-mirror-links-'));
  try {
    const cache = join(root, 'cache');
    const directory = join(cache, MODEL_ID, MODEL_REVISION, 'onnx');
    await mkdir(directory, { recursive: true });
    await symlink(cache, join(root, 'linked'), 'dir');
    await assert.rejects(createModelMirror({ cacheDir: join(root, 'linked') }), Error);
    const target = join(root, 'user-data');
    await writeFile(target, 'preserve');
    await symlink(target, join(directory, 'embed_tokens_q4.onnx'));
    await assert.rejects(createModelMirror({ cacheDir: cache }), Error);
    assert.equal(await readFile(target, 'utf8'), 'preserve');
  } finally { await rm(root, { recursive: true, force: true }); }
});
