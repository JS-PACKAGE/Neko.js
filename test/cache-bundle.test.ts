import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, realpath, rm, readFile, writeFile, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { exportModelBundle, importModelBundle, type BundleBacking } from '../src/cache/bundle.js';
import { getRegisteredModelProfile, isRegisteredModelUrl, isRegisteredResolveCacheUrl } from '../src/cache/registry.js';
import { StreamingSha256 } from '../src/cache/sha256.js';
import { installVerifiedCache } from '../src/cache/model.js';

const file = 'generation_config.json';
const fixture = new TextEncoder().encode(JSON.stringify({
  bos_token_id: 248044, do_sample: true, eos_token_id: [248046, 248044], pad_token_id: 248044,
  temperature: 0.6, top_k: 20, top_p: 0.95, transformers_version: '5.3.0.dev0', trust_remote_code: false,
}, null, 2) + '\n');
// A real pinned metadata asset exercises framing/rollback without loading multi-gigabyte weights.
const selected = getRegisteredModelProfile();
const metadataProfile = { ...selected, files: { [file]: selected.files[file]! } };
function encodedBundle(value: unknown, payload: Uint8Array = fixture): Blob {
  const metadata = new TextEncoder().encode(JSON.stringify(value));
  const prefix = new Uint8Array(4); new DataView(prefix.buffer).setUint32(0, metadata.length);
  return new Blob([prefix, metadata, new Uint8Array(payload)]);
}
function bundleHeader() {
  return { format: 'neko-model-bundle', version: 1, model: selected.id, revision: selected.revision, profile: selected.profile,
    files: [{ name: file, ...selected.files[file]! }] };
}
async function backing(path: string): Promise<BundleBacking> {
  return { path,
    async match(name) {
      try { return new Response(new Uint8Array(await readFile(join(path, name)))).body!; }
      catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined; throw error; }
    },
    async put(name, body) { await writeFile(join(path, name), new Uint8Array(await new Response(body).arrayBuffer())); },
    async delete(name) { await rm(join(path, name), { force: true }); },
  };
}

test('offline bundle roundtrip preserves existing/unrelated entries and removes staging', async () => {
  const root = await mkdtemp(join(await realpath(tmpdir()), 'neko-bundle-roundtrip-'));
  try {
    const cache = await backing(root);
    await writeFile(join(root, 'unrelated.txt'), 'preserve');
    await importModelBundle(encodedBundle(bundleHeader()), metadataProfile, cache);
    assert.deepEqual(new Uint8Array(await readFile(join(root, file))), fixture);
    const exported = new Uint8Array(await new Response(exportModelBundle(metadataProfile, cache)).arrayBuffer());
    await importModelBundle(new Blob([exported]), metadataProfile, cache);
    assert.equal(await readFile(join(root, 'unrelated.txt'), 'utf8'), 'preserve');
    assert.deepEqual((await readdir(root)).sort(), [file, 'unrelated.txt']);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('transferred bundle chunks do not clone unrelated upstream bytes', async () => {
  const storage = new Uint8Array(1024 * 1024 + fixture.length + 37);
  storage.set(fixture, 37);
  const cache: BundleBacking = {
    path: 'transfer-regression',
    async match() {
      return new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(storage.subarray(37, 37 + fixture.length)); controller.close(); },
      });
    },
    async put() { throw new Error('Unexpected write'); },
    async delete() { throw new Error('Unexpected delete'); },
  };
  const source = exportModelBundle(metadataProfile, cache);
  const reader = structuredClone(source, { transfer: [source] }).getReader();
  let payload: Uint8Array | undefined;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      assert.equal(value.buffer.byteLength, value.byteLength);
      payload = value;
    }
    assert.deepEqual(payload, fixture);
  } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
});

test('offline bundles reject identity/path/digest/length corruption before changing installed entries', async () => {
  const root = await mkdtemp(join(await realpath(tmpdir()), 'neko-bundle-validation-'));
  try {
    const cache = await backing(root);
    await writeFile(join(root, file), fixture);
    const mutations = [
      { ...bundleHeader(), model: 'unregistered/model' },
      { ...bundleHeader(), revision: 'main' },
      { ...bundleHeader(), profile: 'all-q4' },
      { ...bundleHeader(), files: [{ ...bundleHeader().files[0], name: '../generation_config.json' }] },
      { ...bundleHeader(), files: [{ ...bundleHeader().files[0], sha256: '0'.repeat(64) }] },
      { ...bundleHeader(), files: [{ ...bundleHeader().files[0], size: fixture.length + 1 }] },
    ];
    for (const header of mutations) await assert.rejects(importModelBundle(encodedBundle(header), metadataProfile, cache));
    for (const bytes of [new Uint8Array(fixture.length), fixture.subarray(0, fixture.length - 1), new Uint8Array([...fixture, 0])]) {
      await assert.rejects(importModelBundle(encodedBundle(bundleHeader(), bytes), metadataProfile, cache));
    }
    assert.deepEqual(new Uint8Array(await readFile(join(root, file))), fixture);
    assert.deepEqual(await readdir(root), [file]);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('bundle cancellation cancels a blocked source and preserves the target cache', async () => {
  const root = await mkdtemp(join(await realpath(tmpdir()), 'neko-bundle-abort-'));
  const installation = await installVerifiedCache({ cacheDir: root, localFilesOnly: true });
  const controller = new AbortController();
  let cancelled = false;
  const source = new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } });
  const reason = new Error('consumer cancelled');
  try {
    const pending = installation.importBundle(source, controller.signal);
    controller.abort(reason);
    await assert.rejects(pending, (error) => error === reason);
    assert.equal(cancelled, true);
    assert.deepEqual(await readdir(root), []);
    const diagnostics = await installation.diagnostics();
    assert.equal(diagnostics.storage.quota, null);
    assert.equal(diagnostics.storage.usage, null);
    assert.equal(diagnostics.model, selected.id);
  } finally { installation.restore(); await rm(root, { recursive: true, force: true }); }
});

test('browser incremental SHA-256 matches native hashing across padding and chunk boundaries', () => {
  for (const length of [0, 1, 55, 56, 63, 64, 65, 127, 128, 1025]) {
    const bytes = Uint8Array.from({ length }, (_, i) => i % 251);
    for (const chunk of [1, 7, 64, 101]) {
      const hash = new StreamingSha256();
      for (let offset = 0; offset < bytes.length; offset += chunk) hash.update(bytes.subarray(offset, offset + chunk));
      assert.equal(hash.digest(), createHash('sha256').update(bytes).digest('hex'));
    }
  }
});

test('alternative model is distinct, immutable, architecturally compatible, and allowlisted only at its pinned revision', () => {
  const alternative = getRegisteredModelProfile('all-q4', 'onnx-community/Qwen3.5-2B-ONNX-OPT');
  assert.notEqual(alternative.id, selected.id);
  assert.equal(alternative.architecture, 'Qwen3_5ForConditionalGeneration');
  assert.equal(alternative.license, 'apache-2.0');
  assert.equal(alternative.files['onnx/decoder_model_merged_q4.onnx_data']!.size, 1207357440);
  assert.ok(Object.isFrozen(alternative.files));
  assert.equal(isRegisteredModelUrl(new URL(`${alternative.baseUrl}config.json`), alternative.files), true);
  assert.equal(isRegisteredModelUrl(new URL(`${alternative.baseUrl}config.json`), selected.files), false);
  assert.equal(isRegisteredModelUrl(new URL(`${alternative.baseUrl.replace(alternative.revision, 'main')}config.json`), alternative.files), false);
  assert.equal(isRegisteredModelUrl(new URL(`${alternative.baseUrl}../unregistered.bin`), alternative.files), false);
  assert.throws(() => getRegisteredModelProfile('default', 'arbitrary/model' as typeof selected.id));
});

test('Hub resolve-cache redirects are trusted only for the exact pinned model, revision and file', () => {
  const profile = getRegisteredModelProfile();
  const path = `/api/resolve-cache/models/${profile.id}/${profile.revision}/config.json`;
  assert.equal(isRegisteredResolveCacheUrl(new URL(`https://huggingface.co${path}?%2Frouting=data`), profile.files), true);
  assert.equal(isRegisteredResolveCacheUrl(new URL(`https://huggingface.co${path.replace(profile.revision, 'main')}`), profile.files), false);
  assert.equal(isRegisteredResolveCacheUrl(new URL(`https://huggingface.co${path.replace('config.json', 'unpinned.bin')}`), profile.files), false);
  assert.equal(isRegisteredResolveCacheUrl(new URL(`https://huggingface.co${path.replace('config.json', '../config.json')}`), profile.files), false);
  assert.equal(isRegisteredResolveCacheUrl(new URL(`https://example.com${path}`), profile.files), false);
  assert.equal(isRegisteredResolveCacheUrl(new URL(`http://huggingface.co${path}`), profile.files), false);
  assert.equal(isRegisteredResolveCacheUrl(new URL(`https://user@huggingface.co${path}`), profile.files), false);
});

test('corrupt installed assets are neither exportable nor replaced by bundle import', async () => {
  const root = await mkdtemp(join(await realpath(tmpdir()), 'neko-bundle-existing-integrity-'));
  try {
    const cache = await backing(root);
    const corrupt = new Uint8Array(fixture.length);
    await writeFile(join(root, file), corrupt);
    await assert.rejects(new Response(exportModelBundle(metadataProfile, cache)).arrayBuffer());
    await assert.rejects(importModelBundle(encodedBundle(bundleHeader()), metadataProfile, cache));
    assert.deepEqual(new Uint8Array(await readFile(join(root, file))), corrupt);
    assert.deepEqual(await readdir(root), [file]);
  } finally { await rm(root, { recursive: true, force: true }); }
});
