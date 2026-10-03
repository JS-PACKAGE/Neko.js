import assert from 'node:assert/strict';
import { mkdtemp, realpath, rm, writeFile } from 'node:fs/promises';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';
import sharp from 'sharp';
import { readImage, loadImage, prepareImage, prepareImageRegions, ImagePreprocessCache, normalizeImageRegion, tileImageRegions, validateImageObservation } from '../../src/web/index.js';
import { prepareImageInternal } from '../../src/web/image.js';
import type { DecodedImage } from '../../src/web/image.js';

test('Node image paths, file URLs, blobs, object URLs and data URLs decode real pixels with the same bounds', async (t) => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-image-'));
  const bytes = await sharp({ create: { width: 24, height: 16, channels: 3, background: '#ff0000' } }).png().toBuffer();
  const path = join(directory, 'photo.png'); await writeFile(path, bytes);
  try {
    const blob = new Blob([new Uint8Array(bytes)], { type: 'image/png' });
    const objectUrl = URL.createObjectURL(blob);
    t.after(() => URL.revokeObjectURL(objectUrl));
    for (const input of [relative(process.cwd(), path), pathToFileURL(path), blob, objectUrl, `data:image/png;base64,${bytes.toString('base64')}`]) {
      const image = await readImage(input);
      assert.equal(image.width, 24); assert.equal(image.height, 16); assert.deepEqual([...image.data.subarray(0, 3)], [255, 0, 0]);
      await assert.rejects(readImage(input, { maxImageBytes: bytes.length - 1 }), /byte limit/);
    }
    const normalized = await loadImage({ id: 'i1', url: path, discoveredBy: ['img'] });
    const metadata = await sharp(normalized.data).metadata(); assert.equal(metadata.format, 'png'); assert.equal(metadata.width, 24);
    await assert.rejects(readImage(new Blob([new Uint8Array(bytes)], { type: 'image/jpeg' })), /raster content type/);
    const controller = new AbortController(); controller.abort(); await assert.rejects(readImage(path, { signal: controller.signal }), { name: 'AbortError' });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('decoded pixel bounds reject invalid raw buffers before image processing', async () => {
  await assert.rejects(readImage({ data: new Uint8Array(3), width: 2, height: 2, channels: 3 }), /invalid pixel data/);
  await assert.rejects(readImage({ data: new Uint8Array(3), width: 50_000, height: 50_000, channels: 3 }), /pixel limit/);
});

test('source-space crops retain real pixels and stable per-region provenance before resizing', async () => {
  const pixels = new Uint8Array(8 * 4 * 3);
  for (let y = 0; y < 4; y++) for (let x = 0; x < 8; x++) {
    const offset = (y * 8 + x) * 3; pixels[offset + (x < 4 ? 0 : 2)] = 255;
  }
  const source: DecodedImage = { data: pixels, width: 8, height: 4, channels: 3 };
  const encoded = await sharp(pixels, { raw: { width: 8, height: 4, channels: 3 } }).png().toBuffer();
  for (const input of [source, new Blob([new Uint8Array(encoded)], { type: 'image/png' })]) {
    const left = await prepareImage(input, { region: { unit: 'normalized', x: 0, y: 0, width: 0.5, height: 1 } });
    const right = await prepareImage(input, { region: { unit: 'pixels', x: 4, y: 0, width: 4, height: 4 }, maxDimension: 2 });
    assert.deepEqual([...left.image.data.subarray(0, 3)], [255, 0, 0]);
    assert.deepEqual([...right.image.data.subarray(0, 3)], [0, 0, 255]);
    assert.equal(left.observation.sourceVersionId, right.observation.sourceVersionId);
    assert.notEqual(left.observation.versionId, right.observation.versionId);
    assert.deepEqual(right.observation.region, { unit: 'pixels', x: 4, y: 0, width: 4, height: 4 });
    assert.deepEqual(right.observation.normalizedRegion, { unit: 'normalized', x: 0.5, y: 0, width: 0.5, height: 1 });
    assert.equal(right.observation.sourceWidth, 8); assert.equal(right.observation.sourceHeight, 4);
    assert.equal(right.observation.width, 2); assert.equal(right.observation.height, 2);
    validateImageObservation(left.observation); validateImageObservation(right.observation);
    assert.throws(() => validateImageObservation({ ...right.observation, normalizedRegion: { ...right.observation.normalizedRegion, x: 0 } }), /normalized bounds/);
    assert.throws(() => validateImageObservation({ ...right.observation, sourceVersionId: 'caller-label' }), /SHA-256/);
  }
  assert.deepEqual(normalizeImageRegion(8, 4, { unit: 'normalized', x: 0.125, y: 0.25, width: 0.0625, height: 0.125 }), { unit: 'pixels', x: 1, y: 1, width: 1, height: 1 });
  await assert.rejects(readImage(source, { region: { unit: 'pixels', x: 7, y: 0, width: 2, height: 1 } }), /within/);
  await assert.rejects(readImage(source, { region: { unit: 'normalized', x: -0.1, y: 0, width: 0.5, height: 1 } }), /offsets/);
});

test('source pixel crops occur before the normalized full-image dimension limit', async () => {
  const source: DecodedImage = { data: new Uint8Array(3000 * 6 * 3), width: 3000, height: 6, channels: 3 };
  for (let y = 0; y < 6; y++) for (let x = 2900; x < 3000; x++) source.data[(y * 3000 + x) * 3 + 2] = 255;
  const bytes = await sharp(source.data, { raw: { width: source.width, height: source.height, channels: source.channels } }).png().toBuffer();
  for (const input of [source, new Blob([new Uint8Array(bytes)], { type: 'image/png' })]) {
    const prepared = await prepareImage(input, { region: { unit: 'pixels', x: 2936, y: 0, width: 64, height: 6 } });
    assert.equal(prepared.observation.sourceWidth, 3000); assert.equal(prepared.image.width, 64); assert.equal(prepared.image.height, 6);
    assert.deepEqual([...prepared.image.data.subarray(0, 3)], [0, 0, 255]); assert.equal(prepared.observation.preprocessing.resize, 'none');
  }
});

test('encoded image crops use EXIF-oriented source dimensions', async () => {
  const bytes = await sharp({ create: { width: 8, height: 4, channels: 3, background: '#ff0000' } }).withMetadata({ orientation: 6 }).jpeg().toBuffer();
  const prepared = await prepareImage(new Blob([new Uint8Array(bytes)], { type: 'image/jpeg' }), { region: { unit: 'pixels', x: 0, y: 4, width: 4, height: 4 } });
  assert.equal(prepared.observation.sourceWidth, 4); assert.equal(prepared.observation.sourceHeight, 8);
  assert.equal(prepared.image.width, 4); assert.equal(prepared.image.height, 4);
  assert.equal(prepared.observation.preprocessing.orientation, 'from-image');
  validateImageObservation(prepared.observation);
});

test('tiling covers source regions without dropping edge pixels or exceeding count and overlap bounds', async () => {
  const regions = tileImageRegions(10, 6, { tileWidth: 4, tileHeight: 4, overlap: 0.25, maxTiles: 6 });
  assert.equal(regions.length, 6);
  assert.deepEqual(regions.map(({ x, y, width, height }) => [x, y, width, height]), [[0, 0, 4, 4], [3, 0, 4, 4], [6, 0, 4, 4], [0, 3, 4, 3], [3, 3, 4, 3], [6, 3, 4, 3]]);
  const covered = new Uint8Array(60);
  for (const region of regions) for (let y = region.y; y < region.y + region.height; y++) for (let x = region.x; x < region.x + region.width; x++) covered[y * 10 + x] = 1;
  assert.ok(covered.every((pixel) => pixel === 1));
  assert.throws(() => tileImageRegions(10, 6, { tileWidth: 4, tileHeight: 4, overlap: 0.25, maxTiles: 5 }), /exceeding maxTiles/);
  assert.throws(() => tileImageRegions(10, 6, { tileWidth: 4, tileHeight: 4, overlap: 0.6 }), /overlap/);
  const source: DecodedImage = { data: new Uint8Array(60 * 3).fill(123), width: 10, height: 6, channels: 3 };
  const prepared = await prepareImageRegions(source, { region: { unit: 'pixels', x: 2, y: 1, width: 6, height: 4 }, tiling: { tileWidth: 3, tileHeight: 4, overlap: 0, maxTiles: 2 } });
  assert.equal(prepared.length, 2);
  assert.deepEqual(prepared.map(({ observation }) => observation.region), [{ unit: 'pixels', x: 2, y: 1, width: 3, height: 4 }, { unit: 'pixels', x: 5, y: 1, width: 3, height: 4 }]);
  assert.equal(prepared[0]!.observation.sourceVersionId, prepared[1]!.observation.sourceVersionId);
  assert.notEqual(prepared[0]!.observation.versionId, prepared[1]!.observation.versionId);
  for (const item of prepared) validateImageObservation(item.observation);
  await assert.rejects(readImage(source, { tiling: { tileWidth: 3, tileHeight: 3 } }), /prepareImageRegions/);
});

test('bounded pixel caches key owned content and preprocessing options, not mutable caller identity', async () => {
  const cache = new ImagePreprocessCache({ maxBytes: 12, maxEntries: 1 });
  const source: DecodedImage = { data: new Uint8Array(12).fill(255), width: 2, height: 2, channels: 3 };
  const firstPromise = prepareImage(source, {}, cache);
  source.data.fill(0);
  const first = await firstPromise;
  assert.ok(first.image.data.every((pixel) => pixel === 255));
  first.image.data.fill(42);
  const equivalent: DecodedImage = { ...source, data: new Uint8Array(12).fill(255) };
  const cached = await prepareImage(equivalent, {}, cache);
  assert.ok(cached.image.data.every((pixel) => pixel === 255));
  assert.equal(cached.observation.versionId, first.observation.versionId);
  assert.equal(cached.observation.preprocessing.cache, 'hit'); assert.equal(cached.observation.preprocessing.reused, 'normalized-pixels');
  cached.image.data.fill(17);
  const independent = await prepareImage(equivalent, {}, cache);
  assert.ok(independent.image.data.every((pixel) => pixel === 255));
  const changed = await prepareImage(source, {}, cache);
  assert.notEqual(changed.observation.sourceVersionId, first.observation.sourceVersionId);
  assert.equal(changed.observation.preprocessing.cache, 'miss');
  const cropped = await prepareImage(source, { region: { unit: 'pixels', x: 0, y: 0, width: 1, height: 2 } }, cache);
  assert.notEqual(cropped.observation.versionId, changed.observation.versionId);
  assert.equal(cropped.observation.preprocessing.cache, 'miss');
  assert.equal(cache.diagnostics().entries, 1); assert.equal(cache.diagnostics().bytes, 6); assert.ok(cache.diagnostics().evictions >= 2);
  cache.clear(); assert.equal(cache.diagnostics().bytes, 0); assert.equal(cache.diagnostics().entries, 0);
  const tooSmall = new ImagePreprocessCache({ maxBytes: 1 });
  await prepareImage(equivalent, {}, tooSmall); assert.equal(tooSmall.diagnostics().entries, 0);
  const disabled = await prepareImage(equivalent, {}, new ImagePreprocessCache({ maxBytes: 0 }));
  assert.equal(disabled.observation.preprocessing.cache, 'disabled');
  const controller = new AbortController(); controller.abort();
  await assert.rejects(prepareImage(equivalent, { signal: controller.signal }, cache), { name: 'AbortError' });
});

test('pixel cache hits cannot bypass remote authorization, offline policy or current byte limits', async () => {
  const bytes = await sharp({ create: { width: 2, height: 2, channels: 3, background: '#ff0000' } }).png().toBuffer();
  let requests = 0; let approvals = 0;
  const server = createServer((_request, response) => { requests++; response.writeHead(200, { 'content-type': 'image/png' }); response.end(bytes); });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const url = `http://127.0.0.1:${address.port}/image.png`;
  const cache = new ImagePreprocessCache();
  const policy = { network: () => { approvals++; return true; } };
  try {
    const first = await prepareImageInternal(url, { _offline: false, _policy: policy }, cache);
    const second = await prepareImageInternal(url, { _offline: false, _policy: policy }, cache);
    assert.equal(second.observation.preprocessing.cache, 'hit'); assert.equal(second.observation.versionId, first.observation.versionId);
    assert.equal(requests, 2); assert.equal(approvals, 2);
    await assert.rejects(prepareImageInternal(url, { _offline: true, _policy: policy }, cache), { code: 'POLICY_DENIED' });
    await assert.rejects(prepareImageInternal(url, { _offline: false, _policy: { network: () => false } }, cache), { code: 'POLICY_DENIED' });
    assert.equal(requests, 2);
    await assert.rejects(prepareImageInternal(url, { _offline: false, _policy: policy, maxImageBytes: bytes.length - 1 }, cache), /byte limit/);
    assert.equal(cache.diagnostics().hits, 1);
  } finally { await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve())); }
});

test('cached local pixels still require fresh canonical file authorization', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-image-cache-'));
  const bytes = await sharp({ create: { width: 2, height: 2, channels: 3, background: '#ff0000' } }).png().toBuffer();
  const path = join(directory, 'photo.png'); await writeFile(path, bytes);
  const cache = new ImagePreprocessCache(); let calls = 0;
  const policy = { localFiles: (canonical: string) => { assert.equal(canonical, path); calls++; return true; } };
  try {
    await prepareImageInternal(path, { _policy: policy }, cache);
    await assert.rejects(prepareImageInternal(path, { _offline: false }, cache), { code: 'POLICY_DENIED' });
    await assert.rejects(prepareImageInternal(path, { _offline: true }, cache), { code: 'POLICY_DENIED' });
    const hit = await prepareImageInternal(path, { _policy: policy }, cache);
    assert.equal(hit.observation.preprocessing.cache, 'hit'); assert.equal(calls, 2);
    await assert.rejects(prepareImageInternal(path, { _policy: { localFiles: () => false } }, cache), { code: 'POLICY_DENIED' });
  } finally { await rm(directory, { recursive: true, force: true }); }
});
