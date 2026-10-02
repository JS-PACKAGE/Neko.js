import assert from 'node:assert/strict';
import { mkdtemp, realpath, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';
import sharp from 'sharp';
import { readImage, loadImage } from '../../src/web/index.js';

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
