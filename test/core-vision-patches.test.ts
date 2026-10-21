import assert from 'node:assert/strict';
import test from 'node:test';
import { Qwen2VLImageProcessor, RawImage, Tensor } from '@huggingface/transformers';
import { permuteData, useFastPatchFlattening } from '../src/core/vision-patches.js';

// Same values as the pinned Qwen3.5 preprocessor_config.json.
const config = { size: { longest_edge: 16777216, shortest_edge: 65536 }, patch_size: 16, temporal_patch_size: 2, merge_size: 2, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5] };
const bytes = (tensor: Tensor) => Buffer.from((tensor.data as Float32Array).buffer);

test('strided permutation equals Tensor.permute for random shapes and axis orders', () => {
  let seed = 3; const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  for (let trial = 0; trial < 300; trial++) {
    const dims = Array.from({ length: 1 + Math.floor(random() * 6) }, () => 1 + Math.floor(random() * 4));
    const axes = [...dims.keys()].sort(() => random() - 0.5);
    const tensor = new Tensor('float32', Float32Array.from({ length: dims.reduce((left, right) => left * right, 1) }, random), dims);
    assert.ok(Buffer.from(permuteData(tensor.data as Float32Array, dims, axes).buffer).equals(bytes(tensor.permute(...axes))), `${dims} / ${axes}`);
  }
});

test('fast patch flattening yields bit-identical processor output and leaves other instances upstream', async () => {
  const upstream = new Qwen2VLImageProcessor(config); const fast = new Qwen2VLImageProcessor(config);
  useFastPatchFlattening(fast);
  assert.equal(Object.hasOwn(upstream, '_call'), false);
  let seed = 5; const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const image = (width: number, height: number, channels: 1 | 3 | 4) => new RawImage(Uint8ClampedArray.from({ length: width * height * channels }, () => Math.floor(random() * 256)), width, height, channels);
  for (const images of [[image(1, 1, 3)], [image(17, 33, 1)], [image(333, 129, 4)], [image(300, 200, 3), image(64, 96, 3)]]) {
    const expected = await upstream(images); const actual = await fast(images);
    assert.deepEqual(Object.keys(actual), Object.keys(expected));
    assert.deepEqual(actual.pixel_values.dims, expected.pixel_values.dims);
    assert.ok(bytes(actual.pixel_values).equals(bytes(expected.pixel_values)));
    assert.deepEqual(actual.image_grid_thw.data, expected.image_grid_thw.data);
    assert.deepEqual([actual.original_sizes, actual.reshaped_input_sizes], [expected.original_sizes, expected.reshaped_input_sizes]);
  }
});
