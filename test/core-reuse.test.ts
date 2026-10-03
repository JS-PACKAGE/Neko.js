import assert from 'node:assert/strict';
import test from 'node:test';
import { DynamicCache, Tensor } from '@huggingface/transformers';
import { InferenceReuseCache } from '../src/core/reuse.js';

function hybrid(length: number) {
  return new DynamicCache({
    'past_key_values.3.key': new Tensor('float32', new Float32Array(length * 2), [1, 1, length, 2]),
    'past_key_values.3.value': new Tensor('float32', new Float32Array(length * 2), [1, 1, length, 2]),
    'past_conv.0': new Tensor('float32', new Float32Array([1, 2]), [1, 2]),
    'past_recurrent.0': new Tensor('float32', new Float32Array([3, 4]), [1, 2]),
  });
}

test('hybrid state snapshots retain every tensor and isolate reusable branches', async () => {
  const cache = new InferenceReuseCache({ stateEntries: 1 });
  const original = hybrid(2);
  const saved = await cache.snapshot(original, [1n, 2n], 'model/schema/image');
  const handle = cache.commitState(saved);
  (original['past_recurrent.0']!.data as Float32Array)[0] = 99;
  const a = cache.checkout(handle, [1n, 2n, 3n], 'model/schema/image');
  const b = cache.checkout(handle, [1n, 2n, 4n], 'model/schema/image');
  assert.equal(a.tokens, 2); assert.equal(a.cache['past_recurrent.0']!.data[0], 3);
  (a.cache['past_recurrent.0']!.data as Float32Array)[0] = 88;
  assert.equal(b.cache['past_recurrent.0']!.data[0], 3);
  assert.throws(() => cache.checkout(handle, [1n, 9n, 3n], 'model/schema/image'), /exact extending/);
  assert.throws(() => cache.checkout(handle, [1n, 2n, 3n], 'other schema'), /same model/);
  assert.throws(() => new InferenceReuseCache().checkout(handle, [1n, 2n, 3n], 'model/schema/image'), /not owned/);
  const next = cache.commitState(await cache.snapshot(hybrid(1), [7n], 'model/schema/image'));
  assert.throws(() => cache.checkout(handle, [1n, 2n, 3n], 'model/schema/image'), /evicted/);
  assert.equal(cache.info().evictions, 1);
  cache.release(next); assert.equal(cache.info().stateEntries, 0);
  cache.clear();
  for (const tensor of [...Object.values(original), ...Object.values(a.cache), ...Object.values(b.cache)]) tensor.dispose();
});

test('retained handles choose the longest genuine compatible checkpoint without truncating hybrid state', async () => {
  const cache = new InferenceReuseCache();
  const prefix = hybrid(2); const final = hybrid(4);
  (final['past_recurrent.0']!.data as Float32Array)[0] = 40;
  (final['past_conv.0']!.data as Float32Array)[0] = 50;
  const checkpoint = await cache.snapshot(prefix, [1n, 2n], 'conversation');
  const completed = await cache.snapshot(final, [1n, 2n, 99n, 4n], 'conversation');
  const handle = cache.commitState(completed, checkpoint);
  assert.equal(cache.info().stateEntries, 1);
  assert.equal(cache.info().stateBytes, checkpoint.bytes + completed.bytes);
  // Public assistant messages omit the generation-only thinking prefill (token 99).
  const publicTurn = cache.checkout(handle, [1n, 2n, 3n, 5n], 'conversation');
  const generatedTurn = cache.checkout(handle, [1n, 2n, 99n, 4n, 5n], 'conversation');
  assert.equal(publicTurn.tokens, 2); assert.equal(publicTurn.cache.get_seq_length(), 2);
  assert.equal(publicTurn.cache['past_recurrent.0']!.data[0], 3);
  assert.equal(generatedTurn.tokens, 4); assert.equal(generatedTurn.cache.get_seq_length(), 4);
  assert.equal(generatedTurn.cache['past_recurrent.0']!.data[0], 40);
  assert.equal(generatedTurn.cache['past_conv.0']!.data[0], 50);
  for (const tensor of Object.values(publicTurn.cache)) (tensor.data as Float32Array)[0] = 80;
  const branch = cache.checkout(handle, [1n, 2n, 6n], 'conversation');
  assert.equal(branch.cache['past_recurrent.0']!.data[0], 3);
  assert.equal(branch.cache['past_conv.0']!.data[0], 1);
  assert.equal(branch.cache['past_key_values.3.key']!.data[0], 0);
  assert.equal(branch.cache['past_key_values.3.value']!.data[0], 0);
  assert.throws(() => cache.checkout(handle, [1n, 8n, 6n], 'conversation'), /exact extending/);
  assert.throws(() => cache.checkout(handle, [1n, 2n, 6n], 'different-images'), /same model/);
  cache.release(handle);
  assert.equal(cache.info().stateBytes, 0);
  assert.throws(() => cache.checkout(handle, [1n, 2n, 6n], 'conversation'), /released/);
  for (const tensor of [...Object.values(prefix), ...Object.values(final), ...Object.values(publicTurn.cache), ...Object.values(generatedTurn.cache), ...Object.values(branch.cache)]) tensor.dispose();
});

test('the byte limit accounts for both checkpoints atomically before publishing or evicting handles', async () => {
  const cache = new InferenceReuseCache({ stateBytes: 100, stateEntries: 2 });
  const original = hybrid(1); const prefix = hybrid(2); const final = hybrid(4);
  const handle = cache.commitState(await cache.snapshot(original, [7n], 'conversation'));
  const checkpoint = await cache.snapshot(prefix, [1n, 2n], 'conversation');
  const completed = await cache.snapshot(final, [1n, 2n, 99n, 4n], 'conversation');
  const before = cache.info();
  assert.throws(() => cache.commitState(completed, checkpoint), /byte limit/);
  assert.deepEqual(cache.info(), before);
  const branch = cache.checkout(handle, [7n, 8n], 'conversation');
  assert.equal(branch.tokens, 1);
  cache.discardState(checkpoint); cache.discardState(completed); cache.clear();
  assert.equal(cache.info().stateBytes, 0);
  for (const tensor of [...Object.values(original), ...Object.values(prefix), ...Object.values(final), ...Object.values(branch.cache)]) tensor.dispose();
});

test('vision feature entries own encoder results, publish transactionally and evict within byte/entry limits', async () => {
  const cache = new InferenceReuseCache({ visionEntries: 1, visionBytes: 16 });
  const features = new Tensor('float32', new Float32Array([1, 2]), [1, 2]);
  const owned = await cache.snapshotFeature(features);
  (features.data as Float32Array)[0] = 90;
  assert.equal(cache.feature('image'), undefined);
  const staged = new Map([['image', owned]]);
  cache.commitFeatures(staged); assert.equal(staged.size, 0);
  assert.equal(cache.feature('image')?.data[0], 1);
  cache.commitFeatures(new Map([['other-image', await cache.snapshotFeature(features)]]));
  assert.equal(cache.feature('image'), undefined); assert.equal(cache.info().visionEntries, 1);
  await assert.rejects(cache.snapshotFeature(new Tensor('float32', new Float32Array(5), [1, 5])), /byte limit/);
  cache.clear(); features.dispose(); assert.equal(cache.info().visionBytes, 0);
});
