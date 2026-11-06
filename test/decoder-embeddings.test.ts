import assert from 'node:assert/strict';
import test from 'node:test';
import ort from 'onnxruntime-node';
import { Tensor } from '@huggingface/transformers';
import { keepDecodeEmbeddingsOnGpu } from '../src/core/decoder-patches.js';

type EmbeddingModel = Parameters<typeof keepDecodeEmbeddingsOnGpu>[0];

// Real ORT tensor ownership, with an opaque GPU handle and controlled download/disposal.
function embedding(dims: number[], failure?: Error) {
  let downloads = 0; let releases = 0;
  const values = Float32Array.from({ length: dims.reduce((a, b) => a * b, 1) }, (_, index) => index + 1);
  const tensor = new Tensor(ort.Tensor.fromGpuBuffer({} as Parameters<typeof ort.Tensor.fromGpuBuffer>[0], {
    dataType: 'float32', dims,
    download: async () => { downloads++; if (failure) throw failure; return values; },
    dispose: () => { releases++; },
  }));
  return { tensor, counts: () => ({ downloads, releases }) };
}
const ids = (batch: number, length: number) => new Tensor('int64', new BigInt64Array(batch * length).fill(1n), [batch, length]);

test('batched single-token embeddings remain GPU-resident until the decoder finishes', async () => {
  const source = embedding([2, 1, 4]); const entered = Promise.withResolvers<void>(); const finish = Promise.withResolvers<void>();
  const model: EmbeddingModel = {
    encode_text: async () => source.tensor,
    async forward(inputs: { input_ids: Tensor }) { await this.encode_text(inputs); entered.resolve(); await finish.promise; },
  };
  keepDecodeEmbeddingsOnGpu(model);
  const pending = model.forward({ input_ids: ids(2, 1) });
  await entered.promise;
  assert.equal(source.tensor.location, 'gpu-buffer');
  assert.deepEqual(source.counts(), { downloads: 0, releases: 0 });
  finish.resolve(); await pending;
  assert.equal(source.tensor.location, 'none');
  assert.deepEqual(source.counts(), { downloads: 0, releases: 1 });
});

test('prefill embeddings remain CPU-readable and writable for image-feature merging', async () => {
  const source = embedding([1, 3, 4]);
  const model: EmbeddingModel = {
    encode_text: async () => source.tensor,
    async forward(inputs: { input_ids: Tensor }) {
      const tensor = await this.encode_text(inputs);
      (tensor.data as Float32Array).set([90, 91, 92, 93], 4);
      return Array.from(tensor.data);
    },
  };
  keepDecodeEmbeddingsOnGpu(model);
  assert.deepEqual(await model.forward({ input_ids: ids(1, 3) }), [1, 2, 3, 4, 90, 91, 92, 93, 9, 10, 11, 12]);
  assert.deepEqual(source.counts(), { downloads: 1, releases: 1 });
  assert.equal(source.tensor.location, 'cpu');
  source.tensor.dispose();
});

test('decoder failure releases owned embeddings without consuming a later caller-owned GPU tensor', async () => {
  const source = embedding([1, 1, 4]); const borrowed = embedding([1, 1, 4]); const failure = new Error('decoder failed');
  const model: EmbeddingModel = {
    encode_text: async () => source.tensor,
    async forward(inputs: { input_ids: Tensor; inputs_embeds?: Tensor }) {
      if (!inputs.inputs_embeds) { await this.encode_text(inputs); throw failure; }
      assert.equal(inputs.inputs_embeds.location, 'gpu-buffer');
    },
  };
  keepDecodeEmbeddingsOnGpu(model);
  await assert.rejects(model.forward({ input_ids: ids(1, 1) }), (error) => error === failure);
  assert.equal(source.tensor.location, 'none');
  assert.deepEqual(source.counts(), { downloads: 0, releases: 1 });
  await model.forward({ input_ids: ids(1, 1), inputs_embeds: borrowed.tensor });
  assert.equal(borrowed.tensor.location, 'gpu-buffer');
  assert.deepEqual(borrowed.counts(), { downloads: 0, releases: 0 });
  borrowed.tensor.dispose();
});

test('failed prefill download releases the GPU tensor and preserves the original error', async () => {
  const failure = new Error('download failed'); const source = embedding([1, 2, 4], failure);
  const model: EmbeddingModel = {
    encode_text: async () => source.tensor,
    async forward(inputs: { input_ids: Tensor }) { await this.encode_text(inputs); },
  };
  keepDecodeEmbeddingsOnGpu(model);
  await assert.rejects(model.forward({ input_ids: ids(1, 2) }), (error) => error === failure);
  assert.equal(source.tensor.location, 'none');
  assert.deepEqual(source.counts(), { downloads: 1, releases: 1 });
});
