import assert from 'node:assert/strict';
import { readdir } from 'node:fs/promises';
import test from 'node:test';
import type { Tensor, Qwen3_5ForCausalLM } from '@huggingface/transformers';

interface BundledRuntime { Tensor2: typeof Tensor; Qwen3_5ForCausalLM: typeof Qwen3_5ForCausalLM; }

// These fixes are applied during bundling; importing node_modules would exercise the unfixed dependency.
async function bundledRuntime(): Promise<BundledRuntime> {
  const directory = new URL('../node/', import.meta.url);
  for (const name of await readdir(directory)) {
    if (!name.startsWith('chunk-') || !name.endsWith('.js')) continue;
    const module = await import(new URL(name, directory).href) as Record<string, unknown>;
    if (typeof module.Qwen3_5ForCausalLM === 'function' && typeof module.Tensor2 === 'function') return module as unknown as BundledRuntime;
  }
  throw new Error('Built Node runtime does not expose its shared model implementation');
}

test('bundled text RoPE continues each row after its real prompt, independent of left padding', async () => {
  const { Tensor2, Qwen3_5ForCausalLM: Model } = await bundledRuntime();
  const model = Object.create(Model.prototype) as {
    config: unknown;
    sessions: unknown;
    get_rope_index(ids: Tensor, images: null, videos: null, mask: Tensor): [Tensor, Tensor];
    prepare_inputs_for_generation(ids: bigint[][], inputs: Record<string, unknown>, config: unknown): { position_ids: Tensor };
  };
  model.config = { vision_config: {} };
  model.sessions = { decoder_model_merged: { inputNames: ['position_ids'] } };
  for (const padding of [0, 1, 8, 64]) {
    const width = 4 + padding;
    const maskData = new BigInt64Array(width * 2).fill(1n);
    maskData.fill(0n, 0, padding);
    maskData.fill(0n, width, width + padding + 2);
    const ids = new Tensor2('int64', new BigInt64Array(width * 2).fill(42n), [2, width]);
    const mask = new Tensor2('int64', maskData, [2, width]);
    const [positions, deltas] = model.get_rope_index(ids, null, null, mask);
    try {
      assert.deepEqual(deltas.tolist(), [[-BigInt(padding)], [-BigInt(padding + 2)]]);
      assert.deepEqual(positions.tolist().map((axis) => (axis as bigint[][]).map((row) => row.at(-1))), [[3n, 1n], [3n, 1n], [3n, 1n]]);
      for (const generated of [0, 1, 7]) {
        const prepared = model.prepare_inputs_for_generation([], {
          input_ids: ids, attention_mask: mask, rope_deltas: deltas,
          past_key_values: { get_seq_length: () => width + generated },
        }, {});
        try {
          assert.deepEqual(prepared.position_ids.tolist(), Array.from({ length: 3 }, () => [[BigInt(4 + generated)], [BigInt(2 + generated)]]));
        } finally { prepared.position_ids.dispose(); }
      }
    } finally { ids.dispose(); mask.dispose(); positions.dispose(); deltas.dispose(); }
  }
});
