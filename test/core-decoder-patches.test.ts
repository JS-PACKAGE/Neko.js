import assert from 'node:assert/strict';
import test from 'node:test';
import type { PreTrainedModel } from '@huggingface/transformers';
import { releaseConsumedDecoderOutputs } from '../src/core/decoder-patches.js';

type Value = { location: string; data: Float32Array<ArrayBuffer> };
const cpu = (length = 4): Value => ({ location: 'cpu', data: new Float32Array(length).fill(1) });

test('decoder output release detaches only state its own runs returned, after it is consumed', async () => {
  const returned: Record<string, Value>[] = [];
  const session = { inputNames: ['past_recurrent.0'], run: async () => { const outputs = { logits: cpu(), 'present_recurrent.0': cpu(), 'present.3.key': cpu() }; returned.push(outputs); return outputs; } };
  releaseConsumedDecoderOutputs({ sessions: { decoder_model_merged: session } } as unknown as PreTrainedModel);
  const run = (feeds: Record<string, unknown>) => (session.run as (feeds: unknown) => Promise<Record<string, Value>>)(feeds);

  // A cache checkout or snapshot clone is never detached, nor is anything the caller still holds before it is fed back.
  const clone = cpu();
  const first = await run({ 'past_recurrent.0': clone });
  assert.equal(clone.data.buffer.detached, false);
  assert.ok(Object.values(first).every((value) => !value.data.buffer.detached));

  // Feeding the returned state back releases it (and the previous logits) once the run completes.
  const shared = new Float32Array(new ArrayBuffer(32));
  const view: Value = { location: 'cpu', data: shared.subarray(0, 4) };
  const second = await run({ 'past_recurrent.0': first['present_recurrent.0'], 'past_key_values.3.key': first['present.3.key'], 'past_conv.0': view, attention_mask: cpu() });
  assert.equal(first['present_recurrent.0']!.data.buffer.detached, true);
  assert.equal(first['present.3.key']!.data.buffer.detached, true);
  assert.equal(first.logits!.data.buffer.detached, true);
  assert.equal(shared.buffer.detached, false);
  assert.ok(Object.values(second).every((value) => !value.data.buffer.detached));

  // The final state is never fed again (generation snapshots it for reuse), so it stays intact.
  assert.equal(returned.length, 2);
  assert.ok(second['present_recurrent.0']!.data.every((value) => value === 1));
});

test('decoder output release leaves GPU-resident and failed-run state alone', async () => {
  let fail = false;
  const gpu = { location: 'gpu-buffer', data: new Float32Array(4) };
  const session = { inputNames: [], run: async () => { if (fail) throw new Error('boom'); return { logits: gpu, 'present_recurrent.0': cpu() }; } };
  releaseConsumedDecoderOutputs({ sessions: { decoder_model_merged: session } } as unknown as PreTrainedModel);
  const run = (feeds: Record<string, unknown>) => (session.run as (feeds: unknown) => Promise<Record<string, Value>>)(feeds);
  const first = await run({});
  fail = true;
  await assert.rejects(run({ 'past_recurrent.0': first['present_recurrent.0'] }), /boom/);
  assert.equal(first['present_recurrent.0']!.data.buffer.detached, false);
  fail = false;
  await run({ 'past_recurrent.0': first['present_recurrent.0'] });
  assert.equal(first['present_recurrent.0']!.data.buffer.detached, true);
  assert.equal(gpu.data.buffer.detached, false);
});
