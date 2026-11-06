import assert from 'node:assert/strict';
import test from 'node:test';
import { DynamicCache, Tensor } from '@huggingface/transformers';
import { prefillBatch, type BatchPrefillRow } from '../src/core/batch-prefill.js';
import { InferenceReuseCache, type GenerationStateSnapshot } from '../src/core/reuse.js';

const KEY = 'present.3.key';
const VALUE = 'present.3.value';
const RECURRENT = 'present_recurrent.0';
const CONV = 'present_conv.0';
const pastName = (name: string) => name.replace('present_conv', 'past_conv').replace('present_recurrent', 'past_recurrent').replace('present', 'past_key_values');
const position = (axis: number, index: number) => axis * 100 + index * 3 + 1;
const positionCode = (index: number) => position(0, index) + 2 * position(1, index) + 3 * position(2, index);

type Outputs = Record<string, unknown>;

interface TensorOwnership {
  disposed: Set<Tensor>;
  track(tensor: Tensor): Tensor;
  assertDisposed(): void;
}

function ownership(): TensorOwnership {
  const observed = new Set<Tensor>();
  const disposed = new Set<Tensor>();
  return {
    disposed,
    track(tensor: Tensor) {
      if (!observed.has(tensor)) {
        observed.add(tensor);
        const dispose = tensor.dispose.bind(tensor);
        tensor.dispose = () => { disposed.add(tensor); return dispose(); };
      }
      return tensor;
    },
    assertDisposed() { for (const tensor of observed) assert.ok(disposed.has(tensor), 'temporary row input, output or cache tensor leaked'); },
  };
}

function batch(tokens: readonly (readonly bigint[])[]) {
  const width = Math.max(...tokens.map((row) => row.length));
  const ids: bigint[] = []; const mask: bigint[] = []; const positions: bigint[] = [];
  for (const row of tokens) {
    ids.push(...Array<bigint>(width - row.length).fill(0n), ...row);
    mask.push(...Array<bigint>(width - row.length).fill(0n), ...Array<bigint>(row.length).fill(1n));
  }
  for (let axis = 0; axis < 3; axis++) for (const row of tokens) {
    positions.push(...Array<bigint>(width - row.length).fill(-99n), ...row.map((_, index) => BigInt(position(axis, index))));
  }
  const inputs: Outputs = {
    input_ids: new Tensor('int64', BigInt64Array.from(ids), [tokens.length, width]),
    attention_mask: new Tensor('int64', BigInt64Array.from(mask), [tokens.length, width]),
    position_ids: new Tensor('int64', BigInt64Array.from(positions), [3, tokens.length, width]),
    past_key_values: new DynamicCache(),
  };
  return { inputs, width };
}

function expected(tokens: readonly bigint[]) {
  let sum = 0; let weighted = 0; let rolling = 0;
  for (const [index, token] of tokens.entries()) {
    sum += Number(token); weighted += Number(token) * positionCode(index); rolling = rolling * 2 + Number(token);
  }
  return { recurrent: [weighted, sum], conv: [rolling, tokens.length], logits: [weighted + rolling, sum + tokens.length, rolling - sum] };
}

function synthetic(options: { gpu?: boolean; failAt?: number; afterCall?: (call: number) => void } = {}) {
  const owned = ownership();
  const calls: { ids: bigint[]; cached: number }[] = [];
  let downloads = 0;
  const forward = async (inputs: Outputs): Promise<Outputs> => {
    const ids = inputs.input_ids as Tensor; const positions = inputs.position_ids as Tensor; const mask = inputs.attention_mask as Tensor;
    const cache = inputs.past_key_values;
    assert.ok(cache instanceof DynamicCache);
    for (const tensor of [ids, positions, mask, ...Object.values(cache)]) owned.track(tensor);
    const cached = cache.get_seq_length();
    const suffix = Array.from(ids.data, BigInt);
    const length = cached + suffix.length;
    assert.deepEqual(ids.dims, [1, suffix.length]);
    assert.deepEqual(positions.dims, [3, 1, suffix.length]);
    assert.deepEqual(mask.dims, [1, length]);
    assert.deepEqual(Array.from(mask.data), Array<bigint>(length).fill(1n));
    for (let axis = 0; axis < 3; axis++) for (let index = 0; index < suffix.length; index++) {
      assert.equal(positions.data[axis * suffix.length + index], BigInt(position(axis, cached + index)), 'original RoPE positions must survive prefix splitting and unpadding');
    }
    calls.push({ ids: suffix, cached });
    if (calls.length === options.failAt) throw new Error('native forward failed');
    const recurrent = cache['past_recurrent.0'] ? Array.from(cache['past_recurrent.0'].data, Number) : [0, 0];
    const conv = cache['past_conv.0'] ? Array.from(cache['past_conv.0'].data, Number) : [0, 0];
    for (const [index, token] of suffix.entries()) {
      const numeric = Number(token);
      const code = Number(positions.data[index]) + 2 * Number(positions.data[suffix.length + index]) + 3 * Number(positions.data[2 * suffix.length + index]);
      recurrent[0]! += numeric * code; recurrent[1]! += numeric;
      conv[0] = conv[0]! * 2 + numeric; conv[1]!++;
    }
    // A native decoder may mutate its working state: neither staged nor committed snapshots may alias it.
    if (cache['past_recurrent.0']) (cache['past_recurrent.0'].data as Float32Array).set(recurrent);
    if (cache['past_conv.0']) (cache['past_conv.0'].data as Float32Array).set(conv);
    const outputs: Outputs = {};
    for (const [name, value] of [[KEY, false], [VALUE, true]] as const) {
      const data = new Float32Array(2 * length * 2);
      const old = cache[pastName(name)];
      for (let head = 0; head < 2; head++) {
        if (old) data.set((old.data as Float32Array).subarray(head * cached * 2, (head + 1) * cached * 2), head * length * 2);
        for (const [index, token] of suffix.entries()) {
          const offset = (head * length + cached + index) * 2;
          data[offset] = Number(token) * (value ? 2 : 1) + head * 1000;
          data[offset + 1] = positionCode(cached + index) + (value ? 50 : 0);
        }
      }
      outputs[name] = new Tensor('float32', data, [1, 2, length, 2]);
    }
    outputs[RECURRENT] = new Tensor('float32', Float32Array.from(recurrent), [1, 2]);
    outputs[CONV] = new Tensor('float32', Float32Array.from(conv), [1, 2]);
    outputs.logits = new Tensor('float32', Float32Array.from([recurrent[0]! + conv[0]!, recurrent[1]! + conv[1]!, conv[0]! - recurrent[1]!]), [1, 1, 3]);
    for (const tensor of Object.values(outputs) as Tensor[]) {
      owned.track(tensor);
      if (options.gpu) {
        const data = tensor.data;
        const getData = tensor.ort_tensor.getData.bind(tensor.ort_tensor);
        let downloaded = false;
        Object.defineProperties(tensor, {
          location: { get: () => downloaded ? 'cpu' : 'gpu-buffer' },
          data: { get: () => { assert.ok(downloaded, 'GPU storage read before download'); return data; } },
        });
        tensor.ort_tensor.getData = async (release?: boolean) => {
          assert.equal(release, true, 'download must release GPU storage'); downloaded = true; downloads++; return getData(release);
        };
      }
    }
    options.afterCall?.(calls.length);
    return outputs;
  };
  return { forward, calls, owned, get downloads() { return downloads; } };
}

function assertMerged(outputs: Outputs, tokens: readonly (readonly bigint[])[], width: number) {
  assert.deepEqual(Object.keys(outputs).sort(), [KEY, VALUE, RECURRENT, CONV, 'logits'].sort());
  for (const [name, value] of [[KEY, false], [VALUE, true]] as const) {
    const tensor = outputs[name] as Tensor;
    assert.deepEqual(tensor.dims, [tokens.length, 2, width, 2]);
    const values: number[] = [];
    for (const row of tokens) for (let head = 0; head < 2; head++) {
      values.push(...Array<number>((width - row.length) * 2).fill(0));
      for (const [index, token] of row.entries()) values.push(Number(token) * (value ? 2 : 1) + head * 1000, positionCode(index) + (value ? 50 : 0));
    }
    assert.deepEqual(Array.from(tensor.data), values, `${name}: left pad the sequence dimension independently for each head`);
  }
  for (const [name, field, dims] of [[RECURRENT, 'recurrent', [tokens.length, 2]], [CONV, 'conv', [tokens.length, 2]], ['logits', 'logits', [tokens.length, 1, 3]]] as const) {
    const tensor = outputs[name] as Tensor;
    assert.deepEqual(tensor.dims, dims);
    assert.deepEqual(Array.from(tensor.data), tokens.flatMap((row) => expected(row)[field]), `${name}: preserve complete independent row state`);
  }
}

function disposeOutputs(outputs: Outputs) { for (const value of Object.values(outputs)) if (value instanceof Tensor) value.dispose(); }
function discard(staged: Map<string, GenerationStateSnapshot>, reuse: InferenceReuseCache) { for (const snapshot of staged.values()) reuse.discardState(snapshot); staged.clear(); }
function assertInputsRetained(inputs: Outputs, retained: TensorOwnership) {
  for (const tensor of Object.values(inputs)) if (tensor instanceof Tensor) assert.ok(!retained.disposed.has(tensor), 'caller batch input was disposed');
}
function watchInputs(inputs: Outputs) {
  const retained = ownership();
  for (const tensor of Object.values(inputs)) if (tensor instanceof Tensor) retained.track(tensor);
  return retained;
}

async function prefixSnapshot(reuse: InferenceReuseCache, tokens: bigint[]) {
  const model = synthetic(); const { inputs } = batch([tokens]);
  const output = await model.forward(inputs);
  const entries: Record<string, Tensor> = {};
  for (const name of [KEY, VALUE, RECURRENT, CONV]) entries[pastName(name)] = output[name] as Tensor;
  const snapshot = await reuse.snapshot(new DynamicCache(entries), tokens, '');
  disposeOutputs(output); disposeOutputs(inputs);
  return snapshot;
}

test('unpadded no-prefix rows merge full attention head strides and independent hybrid states', async () => {
  const tokens = [[5n, 9n], [2n, 7n, 11n, 13n]];
  const { inputs, width } = batch(tokens); const retained = watchInputs(inputs);
  const reuse = new InferenceReuseCache(); const staged = new Map<string, GenerationStateSnapshot>(); const model = synthetic();
  const output = await prefillBatch(inputs, tokens.map((row) => ({ inputTokens: row.length })), model.forward, reuse, staged, () => {});
  assertMerged(output, tokens, width);
  assert.deepEqual(model.calls, [{ ids: tokens[0], cached: 0 }, { ids: tokens[1], cached: 0 }]);
  assert.equal(staged.size, 0); model.owned.assertDisposed(); assertInputsRetained(inputs, retained);
  disposeOutputs(output); disposeOutputs(inputs); reuse.clear();
});

test('a missing common prefix is processed once, staged, and reused without losing recurrent history', async () => {
  const prefix = [2n, 3n]; const tokens = [[...prefix, 5n], [...prefix, 7n, 11n, 13n]];
  const { inputs, width } = batch(tokens); const retained = watchInputs(inputs);
  const reuse = new InferenceReuseCache(); const staged = new Map<string, GenerationStateSnapshot>(); const model = synthetic();
  const rows: BatchPrefillRow[] = tokens.map((row) => ({ inputTokens: row.length, prefixTokens: prefix }));
  const output = await prefillBatch(inputs, rows, model.forward, reuse, staged, () => {});
  assertMerged(output, tokens, width);
  assert.deepEqual(model.calls, [{ ids: prefix, cached: 0 }, { ids: [5n], cached: 2 }, { ids: [7n, 11n, 13n], cached: 2 }]);
  assert.equal(staged.size, 1); assert.equal(reuse.info().prefixEntries, 0, 'helper must not publish caller-owned staged state');
  const snapshot = staged.get(prefix.join(','))!;
  assert.deepEqual(Array.from(snapshot.cache['past_recurrent.0']!.data), expected(prefix).recurrent);
  assert.deepEqual(Array.from(snapshot.cache['past_conv.0']!.data), expected(prefix).conv);
  const branch = await reuse.snapshot(snapshot.cache, prefix, ''); reuse.discardState(branch);
  model.owned.assertDisposed(); assertInputsRetained(inputs, retained);
  disposeOutputs(output); disposeOutputs(inputs); discard(staged, reuse); reuse.clear();
});

test('committed prefix checkout isolates in-place continuation mutations across rows and future batches', async () => {
  const reuse = new InferenceReuseCache(); const prefix = [2n, 3n];
  const snapshot = await prefixSnapshot(reuse, prefix); const snapshotOwnership = ownership();
  for (const tensor of Object.values(snapshot.cache)) snapshotOwnership.track(tensor);
  reuse.commitPrefix(prefix.join(','), snapshot);
  const tokens = [[...prefix, 17n], [...prefix, 19n, 23n]];
  const { inputs, width } = batch(tokens); const retained = watchInputs(inputs);
  const model = synthetic(); const staged = new Map<string, GenerationStateSnapshot>();
  const output = await prefillBatch(inputs, tokens.map((row) => ({ inputTokens: row.length, prefixTokens: prefix })), model.forward, reuse, staged, () => {});
  assertMerged(output, tokens, width);
  assert.deepEqual(model.calls, [{ ids: [17n], cached: 2 }, { ids: [19n, 23n], cached: 2 }]);
  assert.equal(staged.size, 0);
  assert.deepEqual(Array.from(snapshot.cache['past_recurrent.0']!.data), expected(prefix).recurrent);
  assert.deepEqual(Array.from(snapshot.cache['past_conv.0']!.data), expected(prefix).conv);
  for (const tensor of Object.values(snapshot.cache)) assert.ok(!snapshotOwnership.disposed.has(tensor));
  const again = reuse.checkoutPrefix(prefix.join(','), [...prefix, 29n])!;
  assert.deepEqual(Array.from(again.cache['past_recurrent.0']!.data), expected(prefix).recurrent);
  for (const tensor of Object.values(again.cache)) tensor.dispose();
  model.owned.assertDisposed(); assertInputsRetained(inputs, retained);
  disposeOutputs(output); disposeOutputs(inputs); reuse.clear();
});

test('GPU-like row outputs download before reads and become independently owned merged tensors', async () => {
  const tokens = [[5n], [7n, 11n, 13n]]; const { inputs, width } = batch(tokens);
  const reuse = new InferenceReuseCache(); const staged = new Map<string, GenerationStateSnapshot>(); const model = synthetic({ gpu: true });
  const output = await prefillBatch(inputs, tokens.map((row) => ({ inputTokens: row.length })), model.forward, reuse, staged, () => {});
  assertMerged(output, tokens, width);
  assert.equal(model.downloads, 10); model.owned.assertDisposed();
  disposeOutputs(output); disposeOutputs(inputs); reuse.clear();
});

for (const failure of ['abort', 'forward'] as const) test(`${failure} between prefix and suffix disposes temporary state but retains caller-owned staged snapshots`, async () => {
  const prefix = [2n, 3n]; const tokens = [[...prefix, 5n], [...prefix, 7n, 11n]];
  const { inputs } = batch(tokens); const retained = watchInputs(inputs);
  const reuse = new InferenceReuseCache(); const staged = new Map<string, GenerationStateSnapshot>();
  let abort = false;
  const model = synthetic({ ...(failure === 'forward' ? { failAt: 2 } : {}), afterCall: () => { abort = failure === 'abort'; } });
  const error = new Error('cancelled');
  await assert.rejects(prefillBatch(inputs, tokens.map((row) => ({ inputTokens: row.length, prefixTokens: prefix })), model.forward, reuse, staged, () => { if (abort && staged.size > 0) throw error; }), failure === 'abort' ? /cancelled/ : /native forward failed/);
  assert.equal(staged.size, 1, 'snapshot captured before failure remains owned by caller');
  const snapshot = staged.get(prefix.join(','))!;
  assert.deepEqual(Array.from(snapshot.cache['past_recurrent.0']!.data), expected(prefix).recurrent);
  assert.deepEqual(Array.from(snapshot.cache['past_conv.0']!.data), expected(prefix).conv);
  const retainedSnapshot = await reuse.snapshot(snapshot.cache, prefix, ''); reuse.discardState(retainedSnapshot);
  model.owned.assertDisposed(); assertInputsRetained(inputs, retained);
  assert.equal(reuse.info().prefixEntries, 0);
  disposeOutputs(inputs); discard(staged, reuse); reuse.clear();
});
