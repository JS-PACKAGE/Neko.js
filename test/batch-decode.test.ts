import assert from 'node:assert/strict';
import test from 'node:test';
import { DynamicCache, Tensor } from '@huggingface/transformers';
import { BatchDecoder } from '../src/core/batch-decode.js';
import { PreviousLogitsRelease } from '../src/core/decoder-patches.js';

type Values = Record<string, unknown>;
const names = ['present.3.key', 'present.3.value', 'present_recurrent.0', 'present_conv.0'] as const;
const pastName = (name: string) => name.replace('present_conv', 'past_conv').replace('present_recurrent', 'past_recurrent').replace('present', 'past_key_values');

function ownership() {
  const observed = new Set<Tensor>(); const disposed = new Set<Tensor>();
  return {
    observed, disposed,
    track(tensor: Tensor) {
      if (!observed.has(tensor)) {
        observed.add(tensor);
        const dispose = tensor.dispose.bind(tensor);
        tensor.dispose = () => { disposed.add(tensor); return dispose(); };
      }
      return tensor;
    },
    assertRetained() { for (const tensor of observed) assert.ok(!disposed.has(tensor), 'caller-owned tensor was disposed'); },
    assertDisposed() { for (const tensor of observed) assert.ok(disposed.has(tensor), 'temporary gathered tensor or replaced logits leaked'); },
  };
}

type State = { key: number[]; value: number[]; recurrent: number[]; conv: number[] };
function initial(row: number): State {
  return {
    key: Array.from({ length: 8 }, (_, index) => row * 20 + index + 1),
    value: Array.from({ length: 8 }, (_, index) => row * 30 + index + 2),
    recurrent: [row * 3 + 1, row * 4 + 2], conv: [row + 3, row + 5, row + 7],
  };
}
function advance(old: State, token: number, positions: number[], mask: number[]): State {
  const positionCode = positions.reduce((sum, value, axis) => sum + (axis + 1) * value, 0);
  const maskCode = mask.reduce((sum, value, index) => sum + (index + 1) * value, 0);
  const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
  const recurrent = [old.recurrent[0]! + token + sum(old.key) + positionCode, old.recurrent[1]! + sum(old.value) + maskCode];
  const conv = [old.conv[1]!, old.conv[2]!, old.conv[0]! + token + old.recurrent[0]!];
  const append = (values: number[], offset: number) => {
    const stride = values.length / 2; const result: number[] = [];
    for (let head = 0; head < 2; head++) result.push(...values.slice(head * stride, (head + 1) * stride), token + head + offset, positionCode + maskCode + offset);
    return result;
  };
  return { key: append(old.key, 0), value: append(old.value, 7), recurrent, conv };
}
function logits(state: State) {
  return [state.recurrent[0]! + state.conv[0]!, state.recurrent[1]! + state.conv[1]!, state.conv[2]!, state.recurrent[0]! - state.recurrent[1]!];
}
function numbers(tensor: Tensor) { return Array.from(tensor.data, Number); }
function tensorRows(tensor: Tensor, row: number) {
  const stride = tensor.dims.slice(1).reduce((size, dim) => size * dim, 1);
  return numbers(tensor).slice(row * stride, (row + 1) * stride);
}

function scenario(size: number) {
  const done = new Set<number>(); const decoder = new BatchDecoder(size, (row) => done.has(row));
  const caller = ownership(); const temporary = ownership(); const raw = ownership(); const presents = ownership();
  const allOutputs = new Set<Tensor>(); const calls: number[][] = [];
  const states = Array.from({ length: size }, (_, row) => initial(row));
  const seed: Record<string, Tensor> = {};
  for (const [name, field, dims] of [
    [names[0], 'key', [size, 2, 2, 2]], [names[1], 'value', [size, 2, 2, 2]],
    [names[2], 'recurrent', [size, 2]], [names[3], 'conv', [size, 3]],
  ] as const) seed[pastName(name)] = caller.track(new Tensor('float32', Float32Array.from(states.flatMap((state) => state[field])), [...dims]));
  const cache = new DynamicCache(seed);
  let step = 0; let fail = false; let lastNative: Values | undefined;
  const release = new PreviousLogitsRelease();
  const maskFor = (row: number, current: number) => [(row + 1) % 2, 1, ...Array<number>(current + 1).fill(1)];
  const positionsFor = (row: number, current: number) => [row * 11 + current + 3, row * 7 + current + 103, row * 5 + current + 203];
  const native = async (inputs: Values): Promise<Values> => {
    const ids = inputs.input_ids as Tensor; const mask = inputs.attention_mask as Tensor; const positions = inputs.position_ids as Tensor;
    const physical = inputs.past_key_values;
    assert.ok(physical instanceof DynamicCache);
    const count = ids.dims[0]!;
    assert.deepEqual(ids.dims, [count, 1]); assert.deepEqual(mask.dims, [count, step + 3]);
    assert.deepEqual(positions.dims, [3, count, 1]); assert.equal(physical.get_seq_length(), step + 2);
    for (const tensor of [ids, mask, positions, ...Object.values(physical)]) {
      if (!caller.observed.has(tensor) && !allOutputs.has(tensor)) temporary.track(tensor);
    }
    const logicalRows = numbers(ids).map((token) => token - 10 - step * 5);
    calls.push(logicalRows);
    const next: State[] = [];
    for (let row = 0; row < count; row++) {
      const logical = logicalRows[row]!;
      assert.ok(!done.has(logical), 'completed row reached native decoding');
      assert.deepEqual(tensorRows(mask, row), maskFor(logical, step), 'left padding and full mask width must remain unchanged');
      const rope = [0, 1, 2].map((axis) => Number(positions.data[axis * count + row]));
      assert.deepEqual(rope, positionsFor(logical, step), 'original three-axis logical RoPE positions must not shift');
      for (const tensor of Object.values(physical)) assert.equal(tensor.dims[0], count, 'native cache must be physically compact');
      next.push(advance({
        key: tensorRows(physical[pastName(names[0])]!, row), value: tensorRows(physical[pastName(names[1])]!, row),
        recurrent: tensorRows(physical[pastName(names[2])]!, row), conv: tensorRows(physical[pastName(names[3])]!, row),
      }, Number(ids.data[row]), rope, tensorRows(mask, row)));
    }
    if (fail) throw new Error('native forward failed');
    const outputs: Values = {};
    for (const [name, field, dims] of [
      [names[0], 'key', [count, 2, step + 3, 2]], [names[1], 'value', [count, 2, step + 3, 2]],
      [names[2], 'recurrent', [count, 2]], [names[3], 'conv', [count, 3]],
    ] as const) outputs[name] = presents.track(new Tensor('float32', Float32Array.from(next.flatMap((state) => state[field])), [...dims]));
    outputs.logits = raw.track(new Tensor('float32', Float32Array.from(next.flatMap(logits)), [count, 1, 4]));
    for (const tensor of Object.values(outputs) as Tensor[]) allOutputs.add(tensor);
    lastNative = outputs;
    return outputs;
  };
  const decode = async () => {
    const active = Array.from({ length: size }, (_, row) => row).filter((row) => !done.has(row));
    const ids = caller.track(new Tensor('int64', BigInt64Array.from(Array.from({ length: size }, (_, row) => BigInt(10 + row + step * 5))), [size, 1]));
    const mask = caller.track(new Tensor('int64', BigInt64Array.from(Array.from({ length: size }, (_, row) => maskFor(row, step)).flat().map(BigInt)), [size, step + 3]));
    const positions = caller.track(new Tensor('int64', BigInt64Array.from([0, 1, 2].flatMap((axis) => Array.from({ length: size }, (_, row) => BigInt(positionsFor(row, step)[axis]!)))), [3, size, 1]));
    const ropeDeltas = caller.track(new Tensor('int64', BigInt64Array.from(Array.from({ length: size }, (_, row) => BigInt(row + 31))), [size, 1]));
    const inputs: Values = { input_ids: ids, attention_mask: mask, position_ids: positions, past_key_values: cache, rope_deltas: ropeDeltas };
    const output = await decoder.forward(inputs, native);
    const outLogits = output.logits as Tensor;
    assert.deepEqual(outLogits.dims, [size, 1, 4], 'processors must see all original logical slots');
    for (const [physical, logical] of active.entries()) {
      states[logical] = advance(states[logical]!, 10 + logical + step * 5, positionsFor(logical, step), maskFor(logical, step));
      assert.deepEqual(tensorRows(outLogits, logical), logits(states[logical]!), 'surviving logical row decoded with wrong history');
      for (const [name, field] of [[names[0], 'key'], [names[1], 'value'], [names[2], 'recurrent'], [names[3], 'conv']] as const) {
        const present = output[name] as Tensor;
        assert.equal(present.dims[0], active.length);
        assert.deepEqual(tensorRows(present, physical), states[logical]![field]);
      }
    }
    if (active.length < size) {
      assert.ok(raw.disposed.has(lastNative!.logits as Tensor), 'replaced physical logits must be disposed');
      temporary.assertDisposed();
    }
    // Transformers slices a separate last-token tensor before processors release its previous copy.
    const copy = outLogits.slice(null, -1, null);
    release._call([], copy);
    const entries: Record<string, Tensor> = {};
    for (const name of names) entries[pastName(name)] = output[name] as Tensor;
    cache.update(entries);
    caller.assertRetained(); presents.assertRetained();
    assert.equal(inputs.past_key_values, cache);
    assert.deepEqual(numbers(ropeDeltas), Array.from({ length: size }, (_, row) => row + 31));
    step++;
    return { output, copy };
  };
  const dispose = () => {
    decoder.dispose(); temporary.assertDisposed(); caller.assertRetained(); presents.assertRetained();
    for (const tensor of allOutputs) if (!raw.disposed.has(tensor)) tensor.dispose();
    for (const tensor of caller.observed) tensor.dispose();
  };
  return { decoder, done, caller, temporary, cache, calls, decode, dispose,
    set fail(value: boolean) { fail = value; },
  };
}

test('uncompacted decoding preserves independent full-batch histories', async () => {
  const run = scenario(2);
  try {
    for (let index = 0; index < 3; index++) {
      const { copy } = await run.decode();
      copy.dispose();
    }
    assert.deepEqual(run.calls, [[0, 1], [0, 1], [0, 1]]);
  } finally { run.dispose(); }
});

test('B2 to logical row 1 keeps left padding, original RoPE and recurrent history over repeated decode', async () => {
  const run = scenario(2); let logical: Tensor | undefined;
  try {
    (await run.decode()).copy.dispose(); run.done.add(0);
    for (let index = 0; index < 4; index++) {
      const { output, copy } = await run.decode(); logical = output.logits as Tensor; copy.dispose();
    }
    assert.deepEqual(run.calls, [[0, 1], [1], [1], [1], [1]]);
    assert.ok(logical);
    const retained = ownership(); retained.track(logical);
    run.decoder.dispose();
    // The helper owns its reusable scatter tensor, never the caller's logical inputs or cache.
    assert.ok(retained.disposed.has(logical)); run.caller.assertRetained();
  } finally { run.dispose(); }
});

test('B4 to B3 to B1 gathers current physical cache indices only when membership shrinks', async () => {
  const run = scenario(4);
  try {
    (await run.decode()).copy.dispose(); run.done.add(0);
    (await run.decode()).copy.dispose(); (await run.decode()).copy.dispose();
    run.done.add(1); run.done.add(2);
    (await run.decode()).copy.dispose(); (await run.decode()).copy.dispose();
    assert.deepEqual(run.calls, [[0, 1, 2, 3], [1, 2, 3], [1, 2, 3], [3], [3]]);
  } finally { run.dispose(); }
});

for (const compacted of [false, true]) test(`native failure ${compacted ? 'after' : 'during first'} compaction cleans owned tensors and retains caller state`, async () => {
  const run = scenario(2);
  try {
    (await run.decode()).copy.dispose(); run.done.add(0);
    const logical = ownership();
    if (compacted) {
      const { output, copy } = await run.decode(); copy.dispose(); logical.track(output.logits as Tensor);
    }
    run.fail = true;
    const retained = ownership();
    for (const tensor of Object.values(run.cache)) retained.track(tensor);
    const before = Object.values(run.cache).map(numbers);
    await assert.rejects(run.decode(), /native forward failed/);
    run.temporary.assertDisposed(); run.caller.assertRetained(); retained.assertRetained();
    logical.assertDisposed();
    assert.deepEqual(Object.values(run.cache).map(numbers), before);
    run.decoder.dispose(); retained.assertRetained();
  } finally { run.dispose(); }
});

test('GPU-resident hybrid cache keeps finished rows as padding instead of a CPU round trip', async () => {
  const done = new Set<number>([0]); const decoder = new BatchDecoder(2, (row) => done.has(row));
  let downloads = 0;
  const gpuLike = (tensor: Tensor) => {
    Object.defineProperties(tensor, { location: { get: () => 'gpu-buffer' }, data: { get: () => { throw new Error('GPU cache read on the CPU'); } } });
    tensor.ort_tensor.getData = async () => { downloads++; throw new Error('GPU cache downloaded'); };
    return tensor;
  };
  const cache = new DynamicCache({
    [pastName(names[0])]: gpuLike(new Tensor('float32', new Float32Array(16), [2, 2, 2, 2])),
    [pastName(names[2])]: gpuLike(new Tensor('float32', new Float32Array(4), [2, 2])),
  });
  const inputs: Values = {
    input_ids: new Tensor('int64', BigInt64Array.from([10n, 11n]), [2, 1]),
    attention_mask: new Tensor('int64', BigInt64Array.from([1n, 1n, 1n, 1n]), [2, 2]),
    position_ids: new Tensor('int64', BigInt64Array.from([0n, 0n, 1n, 1n, 2n, 2n]), [3, 2, 1]),
    past_key_values: cache,
  };
  const outputs: Values = { logits: new Tensor('float32', new Float32Array(8), [2, 1, 4]) };
  let received: Values | undefined;
  const native = async (passed: Values) => { received = passed; return outputs; };
  try {
    for (let step = 0; step < 2; step++) {
      assert.equal(await decoder.forward(inputs, native), outputs, 'the native outputs pass through unchanged');
      assert.equal(received, inputs, 'the full logical batch reaches the native decoder untouched');
    }
    assert.equal(downloads, 0);
  } finally { decoder.dispose(); (outputs.logits as Tensor).dispose(); for (const tensor of Object.values(inputs)) if (tensor instanceof Tensor) tensor.dispose(); }
});

test('empty active batch is rejected before native dispatch and caller state survives', async () => {
  const run = scenario(2);
  try {
    run.done.add(0); run.done.add(1);
    await assert.rejects(run.decode());
    assert.deepEqual(run.calls, []); run.caller.assertRetained();
  } finally { run.dispose(); }
});
