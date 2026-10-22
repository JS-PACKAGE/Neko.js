import assert from 'node:assert/strict';
import test from 'node:test';
import ort from 'onnxruntime-node';
import { withFp32MatMul, withInt8MatMulNBits, withoutAttentionBias } from '../src/cache/graph-patch.js';

// Minimal ONNX protobuf writer: enough for a MatMulNBits graph that native ONNX Runtime can execute.
const utf8 = new TextEncoder();
function varint(value: number): number[] { const bytes = []; for (; value >= 0x80; value = Math.floor(value / 128)) bytes.push((value % 128) | 0x80); bytes.push(value); return bytes; }
const int = (field: number, value: number) => [...varint(field * 8), ...varint(value)];
const bytes = (field: number, payload: ArrayLike<number>) => [...varint(field * 8 + 2), ...varint(payload.length), ...Array.from(payload)];
const str = (field: number, value: string) => bytes(field, utf8.encode(value));
const intAttribute = (name: string, value: number) => bytes(5, [...str(1, name), ...int(3, value), ...int(20, 2)]);
const tensor = (name: string, type: number, dims: number[], raw: Uint8Array) => bytes(5, [...dims.flatMap((dim) => int(1, dim)), ...int(2, type), ...str(8, name), ...bytes(9, raw)]);
const valueInfo = (field: number, name: string, dims: number[]) => bytes(field, [...str(1, name), ...bytes(2, bytes(1, [...int(1, 1), ...bytes(2, dims.flatMap((dim) => bytes(1, int(1, dim))))]))]);
const K = 64; const N = 8; const BLOCK = 32;

/** copy = Identity(MatMulNBits(A)); `level` is the MatMulNBits node's explicit accuracy level, if any. */
function model(level?: number): Uint8Array<ArrayBuffer> {
  const packed = Uint8Array.from({ length: N * (K / BLOCK) * (BLOCK / 2) }, (_, index) => (index * 37 + 11) % 256);
  const scales = new Float32Array(N * (K / BLOCK)).map((_, index) => 0.01 + (index % 5) * 0.003);
  const graph = [
    ...bytes(1, [...str(1, 'A'), ...str(1, 'B'), ...str(1, 'scales'), ...str(2, 'hidden'), ...str(3, 'matmul'), ...str(4, 'MatMulNBits'), ...str(7, 'com.microsoft'),
      ...intAttribute('K', K), ...intAttribute('N', N), ...intAttribute('bits', 4), ...intAttribute('block_size', BLOCK), ...(level === undefined ? [] : intAttribute('accuracy_level', level))]),
    ...bytes(1, [...str(1, 'hidden'), ...str(2, 'copy'), ...str(4, 'Identity')]),
    ...str(2, 'g'),
    ...tensor('B', 2, [N, K / BLOCK, BLOCK / 2], packed), ...tensor('scales', 1, [N * (K / BLOCK)], new Uint8Array(scales.buffer)),
    ...valueInfo(11, 'A', [1, K]), ...valueInfo(12, 'copy', [1, N]),
  ];
  return Uint8Array.from([...int(1, 8), ...bytes(8, [...str(1, ''), ...int(2, 17)]), ...bytes(8, [...str(1, 'com.microsoft'), ...int(2, 1)]), ...bytes(7, graph)]);
}
async function run(graph: Uint8Array): Promise<Float32Array> {
  const session = await ort.InferenceSession.create(graph);
  try {
    const input = Float32Array.from({ length: K }, (_, index) => Math.sin(index * 0.7) * 3);
    const { copy } = await session.run({ A: new ort.Tensor('float32', input, [1, K]) });
    return copy!.data as Float32Array;
  } finally { await session.release(); }
}

test('int8 MatMulNBits patch keeps the graph executable and numerically close, and leaves explicit levels as published', async () => {
  const original = model();
  const patched = withInt8MatMulNBits(original);
  assert.notEqual(patched, original);
  assert.equal(withInt8MatMulNBits(patched), patched, 'an added accuracy level is recognised as explicit');
  const [reference, int8] = await Promise.all([run(original), run(patched)]);
  const scale = Math.max(...reference.map(Math.abs));
  assert.ok(scale > 0);
  for (let index = 0; index < N; index++) assert.ok(Math.abs(int8[index]! - reference[index]!) <= 0.05 * scale, `output ${index}: ${int8[index]} vs ${reference[index]}`);
  const explicit = model(1);
  assert.equal(withInt8MatMulNBits(explicit), explicit);
  assert.throws(() => withInt8MatMulNBits(original.slice(0, original.length - 3)), TypeError);
});

// fp16 bits of a float32 that fp16 represents exactly (Node 22 has no Float16Array).
const halfBits = (value: number) => { const bits = new Uint32Array(Float32Array.of(value).buffer)[0]!; const exponent = ((bits >>> 23) & 0xff) - 112; return ((bits >>> 16) & 0x8000) | (exponent > 0 ? (exponent << 10) | ((bits >>> 13) & 0x3ff) : 0); };

/** sum = MatMul(half(X), W1) + MatMul(half(X), W2), with the weights stored as `type` (fp16 = 10, fp32 = 1). */
function halfModel(type = 10): Uint8Array<ArrayBuffer> {
  const weight = (seed: number) => {
    const values = Array.from({ length: K * N }, (_, index) => ((index * 5 + seed) % 17 - 8) / 16);
    return type === 10 ? new Uint8Array(Uint16Array.from(values, halfBits).buffer) : new Uint8Array(Float32Array.from(values).buffer);
  };
  const half = type === 10 ? 'Xh' : 'X';
  const graph = [
    ...(type === 10 ? bytes(1, [...str(1, 'X'), ...str(2, 'Xh'), ...str(4, 'Cast'), ...intAttribute('to', 10)]) : []),
    ...bytes(1, [...str(1, half), ...str(1, 'W1'), ...str(2, 'Y1'), ...str(3, 'first'), ...str(4, 'MatMul')]),
    ...bytes(1, [...str(1, half), ...str(1, 'W2'), ...str(2, 'Y2'), ...str(4, 'MatMul')]),
    ...bytes(1, [...str(1, 'Y1'), ...str(1, 'Y2'), ...str(2, 'S'), ...str(4, 'Add')]),
    ...(type === 10 ? bytes(1, [...str(1, 'S'), ...str(2, 'sum'), ...str(4, 'Cast'), ...intAttribute('to', 1)]) : bytes(1, [...str(1, 'S'), ...str(2, 'sum'), ...str(4, 'Identity')])),
    ...str(2, 'g'),
    ...tensor('W1', type, [K, N], weight(1)), ...tensor('W2', type, [K, N], weight(2)),
    ...valueInfo(11, 'X', [2, K]), ...valueInfo(12, 'sum', [2, N]),
  ];
  return Uint8Array.from([...int(1, 8), ...bytes(8, [...str(1, ''), ...int(2, 17)]), ...bytes(7, graph)]);
}

test('fp32 MatMul patch computes fp16-weight MatMuls in fp32, casts shared inputs once and ignores fp32 weights', async () => {
  const original = halfModel();
  const patched = withFp32MatMul(original);
  assert.notEqual(patched, original);
  assert.equal(withFp32MatMul(patched), patched, 'patched MatMuls no longer read fp16 initializers');
  const exact = halfModel(1);
  assert.equal(withFp32MatMul(exact), exact);
  const input = Float32Array.from({ length: 2 * K }, (_, index) => ((index * 7) % 13 - 6) / 4);
  const evaluate = async (graph: Uint8Array) => {
    const session = await ort.InferenceSession.create(graph);
    try { return (await session.run({ X: new ort.Tensor('float32', input, [2, K]) })).sum!.data as Float32Array; } finally { await session.release(); }
  };
  const [reference, half, fp32] = await Promise.all([evaluate(exact), evaluate(original), evaluate(patched)]);
  // Inputs and weights are exact in fp16; the graphs differ from the fp32 reference only by fp16 intermediate and output rounding.
  for (let index = 0; index < 2 * N; index++) {
    const tolerance = 0.01 * Math.max(1, Math.abs(reference[index]!));
    assert.ok(Math.abs(fp32[index]! - reference[index]!) <= tolerance, `fp32 output ${index}: ${fp32[index]} vs ${reference[index]}`);
    assert.ok(Math.abs(half[index]! - reference[index]!) <= tolerance, `fp16 output ${index}: ${half[index]} vs ${reference[index]}`);
  }
});

const S = 4; const HIDDEN = 8;
/** out = MultiHeadAttention(Q, K, V, attention_bias = `bias`), the bias stored as an initializer. */
function attentionModel(bias: number[]): Uint8Array<ArrayBuffer> {
  const graph = [
    ...bytes(1, [...str(1, 'Q'), ...str(1, 'K'), ...str(1, 'V'), ...str(1, ''), ...str(1, ''), ...str(1, 'bias'), ...str(2, 'out'), ...str(4, 'MultiHeadAttention'), ...str(7, 'com.microsoft'), ...intAttribute('num_heads', 2)]),
    ...str(2, 'g'),
    ...tensor('bias', 1, [1, 1, S, S], new Uint8Array(Float32Array.from(bias).buffer)),
    ...valueInfo(11, 'Q', [1, S, HIDDEN]), ...valueInfo(11, 'K', [1, S, HIDDEN]), ...valueInfo(11, 'V', [1, S, HIDDEN]), ...valueInfo(12, 'out', [1, S, HIDDEN]),
  ];
  return Uint8Array.from([...int(1, 8), ...bytes(8, [...str(1, ''), ...int(2, 17)]), ...bytes(8, [...str(1, 'com.microsoft'), ...int(2, 1)]), ...bytes(7, graph)]);
}

test('attention bias removal drops only the MultiHeadAttention bias input', async () => {
  const values = (seed: number) => Float32Array.from({ length: S * HIDDEN }, (_, index) => Math.sin(index * 0.37 + seed));
  const evaluate = async (graph: Uint8Array) => {
    const session = await ort.InferenceSession.create(graph);
    try { return (await session.run({ Q: new ort.Tensor('float32', values(1), [1, S, HIDDEN]), K: new ort.Tensor('float32', values(2), [1, S, HIDDEN]), V: new ort.Tensor('float32', values(3), [1, S, HIDDEN]) })).out!.data as Float32Array; } finally { await session.release(); }
  };
  // Two "images" of two tokens each, masked from each other the way the pinned vision encoders do.
  const masked = attentionModel(Array.from({ length: S * S }, (_, index) => (Math.floor(index / S) < 2) === (index % S < 2) ? 0 : -10000));
  const zero = attentionModel(new Array<number>(S * S).fill(0));
  const patched = withoutAttentionBias(masked);
  assert.notEqual(patched, masked);
  assert.equal(withoutAttentionBias(patched), patched, 'a removed bias is not removed twice');
  const [withMask, withZero, without] = await Promise.all([evaluate(masked), evaluate(zero), evaluate(patched)]);
  for (let index = 0; index < S * HIDDEN; index++) assert.ok(Math.abs(without[index]! - withZero[index]!) < 1e-5, `output ${index}: ${without[index]} vs ${withZero[index]}`);
  assert.ok(withMask.some((value, index) => Math.abs(value - without[index]!) > 1e-3), 'the patched graph no longer applies the cross-image mask');
});
