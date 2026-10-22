import assert from 'node:assert/strict';
import test from 'node:test';
import ort from 'onnxruntime-node';
import { withInt8MatMulNBits } from '../src/cache/graph-patch.js';

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
