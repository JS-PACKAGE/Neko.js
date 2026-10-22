/**
 * In-memory ONNX graph patches, applied after SHA-256 verification; cached files are never modified.
 *
 * Native Node CPU:
 * - The pinned q4 decoder leaves MatMulNBits `accuracy_level` unset (0), so ONNX Runtime multiplies fp32 activations
 *   against dequantized weights. Level 4 lets it quantize each activation block to int8 and use integer dot-product
 *   kernels, which is several times faster on the decoder's per-token matrix-vector products.
 * - Opt-in (`NekoOptions.cpuVisionPrecision: 'fp32'`): the fp16 vision encoder's MatMuls otherwise run ONNX Runtime's CPU
 *   fp16 GEMM, which (with KleidiAI disabled, see `VisionEngine.load`) measured about 2.5 times slower than its fp32 SGEMM.
 *   Casting each fp16 weight initializer to fp32 lets session creation constant-fold fp32 weights (about 300 MB more
 *   resident memory after load); activations are cast around each MatMul.
 *
 * Browser WebGPU:
 * - The pinned 0.8B vision encoders give every MultiHeadAttention a [1, 1, N, N] bias that only masks attention between
 *   different images. ONNX Runtime Web's WebGPU attention with a bias was about 10 times slower on a 4,900-patch image
 *   than without one. The engine encodes one image per run (`encodeImagesSeparately`), where that bias is all zeros, so the
 *   patch drops the bias input.
 */
export type GraphPatch = (model: Uint8Array<ArrayBuffer>) => Uint8Array<ArrayBuffer>;
/** Patches for native Node CPU sessions, keyed by pinned file name. */
export function cpuGraphPatches(fp32Vision: boolean): Readonly<Record<string, GraphPatch>> {
  return { 'onnx/decoder_model_merged_q4.onnx': withInt8MatMulNBits, ...(fp32Vision ? { 'onnx/vision_encoder_fp16.onnx': withFp32MatMul } : {}) };
}
/** Patches for browser WebGPU sessions of the pinned 0.8B model, keyed by pinned file name. */
export const WEBGPU_GRAPH_PATCHES: Readonly<Record<string, GraphPatch>> = { 'onnx/vision_encoder_fp16.onnx': withoutAttentionBias, 'onnx/vision_encoder_q4.onnx': withoutAttentionBias };

const decoder = new TextDecoder();
const encoder = new TextEncoder();
// AttributeProto { name: "accuracy_level" (1), i: 4 (3), type: INT = 2 (20) }, framed as NodeProto.attribute (5).
const ACCURACY_LEVEL_INT8 = Uint8Array.of(0x2a, 21, 0x0a, 14, ...encoder.encode('accuracy_level'), 0x18, 4, 0xa0, 0x01, 2);
const FLOAT = 1; const FLOAT16 = 10;

interface Field { number: number; wire: number; start: number; body: number; end: number }
function readVarint(bytes: Uint8Array, offset: number): [value: number, next: number] {
  let value = 0;
  for (let index = offset, scale = 1; index < bytes.length && index < offset + 10; index++, scale *= 128) {
    const byte = bytes[index]!;
    value += (byte & 0x7f) * scale;
    if (byte < 0x80) return [value, index + 1];
  }
  throw new TypeError('Malformed ONNX protobuf varint');
}
function* fields(bytes: Uint8Array, start: number, end: number): Generator<Field> {
  for (let offset = start; offset < end;) {
    const [tag, body] = readVarint(bytes, offset);
    const wire = tag % 8;
    let payload = body; let next: number;
    if (wire === 0) next = readVarint(bytes, body)[1];
    else if (wire === 1) next = body + 8;
    else if (wire === 5) next = body + 4;
    else if (wire === 2) { const [length, first] = readVarint(bytes, body); payload = first; next = first + length; }
    else throw new TypeError('Unsupported ONNX protobuf wire type');
    if (next > end) throw new TypeError('Truncated ONNX protobuf field');
    yield { number: Math.floor(tag / 8), wire, start: offset, body: payload, end: next };
    offset = next;
  }
}
const byteLength = (parts: Uint8Array[]) => parts.reduce((total, part) => total + part.byteLength, 0);
/** Frames `parts` as one length-delimited `field` entry. */
function framed(field: number, parts: Uint8Array[]): Uint8Array[] {
  const bytes = [field * 8 + 2];
  let length = byteLength(parts);
  for (; length >= 0x80; length = Math.floor(length / 128)) bytes.push((length % 128) | 0x80);
  bytes.push(length);
  return [Uint8Array.from(bytes), ...parts];
}
const string = (field: number, value: string) => framed(field, [encoder.encode(value)]);
const text = (bytes: Uint8Array, field: Field) => decoder.decode(bytes.subarray(field.body, field.end));

/** Replaces length-delimited `field` entries of one message with `child`'s framed entries; `undefined` means unchanged. */
function rewrite(bytes: Uint8Array, start: number, end: number, field: number, child: (entry: Field) => Uint8Array[] | undefined): Uint8Array[] | undefined {
  const parts: Uint8Array[] = []; let copied = start;
  for (const entry of fields(bytes, start, end)) {
    if (entry.number !== field || entry.wire !== 2) continue;
    const replaced = child(entry);
    if (!replaced) continue;
    parts.push(bytes.subarray(copied, entry.start), ...replaced);
    copied = entry.end;
  }
  if (!parts.length) return undefined;
  parts.push(bytes.subarray(copied, end));
  return parts;
}
/** Applies `forGraph(graph)` to the top-level graph's NodeProto entries. Nested subgraphs (absent in the pinned graphs) are left as published. */
function patchNodes(model: Uint8Array<ArrayBuffer>, forGraph: (graph: Field) => (entry: Field) => Uint8Array[] | undefined): Uint8Array<ArrayBuffer> {
  const parts = rewrite(model, 0, model.byteLength, 7, (graph) => {
    const body = rewrite(model, graph.body, graph.end, 1, forGraph(graph));
    return body && framed(7, body);
  });
  if (!parts) return model;
  const result = new Uint8Array(byteLength(parts));
  let offset = 0;
  for (const part of parts) { result.set(part, offset); offset += part.byteLength; }
  return result;
}
function describe(bytes: Uint8Array, entry: Field) {
  const node = { op: '', domain: '', inputs: [] as string[], outputs: [] as string[], accuracyLevel: false };
  for (const field of fields(bytes, entry.body, entry.end)) {
    if (field.wire !== 2) continue;
    if (field.number === 1) node.inputs.push(text(bytes, field));
    else if (field.number === 2) node.outputs.push(text(bytes, field));
    else if (field.number === 4) node.op = text(bytes, field);
    else if (field.number === 7) node.domain = text(bytes, field);
    else if (field.number === 5) for (const attribute of fields(bytes, field.body, field.end)) if (attribute.number === 1 && attribute.wire === 2 && text(bytes, attribute) === 'accuracy_level') node.accuracyLevel = true;
  }
  return node;
}

/** Returns a ModelProto whose top-level MatMulNBits nodes without an explicit accuracy level use int8 activations. */
export function withInt8MatMulNBits(model: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> {
  return patchNodes(model, () => (entry) => {
    const { op, domain, accuracyLevel } = describe(model, entry);
    // Repeated fields may be appended in any order; explicit levels are left as published.
    return op === 'MatMulNBits' && domain === 'com.microsoft' && !accuracyLevel ? framed(1, [model.subarray(entry.body, entry.end), ACCURACY_LEVEL_INT8]) : undefined;
  });
}

/** Returns a ModelProto whose top-level com.microsoft MultiHeadAttention nodes have no `attention_bias` (input 5). */
export function withoutAttentionBias(model: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> {
  return patchNodes(model, () => (entry) => {
    const { op, domain, inputs } = describe(model, entry);
    if (op !== 'MultiHeadAttention' || domain !== 'com.microsoft' || !inputs[5]) return undefined;
    const body: Uint8Array[] = []; let input = 0;
    // An empty name marks an omitted optional input; later inputs keep their positions.
    for (const field of fields(model, entry.body, entry.end)) body.push(field.number === 1 && field.wire === 2 && input++ === 5 ? new Uint8Array([0x0a, 0]) : model.subarray(field.start, field.end));
    return framed(1, body);
  });
}

const cast = (input: string, output: string, to: number) => framed(1, [...string(1, input), ...string(2, output), ...string(4, 'Cast'), ...framed(5, [...string(1, 'to'), Uint8Array.of(0x18, to, 0xa0, 0x01, 2)])]);
/** Returns a ModelProto whose top-level MatMuls with an fp16 initializer weight compute in fp32 and still produce fp16. */
export function withFp32MatMul(model: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> {
  return patchNodes(model, (graph) => {
    const weights = new Set<string>(); const converted = new Set<string>();
    for (const initializer of fields(model, graph.body, graph.end)) {
      if (initializer.number !== 5 || initializer.wire !== 2) continue;
      let type = 0; let name = '';
      for (const field of fields(model, initializer.body, initializer.end)) {
        if (field.number === 2 && field.wire === 0) type = readVarint(model, field.body)[0];
        else if (field.number === 8 && field.wire === 2) name = text(model, field);
      }
      if (type === FLOAT16) weights.add(name);
    }
    return (entry) => {
      const { op, domain, inputs, outputs } = describe(model, entry);
      if (op !== 'MatMul' || domain !== '' || inputs.length !== 2 || outputs.length !== 1 || !weights.has(inputs[1]!)) return undefined;
      const parts: Uint8Array[] = [];
      // Names gain a suffix; a tensor shared by several MatMuls (such as a projection input) is cast once, before its first use.
      for (const name of inputs) if (!converted.has(name)) { converted.add(name); parts.push(...cast(name, `${name}/neko_fp32`, FLOAT)); }
      const output = outputs[0]!; const body: Uint8Array[] = []; let input = 0;
      for (const field of fields(model, entry.body, entry.end)) {
        if (field.number === 1 && field.wire === 2) body.push(...string(1, `${inputs[input++]!}/neko_fp32`));
        else if (field.number === 2 && field.wire === 2) body.push(...string(2, `${output}/neko_fp32`));
        else body.push(model.subarray(field.start, field.end));
      }
      return [...parts, ...framed(1, body), ...cast(`${output}/neko_fp32`, output, FLOAT16)];
    };
  });
}
