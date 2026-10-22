/**
 * In-memory ONNX graph patch applied after SHA-256 verification; cached files are never modified.
 *
 * The pinned q4 decoders leave MatMulNBits `accuracy_level` unset (0), so native CPU ONNX Runtime multiplies fp32
 * activations against dequantized weights. Level 4 lets it quantize each activation block to int8 and use integer
 * dot-product kernels, which is several times faster on the decoder's per-token matrix-vector products.
 */
export const INT8_MATMUL_GRAPHS: Readonly<Record<string, true>> = { 'onnx/decoder_model_merged_q4.onnx': true };

const decoder = new TextDecoder();
// AttributeProto { name: "accuracy_level" (1), i: 4 (3), type: INT = 2 (20) }, framed as NodeProto.attribute (5).
const ACCURACY_LEVEL_INT8 = Uint8Array.of(0x2a, 21, 0x0a, 14, ...new TextEncoder().encode('accuracy_level'), 0x18, 4, 0xa0, 0x01, 2);

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
function header(field: number, length: number): Uint8Array {
  const bytes = [field * 8 + 2];
  for (; length >= 0x80; length = Math.floor(length / 128)) bytes.push((length % 128) | 0x80);
  bytes.push(length);
  return Uint8Array.from(bytes);
}
const text = (bytes: Uint8Array, field: Field) => decoder.decode(bytes.subarray(field.body, field.end));

/** Rewrites repeated length-delimited `field` entries of one message; `undefined` means unchanged. */
function rewrite(bytes: Uint8Array, start: number, end: number, field: number, child: (start: number, end: number) => Uint8Array[] | undefined): Uint8Array[] | undefined {
  const parts: Uint8Array[] = []; let copied = start;
  for (const entry of fields(bytes, start, end)) {
    if (entry.number !== field || entry.wire !== 2) continue;
    const replaced = child(entry.body, entry.end);
    if (!replaced) continue;
    parts.push(bytes.subarray(copied, entry.start), header(field, replaced.reduce((total, part) => total + part.byteLength, 0)), ...replaced);
    copied = entry.end;
  }
  if (!parts.length) return undefined;
  parts.push(bytes.subarray(copied, end));
  return parts;
}
function node(bytes: Uint8Array, start: number, end: number): Uint8Array[] | undefined {
  let op = ''; let domain = ''; let explicit = false;
  for (const entry of fields(bytes, start, end)) {
    if (entry.wire !== 2) continue;
    if (entry.number === 4) op = text(bytes, entry);
    else if (entry.number === 7) domain = text(bytes, entry);
    else if (entry.number === 5) for (const attribute of fields(bytes, entry.body, entry.end)) if (attribute.number === 1 && attribute.wire === 2 && text(bytes, attribute) === 'accuracy_level') explicit = true;
  }
  // Repeated fields may be appended in any order. Explicit levels and nested subgraphs (absent in the pinned decoders) are left as published.
  return op === 'MatMulNBits' && domain === 'com.microsoft' && !explicit ? [bytes.subarray(start, end), ACCURACY_LEVEL_INT8] : undefined;
}

/** Returns a ModelProto whose top-level MatMulNBits nodes without an explicit accuracy level use int8 activations. */
export function withInt8MatMulNBits(model: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> {
  const parts = rewrite(model, 0, model.byteLength, 7, (start, end) => rewrite(model, start, end, 1, (first, last) => node(model, first, last)));
  if (!parts) return model;
  const result = new Uint8Array(parts.reduce((total, part) => total + part.byteLength, 0));
  let offset = 0;
  for (const part of parts) { result.set(part, offset); offset += part.byteLength; }
  return result;
}
