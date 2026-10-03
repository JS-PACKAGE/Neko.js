import type { OwnedDocumentBytes } from './types.js';

export function integer(value: number | undefined, fallback: number, maximum: number, name: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) throw new RangeError(`${name} must be between 1 and ${maximum}`);
  return result;
}
export async function ownedBytes(input: OwnedDocumentBytes, maximum: number, signal?: AbortSignal): Promise<Uint8Array<ArrayBuffer>> {
  signal?.throwIfAborted();
  if (input instanceof Blob) {
    if (!input.size || input.size > maximum) throw new RangeError('Document input exceeds byte bounds');
    const bytes = new Uint8Array(await input.arrayBuffer());
    signal?.throwIfAborted();
    return bytes;
  }
  const source = input instanceof ArrayBuffer ? new Uint8Array(input) : input;
  if (!(source instanceof Uint8Array)) throw new TypeError('Document input must be owned bytes, an ArrayBuffer, or a Blob; paths and URLs are not accepted');
  if (!source.byteLength || source.byteLength > maximum) throw new RangeError('Document input exceeds byte bounds');
  return new Uint8Array(source);
}
