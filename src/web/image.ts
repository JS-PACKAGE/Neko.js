import { RawImage } from '@huggingface/transformers';
import type * as NodeFs from 'node:fs/promises';
import type * as NodeUrl from 'node:url';
import type sharp from 'sharp';
import type { PageImage } from '../types.js';
import { boundedBytes, fetchLimited, type ExtractOptions } from './extract.js';
import { authorizeLocalFile, policyDestination } from './policy.js';
import type { ResourcePolicy } from './policy.js';

export interface DecodedImage { data: Uint8Array | Uint8ClampedArray; width: number; height: number; channels: 1 | 2 | 3 | 4; }
export type ImageInput = string | URL | Blob | DecodedImage;
export type ImageOptions = Pick<ExtractOptions, 'signal' | 'maxImageBytes' | 'timeoutMs' | 'validateDestination' | '_policy' | '_offline'>;
const maxPixels = 40_000_000;
const maxDimension = 1280;
const isNode = typeof process !== 'undefined' && process.release?.name === 'node';

function rasterType(bytes: Uint8Array): string | undefined {
  if (bytes.length >= 8 && [137, 80, 78, 71, 13, 10, 26, 10].every((byte, index) => bytes[index] === byte)) return 'image/png';
  if (bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return 'image/jpeg';
  if (bytes.length >= 12 && String.fromCharCode(...bytes.subarray(0, 4)) === 'RIFF' && String.fromCharCode(...bytes.subarray(8, 12)) === 'WEBP') return 'image/webp';
  if (bytes.length >= 6 && /^GIF8[79]a$/.test(String.fromCharCode(...bytes.subarray(0, 6)))) return 'image/gif';
  if (bytes.length >= 2 && bytes[0] === 66 && bytes[1] === 77) return 'image/bmp';
  if (bytes.length >= 4 && ((bytes[0] === 73 && bytes[1] === 73 && bytes[2] === 42 && bytes[3] === 0) || (bytes[0] === 77 && bytes[1] === 77 && bytes[2] === 0 && bytes[3] === 42))) return 'image/tiff';
  if (bytes.length >= 12 && String.fromCharCode(...bytes.subarray(4, 8)) === 'ftyp' && ['avif', 'avis'].includes(String.fromCharCode(...bytes.subarray(8, 12)))) return 'image/avif';
  return undefined;
}

function dataBytes(url: string, maxBytes: number): { bytes: Uint8Array<ArrayBuffer>; type: string } {
  const match = /^data:(image\/(?:png|jpeg|webp|gif|avif|tiff|bmp))(;base64)?,(.*)$/is.exec(url);
  if (!match) throw new TypeError('Only raster image data URLs are supported');
  const payload = match[3]!;
  let bytes: Uint8Array<ArrayBuffer>;
  if (match[2]) {
    const compact = payload.replace(/\s/g, '');
    if (Math.floor(compact.length * 3 / 4) - (compact.endsWith('==') ? 2 : compact.endsWith('=') ? 1 : 0) > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    const binary = atob(compact);
    bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  } else {
    let length = 0;
    for (let index = 0; index < payload.length; index++) {
      if (payload[index] === '%') {
        if (!/^[\da-f]{2}$/i.test(payload.slice(index + 1, index + 3))) throw new TypeError('Invalid percent-encoded image data URL');
        index += 2;
      } else if (payload.charCodeAt(index) > 127) throw new TypeError('Non-ASCII data URL bytes must be percent-encoded');
      if (++length > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    }
    bytes = new Uint8Array(length);
    let offset = 0;
    for (let index = 0; index < payload.length; index++) {
      if (payload[index] === '%') { bytes[offset++] = Number.parseInt(payload.slice(index + 1, index + 3), 16); index += 2; }
      else bytes[offset++] = payload.charCodeAt(index);
    }
  }
  if (bytes.byteLength > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
  return { bytes, type: match[1]!.toLowerCase() };
}

async function localBytes(path: string | URL, maxBytes: number, signal: AbortSignal, policy?: ResourcePolicy): Promise<Uint8Array<ArrayBuffer>> {
  // Node filesystem modules are absent from browser runtimes.
  const protocol = 'node:';
  const fs: typeof NodeFs = await import(`${protocol}fs/promises`);
  const url: typeof NodeUrl = await import(`${protocol}url`);
  const canonical = await fs.realpath(path instanceof URL ? url.fileURLToPath(path) : path);
  if (policy) await authorizeLocalFile(policy, canonical, signal);
  signal.throwIfAborted();
  const handle = await fs.open(canonical, 'r');
  try {
    const info = await handle.stat();
    if (!info.isFile()) throw new TypeError('Image path must point to a regular file');
    if (info.size > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    const chunks: Uint8Array[] = [];
    let size = 0;
    while (true) {
      signal.throwIfAborted();
      const buffer = new Uint8Array(Math.min(64 * 1024, maxBytes - size + 1));
      const { bytesRead } = await handle.read(buffer);
      if (!bytesRead) break;
      size += bytesRead;
      if (size > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
      chunks.push(buffer.subarray(0, bytesRead));
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return bytes;
  } finally { await handle.close(); }
}

function dimensions(width: number, height: number): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 || width * height > maxPixels) throw new RangeError(`Decoded image exceeds pixel limit (${maxPixels})`);
}

export async function readImage(source: ImageInput, options: ImageOptions = {}): Promise<DecodedImage> {
  const maxBytes = options.maxImageBytes ?? 10 * 1024 * 1024;
  const timeoutMs = options.timeoutMs ?? 10_000;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new RangeError('maxImageBytes must be a positive safe integer');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647) throw new RangeError('timeoutMs must be between 1 and 2147483647');
  const signal = AbortSignal.any([AbortSignal.timeout(timeoutMs), ...(options.signal ? [options.signal] : [])]);
  signal.throwIfAborted();
  if (typeof source === 'object' && !(source instanceof Blob) && !(source instanceof URL)) {
    dimensions(source.width, source.height);
    if (!(source.data instanceof Uint8Array || source.data instanceof Uint8ClampedArray) || ![1, 2, 3, 4].includes(source.channels) || source.data.length !== source.width * source.height * source.channels) throw new TypeError('Decoded image has invalid pixel data');
    if (source.data.byteLength > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    const raw = source instanceof RawImage ? source : new RawImage(source.data, source.width, source.height, source.channels);
    const scale = Math.min(1, maxDimension / source.width, maxDimension / source.height);
    const result = await raw.resize(Math.max(1, Math.floor(source.width * scale)), Math.max(1, Math.floor(source.height * scale)));
    signal.throwIfAborted();
    return { data: result.data, width: result.width, height: result.height, channels: source.channels };
  }
  let bytes: Uint8Array<ArrayBuffer>;
  let declared: string | undefined;
  if (source instanceof Blob) {
    if (source.size > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    declared = source.type || undefined;
    bytes = new Uint8Array(await source.arrayBuffer());
  } else if (/^data:/i.test(String(source))) {
    const result = dataBytes(String(source), maxBytes); bytes = result.bytes; declared = result.type;
  } else if (isNode && (source instanceof URL ? source.protocol === 'file:' : !/^[a-z][a-z\d+.-]*:/i.test(source))) {
    bytes = await localBytes(source, maxBytes, signal, options._policy);
  } else {
    const url = new URL(source, typeof location === 'object' ? location.href : undefined);
    if (isNode && url.protocol === 'file:') bytes = await localBytes(url, maxBytes, signal, options._policy);
    else if (url.protocol === 'blob:') {
      const response = await fetch(url, { signal });
      if (!response.ok) { await response.body?.cancel(); throw new Error(`Local image fetch failed with status ${response.status}`); }
      declared = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase() || undefined;
      bytes = await boundedBytes(response, maxBytes, signal);
    }
    else {
      const destination = options._policy ? policyDestination(options._policy, 'image', options._offline, options.validateDestination, signal) : options.validateDestination;
      const result = await fetchLimited(url.href, maxBytes, timeoutMs, signal, destination);
      bytes = result.bytes; declared = result.contentType;
    }
  }
  signal.throwIfAborted();
  const type = rasterType(bytes);
  if (!type || (declared && type !== declared.toLowerCase())) throw new TypeError('Image bytes do not match a supported raster content type');
  if (isNode) {
    // sharp is platform-specific and cannot be loaded by a browser bundle.
    const moduleName = 'sharp';
    const { default: decode }: { default: typeof sharp } = await import(moduleName);
    const decoder = decode(bytes, { limitInputPixels: maxPixels, animated: false });
    const metadata = await decoder.metadata();
    dimensions(metadata.width ?? 0, metadata.height ?? 0);
    const { data, info } = await decoder.rotate().resize({ width: maxDimension, height: maxDimension, fit: 'inside', withoutEnlargement: true }).removeAlpha().toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
    signal.throwIfAborted();
    if (![1, 2, 3, 4].includes(info.channels)) throw new TypeError('Decoder produced unsupported image channels');
    return { data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength), width: info.width, height: info.height, channels: info.channels as DecodedImage['channels'] };
  }
  const bitmap = await createImageBitmap(new Blob([bytes], { type }), { imageOrientation: 'from-image' });
  try {
    dimensions(bitmap.width, bitmap.height);
    const scale = Math.min(1, maxDimension / bitmap.width, maxDimension / bitmap.height);
    const width = Math.max(1, Math.floor(bitmap.width * scale));
    const height = Math.max(1, Math.floor(bitmap.height * scale));
    const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(width, height) : document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
    if (!context) throw new Error('Could not create a 2D canvas context');
    context.drawImage(bitmap, 0, 0, width, height);
    const pixels = context.getImageData(0, 0, width, height);
    signal.throwIfAborted();
    return { data: pixels.data, width, height, channels: 4 };
  } finally { bitmap.close(); }
}

export async function loadImage(image: PageImage, options: ImageOptions = {}): Promise<{ imageId: string; data: Uint8Array; mimeType: 'image/png' }> {
  const decoded = await readImage(image.url, options);
  const raw = new RawImage(decoded.data, decoded.width, decoded.height, decoded.channels);
  const data = isNode ? new Uint8Array(await raw.toSharp().png().toBuffer()) : new Uint8Array(await (await raw.toBlob()).arrayBuffer());
  options.signal?.throwIfAborted();
  return { imageId: image.id, data, mimeType: 'image/png' };
}
