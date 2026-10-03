import { RawImage } from '@huggingface/transformers';
import type * as NodeFs from 'node:fs/promises';
import type * as NodeUrl from 'node:url';
import type { Sharp } from 'sharp';
import type sharpFactory from 'sharp';
import type { PageImage } from '../types.js';
import { boundedBytes, fetchLimited } from './extract.js';
import type { ExtractOptions } from './extract.js';
import { authorizeLocalFile, policyDestination } from './policy.js';
import type { ResourcePolicy } from './policy.js';
import { hashBytes, hashValue } from './source.js';
import { cachedImage, cacheImage } from './image-cache.js';
import type { ImagePreprocessCache } from './image-cache.js';
import { MAX_IMAGE_DIMENSION, MAX_IMAGE_PIXELS, normalizeImageRegion, normalizedImageRegion, tileImageRegions, validateImageDimensions, validateImageTiling } from './image-regions.js';
import type { ImageRegion, ImageTilingOptions, PixelRegion, NormalizedRegion } from './image-regions.js';

export interface DecodedImage { data: Uint8Array | Uint8ClampedArray; width: number; height: number; channels: 1 | 2 | 3 | 4; }
export type ImageInput = string | URL | Blob | DecodedImage;
export interface ImageOptions extends Pick<ExtractOptions, 'signal' | 'maxImageBytes' | 'timeoutMs' | 'validateDestination'> {
  region?: ImageRegion;
  tiling?: ImageTilingOptions;
  maxDimension?: number;
}
export interface InternalImageOptions extends ImageOptions { _policy?: ResourcePolicy | undefined; _offline?: boolean | undefined; }
export interface ImageObservation {
  versionId: string;
  sourceVersionId: string;
  sourceWidth: number;
  sourceHeight: number;
  region: PixelRegion;
  normalizedRegion: NormalizedRegion;
  width: number;
  height: number;
  preprocessing: {
    pipeline: 'sharp-raster' | 'sharp-raw' | 'canvas-raster' | 'canvas-raw';
    orientation: 'from-image' | 'as-provided';
    color: 'srgb-alpha-removed' | 'browser-rgba' | 'as-provided';
    resize: 'none' | 'lanczos3' | 'bilinear' | 'browser-default';
    maxDimension: number;
    inputBytes: number;
    cache: 'hit' | 'miss' | 'disabled';
    reused: 'normalized-pixels' | 'none';
  };
}
export interface PreparedImageRegion { image: DecodedImage; observation: ImageObservation; }

export function validateImageObservation(value: unknown): asserts value is ImageObservation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('Image observation must be an object');
  const observation = value as ImageObservation;
  if (typeof observation.versionId !== 'string' || !/^[a-f0-9]{64}$/.test(observation.versionId) || typeof observation.sourceVersionId !== 'string' || !/^[a-f0-9]{64}$/.test(observation.sourceVersionId)) throw new TypeError('Image observation must have SHA-256 content identities');
  validateImageDimensions(observation.sourceWidth, observation.sourceHeight);
  validateImageDimensions(observation.width, observation.height);
  if (observation.region?.unit !== 'pixels') throw new TypeError('Image observation region must contain source pixel bounds');
  const region = normalizeImageRegion(observation.sourceWidth, observation.sourceHeight, observation.region);
  const normalized = normalizedImageRegion(observation.sourceWidth, observation.sourceHeight, region);
  if (observation.normalizedRegion?.unit !== 'normalized' || normalized.x !== observation.normalizedRegion.x || normalized.y !== observation.normalizedRegion.y || normalized.width !== observation.normalizedRegion.width || normalized.height !== observation.normalizedRegion.height) throw new TypeError('Image observation normalized bounds do not match source pixels');
  const preprocessing = observation.preprocessing;
  if (!preprocessing || typeof preprocessing !== 'object' || Array.isArray(preprocessing) || !['sharp-raster', 'sharp-raw', 'canvas-raster', 'canvas-raw'].includes(preprocessing.pipeline)) throw new TypeError('Image observation preprocessing pipeline is invalid');
  if (!Number.isSafeInteger(preprocessing.maxDimension) || preprocessing.maxDimension < 1 || preprocessing.maxDimension > MAX_IMAGE_DIMENSION || !Number.isSafeInteger(preprocessing.inputBytes) || preprocessing.inputBytes < 1) throw new RangeError('Image observation preprocessing bounds are invalid');
  const scale = Math.min(1, preprocessing.maxDimension / region.width, preprocessing.maxDimension / region.height);
  if (observation.width !== Math.max(1, Math.floor(region.width * scale)) || observation.height !== Math.max(1, Math.floor(region.height * scale))) throw new TypeError('Image observation dimensions do not match its preprocessing');
  const raw = preprocessing.pipeline.endsWith('-raw'); const node = preprocessing.pipeline.startsWith('sharp-');
  const resized = observation.width !== region.width || observation.height !== region.height;
  const resize = !resized ? 'none' : raw ? node ? 'bilinear' : 'browser-default' : node ? 'lanczos3' : 'browser-default';
  if (preprocessing.orientation !== (raw ? 'as-provided' : 'from-image') || preprocessing.color !== (raw ? 'as-provided' : node ? 'srgb-alpha-removed' : 'browser-rgba') || preprocessing.resize !== resize) throw new TypeError('Image observation preprocessing labels are inconsistent');
  if (!['hit', 'miss', 'disabled'].includes(preprocessing.cache) || preprocessing.reused !== (preprocessing.cache === 'hit' ? 'normalized-pixels' : 'none')) throw new TypeError('Image observation cache provenance is invalid');
}
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

async function localBytes(path: string | URL, maxBytes: number, signal: AbortSignal, policy?: ResourcePolicy, enforcePolicy = false): Promise<Uint8Array<ArrayBuffer>> {
  // Node filesystem modules are absent from browser runtimes.
  const protocol = 'node:';
  const fs: typeof NodeFs = await import(`${protocol}fs/promises`);
  const url: typeof NodeUrl = await import(`${protocol}url`);
  const canonical = await fs.realpath(path instanceof URL ? url.fileURLToPath(path) : path);
  if (enforcePolicy || policy !== undefined) await authorizeLocalFile(policy, canonical, signal);
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

async function prepareRegions(source: ImageInput, options: InternalImageOptions, cache?: ImagePreprocessCache): Promise<PreparedImageRegion[]> {
  const maxBytes = options.maxImageBytes ?? 10 * 1024 * 1024;
  const timeoutMs = options.timeoutMs ?? 10_000;
  const maxDimension = options.maxDimension ?? MAX_IMAGE_DIMENSION;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new RangeError('maxImageBytes must be a positive safe integer');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647) throw new RangeError('timeoutMs must be between 1 and 2147483647');
  if (!Number.isSafeInteger(maxDimension) || maxDimension < 1 || maxDimension > MAX_IMAGE_DIMENSION) throw new RangeError(`maxDimension must be between 1 and ${MAX_IMAGE_DIMENSION}`);
  const requestedRegion = options.region === undefined ? undefined : { ...options.region };
  const tiling = options.tiling === undefined ? undefined : { ...options.tiling };
  if (tiling) validateImageTiling(tiling);
  const signal = AbortSignal.any([AbortSignal.timeout(timeoutMs), ...(options.signal ? [options.signal] : [])]);
  signal.throwIfAborted();
  let raw: RawImage | undefined;
  let rawBytes: Uint8Array<ArrayBuffer> | undefined;
  let rawCanvas: OffscreenCanvas | HTMLCanvasElement | undefined;
  let bytes: Uint8Array<ArrayBuffer> | undefined;
  let declared: string | undefined;
  let type: string | undefined;
  let width: number; let height: number;
  let decoder: Sharp | undefined;
  let bitmap: ImageBitmap | undefined;
  let sourceVersionId: string;
  let inputBytes: number;
  const validateDestination = options.validateDestination; const policy = options._policy; const offline = options._offline;
  // Snapshot caller-owned pixels and mutable URLs before any asynchronous work or digest.
  if (typeof source === 'object' && source !== null && !(source instanceof Blob) && !(source instanceof URL)) {
    const { width: sourceWidth, height: sourceHeight, channels, data } = source;
    validateImageDimensions(sourceWidth, sourceHeight);
    if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray) || ![1, 2, 3, 4].includes(channels) || data.length !== sourceWidth * sourceHeight * channels) throw new TypeError('Decoded image has invalid pixel data');
    if (data.byteLength > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    const owned = new Uint8Array(data);
    rawBytes = owned;
    raw = new RawImage(owned, sourceWidth, sourceHeight, channels);
    width = sourceWidth; height = sourceHeight; inputBytes = owned.byteLength;
    sourceVersionId = await hashValue({ kind: 'decoded-pixels', pixels: await hashBytes(owned), width, height, channels });
  } else {
    const input = source instanceof URL ? new URL(source.href) : source;
    if (input instanceof Blob) {
      if (input.size > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
      declared = input.type?.split(';', 1)[0]?.trim().toLowerCase() || undefined;
      bytes = new Uint8Array(await input.arrayBuffer());
    } else if (typeof input !== 'string' && !(input instanceof URL)) throw new TypeError('Image input must be a URL, path, Blob, or decoded pixels');
    else if (/^data:/i.test(String(input))) {
      const result = dataBytes(String(input), maxBytes); bytes = result.bytes; declared = result.type;
    } else if (isNode && (input instanceof URL ? input.protocol === 'file:' : !/^[a-z][a-z\d+.-]+:/i.test(input))) {
      bytes = await localBytes(input, maxBytes, signal, policy, offline !== undefined);
    } else {
      const url = new URL(input, typeof location === 'object' ? location.href : undefined);
      if (isNode && url.protocol === 'file:') bytes = await localBytes(url, maxBytes, signal, policy, offline !== undefined);
      else if (url.protocol === 'blob:') {
        const response = await fetch(url, { signal });
        if (!response.ok) { await response.body?.cancel(); throw new Error(`Local image fetch failed with status ${response.status}`); }
        declared = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase() || undefined;
        bytes = await boundedBytes(response, maxBytes, signal);
      } else {
        const destination = policy !== undefined || offline !== undefined ? policyDestination(policy, 'image', offline, validateDestination, signal) : validateDestination;
        const result = await fetchLimited(url.href, maxBytes, timeoutMs, signal, destination);
        bytes = result.bytes; declared = result.contentType;
      }
    }
    signal.throwIfAborted();
    if (bytes.byteLength > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    type = rasterType(bytes);
    if (!type || (declared && type !== declared.toLowerCase())) throw new TypeError('Image bytes do not match a supported raster content type');
    inputBytes = bytes.byteLength;
    sourceVersionId = await hashValue({ kind: 'raster-bytes', bytes: await hashBytes(bytes), type });
    signal.throwIfAborted();
    if (isNode) {
      // sharp is platform-specific and cannot be loaded by a browser bundle.
      const moduleName = 'sharp';
      const { default: decode }: { default: typeof sharpFactory } = await import(moduleName);
      decoder = decode(bytes, { limitInputPixels: MAX_IMAGE_PIXELS, animated: false });
      try {
        const metadata = await decoder.metadata();
        validateImageDimensions(metadata.width ?? 0, metadata.height ?? 0);
        width = metadata.autoOrient.width; height = metadata.autoOrient.height;
      } catch (error) { decoder.destroy(); throw error; }
    } else {
      bitmap = await createImageBitmap(new Blob([bytes], { type }), { imageOrientation: 'from-image' });
      width = bitmap.width; height = bitmap.height;
    }
  }
  try {
    signal.throwIfAborted();
    validateImageDimensions(width, height);
    const regions = tiling ? tileImageRegions(width, height, tiling, requestedRegion) : [normalizeImageRegion(width, height, requestedRegion)];
    const targets = regions.map((region) => {
      const scale = Math.min(1, maxDimension / region.width, maxDimension / region.height);
      return { width: Math.max(1, Math.floor(region.width * scale)), height: Math.max(1, Math.floor(region.height * scale)) };
    });
    if (targets.reduce((sum, target) => sum + target.width * target.height, 0) > MAX_IMAGE_PIXELS) throw new RangeError(`Prepared image regions exceed pixel limit (${MAX_IMAGE_PIXELS})`);
    const pipeline: ImageObservation['preprocessing']['pipeline'] = isNode ? raw ? 'sharp-raw' : 'sharp-raster' : raw ? 'canvas-raw' : 'canvas-raster';
    const cacheDiagnostics = cache?.diagnostics();
    const cacheEnabled = !!cacheDiagnostics?.maxBytes && !!cacheDiagnostics.maxEntries;
    const prepared: PreparedImageRegion[] = [];
    for (let index = 0; index < regions.length; index++) {
      signal.throwIfAborted();
      const region = regions[index]!; const target = targets[index]!;
      const cacheKey = await hashValue({ version: 1, sourceVersionId, region, target, maxDimension, pipeline });
      signal.throwIfAborted();
      const cached = cacheEnabled ? cachedImage(cache, cacheKey) : undefined;
      let image: DecodedImage;
      let pixelDigest: string;
      if (cached) { image = cached.image; pixelDigest = cached.pixelDigest; }
      else {
        if (raw && isNode) {
          const cropped = await raw.crop([region.x, region.y, region.x + region.width - 1, region.y + region.height - 1]);
          signal.throwIfAborted();
          const result = await cropped.resize(target.width, target.height);
          image = { data: result.data, width: result.width, height: result.height, channels: result.channels };
        } else if (decoder) {
          const { data, info } = await decoder.clone().rotate().extract({ left: region.x, top: region.y, width: region.width, height: region.height }).resize({ width: target.width, height: target.height, fit: 'fill', kernel: 'lanczos3', withoutEnlargement: true }).removeAlpha().toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
          if (![1, 2, 3, 4].includes(info.channels)) throw new TypeError('Decoder produced unsupported image channels');
          image = { data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength), width: info.width, height: info.height, channels: info.channels as DecodedImage['channels'] };
        } else {
          if (raw && !rawCanvas) {
            // Construct one canvas from owned pixels; RawImage's browser path rejects Uint8Array and gray/alpha.
            const pixels = rawBytes!;
            const rgba = raw.channels === 4 ? new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, pixels.byteLength) : new Uint8ClampedArray(width * height * 4);
            if (raw.channels !== 4) for (let source = 0, destination = 0; source < pixels.length; source += raw.channels, destination += 4) {
              rgba[destination] = pixels[source]!;
              rgba[destination + 1] = pixels[source + (raw.channels === 3 ? 1 : 0)]!;
              rgba[destination + 2] = pixels[source + (raw.channels === 3 ? 2 : 0)]!;
              rgba[destination + 3] = raw.channels === 2 ? pixels[source + 1]! : 255;
            }
            rawCanvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(width, height) : document.createElement('canvas');
            rawCanvas.width = width; rawCanvas.height = height;
            const sourceContext = rawCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
            if (!sourceContext) throw new Error('Could not create a source canvas context');
            sourceContext.putImageData(new ImageData(rgba, width, height), 0, 0);
          }
          const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(target.width, target.height) : document.createElement('canvas');
          canvas.width = target.width; canvas.height = target.height;
          const context = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
          if (!context) throw new Error('Could not create a 2D canvas context');
          context.drawImage(rawCanvas ?? bitmap!, region.x, region.y, region.width, region.height, 0, 0, target.width, target.height);
          image = { data: context.getImageData(0, 0, target.width, target.height).data, width: target.width, height: target.height, channels: 4 };
        }
        signal.throwIfAborted();
        validateImageDimensions(image.width, image.height);
        if (image.width !== target.width || image.height !== target.height || image.data.length !== image.width * image.height * image.channels) throw new TypeError('Image preprocessing produced invalid pixel dimensions');
        pixelDigest = await hashBytes(image.data);
        signal.throwIfAborted();
        if (cacheEnabled) cacheImage(cache, cacheKey, image, pixelDigest);
      }
      const versionId = await hashValue({ preprocessing: cacheKey, pixels: pixelDigest, width: image.width, height: image.height, channels: image.channels });
      signal.throwIfAborted();
      const resized = image.width !== region.width || image.height !== region.height;
      const observation: ImageObservation = Object.freeze({
        versionId, sourceVersionId, sourceWidth: width, sourceHeight: height,
        region: Object.freeze(region), normalizedRegion: Object.freeze(normalizedImageRegion(width, height, region)), width: image.width, height: image.height,
        preprocessing: Object.freeze({ pipeline, orientation: raw ? 'as-provided' : 'from-image', color: raw ? 'as-provided' : isNode ? 'srgb-alpha-removed' : 'browser-rgba', resize: !resized ? 'none' : raw ? isNode ? 'bilinear' : 'browser-default' : isNode ? 'lanczos3' : 'browser-default', maxDimension, inputBytes, cache: cached ? 'hit' : cacheEnabled ? 'miss' : 'disabled', reused: cached ? 'normalized-pixels' : 'none' }),
      });
      prepared.push({ image, observation });
    }
    return prepared;
  } finally { bitmap?.close(); decoder?.destroy(); }
}

export function prepareImageRegions(source: ImageInput, options: ImageOptions = {}, cache?: ImagePreprocessCache): Promise<PreparedImageRegion[]> {
  const { signal, maxImageBytes, timeoutMs, validateDestination, region, tiling, maxDimension } = options;
  return prepareRegions(source, { ...(signal === undefined ? {} : { signal }), ...(maxImageBytes === undefined ? {} : { maxImageBytes }), ...(timeoutMs === undefined ? {} : { timeoutMs }), ...(validateDestination === undefined ? {} : { validateDestination }), ...(region === undefined ? {} : { region }), ...(tiling === undefined ? {} : { tiling }), ...(maxDimension === undefined ? {} : { maxDimension }) }, cache);
}
export function prepareImageRegionsInternal(source: ImageInput, options: InternalImageOptions = {}, cache?: ImagePreprocessCache): Promise<PreparedImageRegion[]> {
  return prepareRegions(source, options, cache);
}
export async function prepareImage(source: ImageInput, options: ImageOptions = {}, cache?: ImagePreprocessCache): Promise<PreparedImageRegion> {
  if (options.tiling !== undefined) throw new TypeError('Use prepareImageRegions for tiled image inputs');
  return (await prepareImageRegions(source, options, cache))[0]!;
}
export async function prepareImageInternal(source: ImageInput, options: InternalImageOptions = {}, cache?: ImagePreprocessCache): Promise<PreparedImageRegion> {
  if (options.tiling !== undefined) throw new TypeError('Use prepareImageRegionsInternal for tiled image inputs');
  return (await prepareRegions(source, options, cache))[0]!;
}
export async function readImage(source: ImageInput, options: ImageOptions = {}): Promise<DecodedImage> {
  return (await prepareImage(source, options)).image;
}

export async function loadImage(image: PageImage, options: ImageOptions = {}): Promise<{ imageId: string; data: Uint8Array; mimeType: 'image/png' }> {
  const decoded = await readImage(image.url, options);
  const raw = new RawImage(decoded.data, decoded.width, decoded.height, decoded.channels);
  const data = isNode ? new Uint8Array(await raw.toSharp().png().toBuffer()) : new Uint8Array(await (await raw.toBlob()).arrayBuffer());
  options.signal?.throwIfAborted();
  return { imageId: image.id, data, mimeType: 'image/png' };
}
