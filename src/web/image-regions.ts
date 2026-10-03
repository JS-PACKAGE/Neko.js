export interface PixelRegion { unit: 'pixels'; x: number; y: number; width: number; height: number; }
export interface NormalizedRegion { unit: 'normalized'; x: number; y: number; width: number; height: number; }
export type ImageRegion = PixelRegion | NormalizedRegion;
export interface ImageTilingOptions { tileWidth: number; tileHeight: number; overlap?: number; maxTiles?: number; }
export const MAX_IMAGE_PIXELS = 40_000_000;
export const MAX_IMAGE_DIMENSION = 1280;
export const MAX_IMAGE_TILES = 64;

export function validateImageDimensions(width: number, height: number): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 || width * height > MAX_IMAGE_PIXELS) throw new RangeError(`Decoded image exceeds pixel limit (${MAX_IMAGE_PIXELS})`);
}

export function normalizeImageRegion(width: number, height: number, region?: ImageRegion): PixelRegion {
  validateImageDimensions(width, height);
  if (region === undefined) return { unit: 'pixels', x: 0, y: 0, width, height };
  if (!region || !['pixels', 'normalized'].includes(region.unit)) throw new TypeError('Image region unit must be pixels or normalized');
  const { x, y, width: cropWidth, height: cropHeight } = region;
  if (![x, y, cropWidth, cropHeight].every(Number.isFinite) || x < 0 || y < 0 || cropWidth <= 0 || cropHeight <= 0) throw new RangeError('Image region must have finite non-negative offsets and positive dimensions');
  if (region.unit === 'pixels') {
    if (![x, y, cropWidth, cropHeight].every(Number.isSafeInteger) || x + cropWidth > width || y + cropHeight > height) throw new RangeError('Pixel image region must contain integers within the oriented source dimensions');
    return { unit: 'pixels', x, y, width: cropWidth, height: cropHeight };
  }
  const epsilon = Number.EPSILON * 4;
  if (x >= 1 || y >= 1 || x + cropWidth > 1 + epsilon || y + cropHeight > 1 + epsilon) throw new RangeError('Normalized image region must be within [0, 1]');
  // Include every intersected source pixel; provenance records these actual integer bounds.
  const left = Math.floor(x * width); const top = Math.floor(y * height);
  const right = Math.min(width, Math.ceil(Math.min(1, x + cropWidth) * width));
  const bottom = Math.min(height, Math.ceil(Math.min(1, y + cropHeight) * height));
  return { unit: 'pixels', x: left, y: top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) };
}

export function normalizedImageRegion(width: number, height: number, region: PixelRegion): NormalizedRegion {
  const pixels = normalizeImageRegion(width, height, region);
  return { unit: 'normalized', x: pixels.x / width, y: pixels.y / height, width: pixels.width / width, height: pixels.height / height };
}

export function validateImageTiling(options: ImageTilingOptions): void {
  if (!options || !Number.isSafeInteger(options.tileWidth) || !Number.isSafeInteger(options.tileHeight) || options.tileWidth < 1 || options.tileHeight < 1 || options.tileWidth > MAX_IMAGE_PIXELS || options.tileHeight > MAX_IMAGE_PIXELS) throw new RangeError('Image tile dimensions must be positive bounded safe integers');
  const overlap = options.overlap ?? 0.15; const maxTiles = options.maxTiles ?? 16;
  if (!Number.isFinite(overlap) || overlap < 0 || overlap > 0.5) throw new RangeError('Image tile overlap must be between 0 and 0.5');
  if (!Number.isSafeInteger(maxTiles) || maxTiles < 1 || maxTiles > MAX_IMAGE_TILES) throw new RangeError(`maxTiles must be between 1 and ${MAX_IMAGE_TILES}`);
}

export function tileImageRegions(width: number, height: number, options: ImageTilingOptions, region?: ImageRegion): PixelRegion[] {
  const bounds = normalizeImageRegion(width, height, region);
  validateImageTiling(options);
  const tileWidth = Math.min(options.tileWidth, bounds.width); const tileHeight = Math.min(options.tileHeight, bounds.height);
  const overlap = options.overlap ?? 0.15;
  const stepX = Math.max(1, Math.ceil(tileWidth * (1 - overlap))); const stepY = Math.max(1, Math.ceil(tileHeight * (1 - overlap)));
  const columns = 1 + Math.ceil((bounds.width - tileWidth) / stepX); const rows = 1 + Math.ceil((bounds.height - tileHeight) / stepY);
  const maxTiles = options.maxTiles ?? 16;
  if (columns * rows > maxTiles) throw new RangeError(`Image tiling requires ${columns * rows} regions, exceeding maxTiles (${maxTiles})`);
  const regions: PixelRegion[] = [];
  // Edge tiles shrink rather than shifting backwards and exceeding the configured overlap.
  for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
    const x = column * stepX; const y = row * stepY;
    regions.push({ unit: 'pixels', x: bounds.x + x, y: bounds.y + y, width: Math.min(tileWidth, bounds.width - x), height: Math.min(tileHeight, bounds.height - y) });
  }
  return regions;
}
