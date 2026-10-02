import type { Page, PageImage, Paragraph, PageSnapshot, SourceSelection } from '../types.js';
import type * as NodeCrypto from 'node:crypto';
import { awaitUser } from '../errors.js';

function record(value: unknown, name: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${name} must be an object`);
}
function text(value: unknown, name: string): asserts value is string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be non-empty text`);
}
export function validatePage(value: unknown): asserts value is Page {
  record(value, 'Page'); text(value.url, 'Page URL');
  const url = new URL(value.url);
  if (!['http:', 'https:', 'about:'].includes(url.protocol) || (url.protocol === 'about:' && url.href !== 'about:blank')) throw new TypeError('Page URL must be HTTP(S) or about:blank');
  if (value.title !== undefined && typeof value.title !== 'string') throw new TypeError('Page title must be text');
  if (!Array.isArray(value.paragraphs) || !Array.isArray(value.images)) throw new TypeError('Page sources must be arrays');
  const ids = new Set<string>();
  for (const paragraph of value.paragraphs) {
    record(paragraph, 'Paragraph'); text(paragraph.id, 'Paragraph ID'); text(paragraph.text, 'Paragraph text');
    if (ids.has(paragraph.id)) throw new TypeError(`Duplicate source ID ${paragraph.id}`); ids.add(paragraph.id);
    if (paragraph.heading !== undefined && typeof paragraph.heading !== 'string') throw new TypeError('Paragraph heading must be text');
    record(paragraph.source, 'Paragraph source');
    if (paragraph.source.kind !== 'html') throw new TypeError('Paragraph source must be html');
    const { startOffset, endOffset } = paragraph.source;
    if ((startOffset === undefined) !== (endOffset === undefined) || (startOffset !== undefined && (!Number.isSafeInteger(startOffset) || !Number.isSafeInteger(endOffset) || (startOffset as number) < 0 || (endOffset as number) < (startOffset as number)))) throw new TypeError('Paragraph source offsets must be an ordered pair of non-negative integers');
  }
  for (const image of value.images) {
    record(image, 'Image'); text(image.id, 'Image ID'); text(image.url, 'Image URL');
    if (ids.has(image.id)) throw new TypeError(`Duplicate source ID ${image.id}`); ids.add(image.id);
    if (!Array.isArray(image.discoveredBy) || !image.discoveredBy.length || image.discoveredBy.some((kind: unknown) => !['img', 'picture', 'background', 'og:image'].includes(String(kind))) || new Set(image.discoveredBy).size !== image.discoveredBy.length) throw new TypeError('Image discovery sources are invalid');
    for (const key of ['alt', 'caption', 'sourceElement']) if (image[key] !== undefined && typeof image[key] !== 'string') throw new TypeError(`Image ${key} must be text`);
    if (/^[a-z][a-z\d+.-]*:/i.test(image.url) && !/^(https?:|data:image\/(?:png|jpeg|webp|gif|avif|tiff|bmp)[;,])/i.test(image.url)) throw new TypeError('Page image must be HTTP(S), raster data, or an unresolved relative source');
  }
}
function canonical(value: unknown, ancestors: Set<object>): string {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (typeof value !== 'object' || value === null || ancestors.has(value)) throw new TypeError('Snapshot contents must be finite, acyclic JSON data');
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return `[${value.map((item) => canonical(item, ancestors)).join(',')}]`;
    return `{${Object.keys(value).sort().filter((key) => (value as Record<string, unknown>)[key] !== undefined).map((key) => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key], ancestors)}`).join(',')}}`;
  } finally { ancestors.delete(value); }
}
export async function hashBytes(bytes: Uint8Array | Uint8ClampedArray): Promise<string> {
  if (globalThis.crypto?.subtle) {
    const buffer = bytes.buffer instanceof ArrayBuffer ? bytes : new Uint8Array(bytes);
    const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', buffer as Uint8Array<ArrayBuffer>));
    let result = ''; for (const byte of digest) result += byte.toString(16).padStart(2, '0'); return result;
  }
  // Node crypto is platform-specific and cannot be imported by browser bundles.
  const protocol = 'node:';
  const crypto: typeof NodeCrypto = await import(`${protocol}crypto`);
  return crypto.createHash('sha256').update(bytes).digest('hex');
}
export function hashValue(value: unknown): Promise<string> { return hashBytes(new TextEncoder().encode(canonical(value, new Set()))); }

const ownedPages = new WeakSet<Page>();
const snapshots = new WeakMap<Page, Promise<PageSnapshot>>();
function ownPage(input: Page): Page {
  if (ownedPages.has(input)) return input;
  validatePage(input);
  const { url, title } = input;
  const paragraphs: Paragraph[] = input.paragraphs.map(({ id, text, heading, source }) => Object.freeze({ id, text, ...(heading === undefined ? {} : { heading }), source: Object.freeze({ kind: source.kind, ...(source.startOffset === undefined ? {} : { startOffset: source.startOffset, endOffset: source.endOffset }) }) }));
  const images: PageImage[] = input.images.map(({ id, url, alt, caption, discoveredBy, sourceElement }) => {
    const discovery = [...discoveredBy]; Object.freeze(discovery);
    return Object.freeze({ id, url, ...(alt === undefined ? {} : { alt }), ...(caption === undefined ? {} : { caption }), discoveredBy: discovery, ...(sourceElement === undefined ? {} : { sourceElement }) });
  });
  Object.freeze(paragraphs); Object.freeze(images);
  const page: Page = Object.freeze({ url, ...(title === undefined ? {} : { title }), paragraphs, images });
  validatePage(page); ownedPages.add(page); return page;
}
function selectedIds(requested: string[] | undefined, sources: { id: string }[], name: string): Set<string> | undefined {
  if (requested === undefined) return undefined;
  if (!Array.isArray(requested) || requested.some((id) => typeof id !== 'string') || new Set(requested).size !== requested.length) throw new TypeError(`${name} must contain unique source IDs`);
  const available = new Set(sources.map(({ id }) => id));
  for (const id of requested) if (!available.has(id)) throw new TypeError(`Unknown ${name} source ${id}`);
  return new Set(requested);
}
export async function selectPage(input: Page, selection: SourceSelection = {}, signal?: AbortSignal): Promise<Page> {
  const source = ownPage(input);
  const paragraphIds = selectedIds(selection.paragraphIds, source.paragraphs, 'paragraphIds');
  const imageIds = selectedIds(selection.imageIds, source.images, 'imageIds');
  if (selection.paragraph !== undefined && typeof selection.paragraph !== 'function') throw new TypeError('paragraph selector must be a function');
  if (selection.image !== undefined && typeof selection.image !== 'function') throw new TypeError('image selector must be a function');
  const paragraphPredicate = selection.paragraph; const imagePredicate = selection.image;
  if (!paragraphIds && !imageIds && !paragraphPredicate && !imagePredicate) return source;
  const paragraphs: Paragraph[] = []; const images: PageImage[] = [];
  for (const paragraph of source.paragraphs) if ((!paragraphIds || paragraphIds.has(paragraph.id)) && (!paragraphPredicate || await awaitUser(() => paragraphPredicate(paragraph), signal, 'extract'))) paragraphs.push(paragraph);
  for (const image of source.images) if ((!imageIds || imageIds.has(image.id)) && (!imagePredicate || await awaitUser(() => imagePredicate(image), signal, 'extract'))) images.push(image);
  Object.freeze(paragraphs); Object.freeze(images);
  const page: Page = Object.freeze({ url: source.url, ...(source.title === undefined ? {} : { title: source.title }), paragraphs, images });
  ownedPages.add(page); return page;
}
export async function snapshotPage(page: Page): Promise<PageSnapshot> {
  const source = ownPage(page);
  const previous = snapshots.get(source); if (previous) return previous;
  const id = hashValue(source);
  const paragraphs = source.paragraphs.map((paragraph) => { const id = paragraph.id; const digest = hashValue(paragraph); return digest.then((versionId) => Object.freeze({ id, versionId })); });
  const images = source.images.map((image) => { const id = image.id; const digest = hashValue(image); return digest.then((metadataVersionId) => Object.freeze({ id, metadataVersionId })); });
  const snapshot = Promise.all([id, Promise.all(paragraphs), Promise.all(images)]).then((values): PageSnapshot => { Object.freeze(values[1]); Object.freeze(values[2]); return Object.freeze({ id: values[0], algorithm: 'sha256', source, paragraphs: values[1], images: values[2] }); });
  snapshots.set(source, snapshot); return snapshot;
}
