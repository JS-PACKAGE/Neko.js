import type * as PdfModule from 'pdfjs-dist';
import type * as CanvasModule from '@napi-rs/canvas';
import type * as NodeUrl from 'node:url';
import type { PDFPageProxy, TextContent } from 'pdfjs-dist/types/src/display/api.js';
import type { DocumentBlock } from '../index.js';
import type { DecodedImage } from '../../web/image.js';
import { hashBytes } from '../../web/source.js';
import { ocrImage } from '../ocr.js';
import { integer, ownedBytes } from './input.js';
import { nativeLayout } from './layout.js';
import type { OwnedDocumentBytes, PdfDocument, PdfOptions, PdfPage } from './types.js';
export type * from './types.js';
export { documentForIndex } from './adapters.js';

const isNode = typeof process !== 'undefined' && process.release?.name === 'node';
type PdfRuntime = typeof PdfModule;
async function runtime(assetBase?: string | URL): Promise<{ pdf: PdfRuntime; assets: string }> {
  if (isNode) {
    // Platform-specific parser initializes native canvas globals: load only for an explicit PDF request.
    const name = 'pdfjs-dist/legacy/build/pdf.mjs';
    const pdf: PdfRuntime = await import(name);
    const base = assetBase === undefined ? new URL('../../', import.meta.resolve(name)) : new URL(String(assetBase));
    if (base.protocol !== 'file:') throw new TypeError('Node PDF assets must be a local file URL directory');
    const protocol = 'node:'; const { fileURLToPath }: typeof NodeUrl = await import(`${protocol}url`);
    return { pdf, assets: fileURLToPath(base).replace(/[/\\]?$/, '/') };
  }
  const base = assetBase === undefined ? new URL('./assets/pdf/', import.meta.url) : new URL(String(assetBase), import.meta.url);
  if (!['https:', 'http:', 'file:'].includes(base.protocol) || base.username || base.password) throw new TypeError('PDF assetBase must be an application-controlled module directory');
  if (!base.pathname.endsWith('/')) throw new TypeError('PDF assetBase must have a trailing slash');
  const moduleUrl = new URL('pdf.mjs', base);
  const pdf: PdfRuntime = await import(/* @vite-ignore */ moduleUrl.href);
  pdf.GlobalWorkerOptions.workerSrc = new URL('pdf.worker.mjs', base).href;
  return { pdf, assets: base.href };
}

async function boundedText(page: PDFPageProxy, maxItems: number, maxCharacters: number, signal?: AbortSignal): Promise<TextContent> {
  const reader = page.streamTextContent({ includeMarkedContent: true }).getReader();
  const output: TextContent = { items: [], styles: Object.create(null) as TextContent['styles'], lang: null }; let characters = 0;
  const abort = () => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener('abort', abort, { once: true });
  try {
    while (true) {
      signal?.throwIfAborted(); const { value, done } = await reader.read(); signal?.throwIfAborted(); if (done) break;
      if (output.items.length + value.items.length > maxItems) throw new RangeError('PDF text items exceed limit');
      for (const item of value.items) if ('str' in item) characters += item.str.length;
      if (characters > maxCharacters) throw new RangeError('PDF text exceeds character limit');
      for (const item of value.items) output.items.push(item);
      Object.assign(output.styles, value.styles); output.lang = value.lang ?? output.lang;
    }
    return output;
  } finally { signal?.removeEventListener('abort', abort); await reader.cancel().catch(() => undefined); reader.releaseLock(); }
}

async function renderPage(page: PDFPageProxy, pdf: PdfRuntime, scale: number, maxPixels: number, signal?: AbortSignal): Promise<DecodedImage> {
  const viewport = page.getViewport({ scale }); const width = Math.ceil(viewport.width); const height = Math.ceil(viewport.height);
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 || width * height > maxPixels) throw new RangeError('PDF rendered page exceeds pixel limit');
  let canvas: HTMLCanvasElement | CanvasModule.Canvas;
  // The Node native canvas cannot be imported into a browser bundle.
  if (isNode) { const name = '@napi-rs/canvas'; const { createCanvas }: typeof CanvasModule = await import(name); canvas = createCanvas(width, height); }
  else { if (typeof document === 'undefined') throw new Error('PDF raster rendering requires a browser document; compose OCR inference with a worker host from the main thread'); canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height; }
  const context = canvas.getContext('2d'); if (!context) throw new Error('PDF rendering requires a 2D canvas');
  const task = page.render({ canvas: canvas as HTMLCanvasElement, canvasContext: context as CanvasRenderingContext2D, viewport, annotationMode: pdf.AnnotationMode.DISABLE, background: 'rgb(255,255,255)' });
  const abort = () => { task.cancel(); }; signal?.addEventListener('abort', abort, { once: true });
  try {
    signal?.throwIfAborted(); await task.promise; signal?.throwIfAborted();
    const data = context.getImageData(0, 0, width, height).data;
    return { data, width, height, channels: 4 };
  } finally { signal?.removeEventListener('abort', abort); canvas.width = 0; canvas.height = 0; }
}

/** Parses owned PDF data only; scripts, annotations, attachments, links and XFA are never activated. */
export async function extractPdf(source: OwnedDocumentBytes, options: PdfOptions = {}): Promise<PdfDocument> {
  options = { ...options };
  const maxBytes = integer(options.maxBytes, 32 * 1024 * 1024, 256 * 1024 * 1024, 'maxBytes');
  const maxPages = integer(options.maxPages, 100, 1000, 'maxPages');
  const maxPagePixels = integer(options.maxPagePixels, 8_000_000, 40_000_000, 'maxPagePixels');
  const maxTotalPixels = integer(options.maxTotalPixels, 40_000_000, 400_000_000, 'maxTotalPixels');
  const maxItems = integer(options.maxItems, 100_000, 1_000_000, 'maxItems');
  const maxCharacters = integer(options.maxTextCharacters, 2_000_000, 10_000_000, 'maxTextCharacters');
  const maxCells = integer(options.maxCells, 4096, 16_384, 'maxCells');
  const renderScale = options.renderScale ?? 1.5;
  if (!Number.isFinite(renderScale) || renderScale < 0.25 || renderScale > 4) throw new RangeError('renderScale must be between 0.25 and 4');
  const ocr = options.ocr ?? 'none'; if (!['none', 'scanned', 'all'].includes(ocr)) throw new TypeError('Invalid PDF OCR mode');
  if (ocr !== 'none' && typeof options.infer !== 'function') throw new TypeError('PDF OCR requires an inferStructured host');
  const bytes = await ownedBytes(source, maxBytes, options.signal);
  if (!String.fromCharCode(...bytes.subarray(0, Math.min(1024, bytes.length))).includes('%PDF-')) throw new TypeError('Input is not a PDF');
  const sourceVersionId = await hashBytes(bytes); options.signal?.throwIfAborted();
  const { pdf, assets } = await runtime(options.assetBase); options.signal?.throwIfAborted();
  const task = pdf.getDocument({ data: bytes, verbosity: 0, ...(options.password === undefined ? {} : { password: options.password }), cMapUrl: `${assets}cmaps/`, cMapPacked: true, standardFontDataUrl: `${assets}standard_fonts/`, wasmUrl: `${assets}wasm/`, iccUrl: `${assets}iccs/`, useWorkerFetch: !isNode, useSystemFonts: false, disableFontFace: true, enableXfa: false, stopAtErrors: true, disableAutoFetch: true, disableStream: true, disableRange: true, maxImageSize: maxPagePixels, canvasMaxAreaInBytes: maxPagePixels * 4 });
  const abort = () => { void task.destroy().catch(() => undefined); }; options.signal?.addEventListener('abort', abort, { once: true });
  try {
    const document = await task.promise; options.signal?.throwIfAborted();
    if (document.numPages > maxPages) throw new RangeError('PDF exceeds page limit');
    const pages: PdfPage[] = []; const blocks: DocumentBlock[] = []; let text = ''; let itemCount = 0; let characters = 0; let pixels = 0; let cells = 0;
    let usage: PdfDocument['usage'] = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    const warnings = ['PDF text and model output are untrusted. Geometric reading order and untagged table grouping are heuristics, not verified semantic structure.'];
    if (ocr !== 'none') warnings.push('OCR accuracy and model geometry are not verified.');
    for (let number = 1; number <= document.numPages; number++) {
      options.signal?.throwIfAborted(); const page = await document.getPage(number);
      try {
        const viewport = page.getViewport({ scale: 1 });
        if (![viewport.width, viewport.height].every((value) => Number.isFinite(value) && value > 0) || Math.ceil(viewport.width) * Math.ceil(viewport.height) > maxPagePixels) throw new RangeError('PDF page geometry exceeds pixel limit at scale 1');
        const content = await boundedText(page, maxItems - itemCount, maxCharacters - characters, options.signal); const tree = await page.getStructTree(); options.signal?.throwIfAborted();
        const layout = nativeLayout(content, tree, number, viewport.transform, viewport.width, viewport.height, { maxItems: maxItems - itemCount, maxCharacters: maxCharacters - characters, maxCells: maxCells - cells }, options.signal);
        itemCount += layout.itemCount; characters += layout.characters; cells += layout.tables.reduce((sum, table) => sum + table.cells.length, 0);
        const extracted: PdfPage = { pageNumber: number, width: viewport.width, height: viewport.height, rotation: page.rotate, geometryUnit: 'pdf-points', readingOrder: layout.tagged ? 'pdf-tags' : 'geometry-heuristic', blocks: layout.blocks, tables: layout.tables };
        if (ocr === 'all' || (ocr === 'scanned' && !layout.blocks.length)) {
          const renderViewport = page.getViewport({ scale: renderScale }); const count = Math.ceil(renderViewport.width) * Math.ceil(renderViewport.height);
          if (pixels + count > maxTotalPixels) throw new RangeError('PDF rendering exceeds total pixel limit'); pixels += count;
          const image = await renderPage(page, pdf, renderScale, maxPagePixels, options.signal);
          const recognized = await ocrImage(image, options.infer!, { ...options, maxBytes: Math.max(maxBytes, image.data.byteLength), maxPixels: maxPagePixels, maxCells: Math.min(4096, maxCells - cells || 1), maxTextCharacters: Math.min(2_000_000, maxCharacters - characters || 1) });
          const sx = viewport.width / recognized.width; const sy = viewport.height / recognized.height;
          const convert = ([x, y, width, height]: [number, number, number, number]): [number, number, number, number] => [x * sx, y * sy, width * sx, height * sy];
          const blockIds = new Map(recognized.extractedBlocks.map(({ id }) => [id, `p${number}-${id}`]));
          extracted.blocks = recognized.extractedBlocks.map((block) => ({ ...block, id: blockIds.get(block.id)!, bounds: convert(block.bounds) }));
          extracted.tables = recognized.tables.map((table) => ({ ...table, id: `p${number}-${table.id}`, cells: table.cells.map((cell) => ({ ...cell, blockIds: cell.blockIds.map((id) => blockIds.get(id)!), bounds: convert(cell.bounds) })) }));
          extracted.readingOrder = 'model-ocr-untrusted'; extracted.ocr = { width: recognized.width, height: recognized.height, accuracy: 'not-verified', sourceVersionId: recognized.sourceVersionId, usage: recognized.usage, ...(recognized.model === undefined ? {} : { model: recognized.model }), ...(recognized.execution === undefined ? {} : { execution: recognized.execution }) };
          if (usage !== null) { if (recognized.usage === null) usage = null; else { usage.inputTokens += recognized.usage.inputTokens; usage.outputTokens += recognized.usage.outputTokens; usage.totalTokens += recognized.usage.totalTokens; } }
          characters += recognized.text.length; cells += recognized.tables.reduce((sum, table) => sum + table.cells.length, 0);
          if (characters > maxCharacters || cells > maxCells) throw new RangeError('PDF OCR exceeds aggregate text/table limits');
        }
        pages.push(extracted);
        for (const block of extracted.blocks) {
          if (text) text += '\n'; const startOffset = text.length; text += block.text;
          if (text.length > maxCharacters) throw new RangeError('PDF canonical text exceeds character limit');
          blocks.push({ id: block.id, startOffset, endOffset: text.length, source: { kind: block.provenance === 'native-text' ? 'pdf' : 'ocr', pageNumber: number, bounds: block.bounds, geometryUnit: 'pdf-points', provenance: block.provenance } });
        }
      } finally { page.cleanup(); }
    }
    return { id: options.id ?? `pdf:${sourceVersionId}`, ...(options.title === undefined ? {} : { title: options.title }), text, blocks, format: 'pdf', sourceVersionId, pages, warnings, usage };
  } finally { options.signal?.removeEventListener('abort', abort); await task.destroy(); }
}
