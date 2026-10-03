import type { InferenceResult, StructuredInferOptions } from '../../core/engine.js';
import type { DecodedImage } from '../../web/image.js';
import type { DocumentInput } from '../index.js';

export type OwnedDocumentBytes = Uint8Array | ArrayBuffer | Blob;
export type StructuredInference = (options: StructuredInferOptions) => Promise<{ value: unknown } & Partial<Pick<InferenceResult, 'usage' | 'model' | 'execution'>>>;
export interface OcrAccounting {
  /** Null only when an explicitly supplied custom inference host omits usage. */
  usage: InferenceResult['usage'] | null;
  model?: InferenceResult['model'];
  execution?: InferenceResult['execution'];
}
/** Top-left origin, x/y/width/height in the page's declared units. */
export type DocumentBounds = [number, number, number, number];
export interface ExtractedBlock {
  id: string;
  text: string;
  bounds: DocumentBounds;
  order: number;
  provenance: 'native-text' | 'model-ocr-untrusted';
  kind: 'text' | 'heading' | 'table-cell';
  sourceItemIndices?: number[];
  markedContentId?: string;
}
export interface ExtractedTableCell {
  row: number;
  column: number;
  rowSpan: number;
  columnSpan: number;
  header: boolean;
  blockIds: string[];
  bounds: DocumentBounds;
}
export interface ExtractedTable {
  id: string;
  structure: 'pdf-tags' | 'geometry-heuristic' | 'model-ocr-untrusted';
  cells: ExtractedTableCell[];
}
export interface OcrOptions {
  id?: string;
  title?: string;
  signal?: AbortSignal;
  maxBytes?: number;
  maxPixels?: number;
  maxBlocks?: number;
  maxCells?: number;
  maxTextCharacters?: number;
  maxNewTokens?: number;
  contextWindowTokens?: number;
  hardDeadlineMs?: number;
}
export interface OcrDocument extends DocumentInput, OcrAccounting {
  format: 'ocr';
  width: number;
  height: number;
  geometryUnit: 'pixels';
  sourceVersionId: string;
  extractedBlocks: ExtractedBlock[];
  tables: ExtractedTable[];
  provenance: 'model-ocr-untrusted';
  accuracy: 'not-verified';
}
export type OcrImageInput = OwnedDocumentBytes | DecodedImage;
export interface PdfExtractOptions extends OcrOptions {
  maxPages?: number;
  maxPagePixels?: number;
  maxTotalPixels?: number;
  maxItems?: number;
  renderScale?: number;
  ocr?: 'none' | 'scanned' | 'all';
  /** Directory containing pdf.mjs, pdf.worker.mjs, cmaps/, standard_fonts/, wasm/, iccs/. */
  assetBase?: string | URL;
  password?: string;
}
export interface PdfOptions extends PdfExtractOptions { infer?: StructuredInference; }
export interface PdfPage {
  pageNumber: number;
  width: number;
  height: number;
  rotation: number;
  geometryUnit: 'pdf-points';
  readingOrder: 'pdf-tags' | 'geometry-heuristic' | 'model-ocr-untrusted';
  blocks: ExtractedBlock[];
  tables: ExtractedTable[];
  ocr?: { width: number; height: number; accuracy: 'not-verified'; sourceVersionId: string } & OcrAccounting;
}
export interface PdfDocument extends DocumentInput {
  format: 'pdf';
  sourceVersionId: string;
  pages: PdfPage[];
  /** Native-only extraction is zero; null if any OCR host omitted its usage. */
  usage: InferenceResult['usage'] | null;
  warnings: string[];
}
