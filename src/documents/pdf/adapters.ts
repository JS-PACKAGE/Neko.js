import type { DocumentInput } from '../index.js';
import type { OcrDocument, PdfDocument } from './types.js';

/** Owns the normalized index payload, preserving exact UTF-16 spans and source provenance. */
export function documentForIndex(document: PdfDocument | OcrDocument): DocumentInput {
  return {
    id: document.id,
    text: document.text,
    ...(document.title === undefined ? {} : { title: document.title }),
    ...(document.url === undefined ? {} : { url: document.url }),
    ...(document.blocks === undefined ? {} : { blocks: document.blocks.map((block) => ({
      id: block.id, startOffset: block.startOffset, endOffset: block.endOffset,
      ...(block.source === undefined ? {} : { source: {
        ...block.source,
        ...(block.source.bounds === undefined ? {} : { bounds: [...block.source.bounds] as [number, number, number, number] }),
      } }),
    })) }),
  };
}
