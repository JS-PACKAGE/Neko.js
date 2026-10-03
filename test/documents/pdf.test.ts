import assert from 'node:assert/strict';
import test from 'node:test';
import { extractPdf, documentForIndex } from '../../src/documents/pdf/index.js';
import { createDocumentIndex } from '../../src/documents/index.js';
import { ocrImage } from '../../src/documents/ocr.js';
import { nativePdfFixture, rasterOcrFixture, scannedPdfFixture } from './pdf-fixtures.js';

// Callback assertions cover the public inference boundary, not OCR/model accuracy.
test('native multipage PDF extracts geometry, table cells, exact offsets without inference', async () => {
  const input = nativePdfFixture(); const size = input.byteLength;
  const document = await extractPdf(input, { infer: async () => { throw new Error('Native extraction must not infer'); } });
  assert.equal(input.byteLength, size);
  assert.equal(document.pages.length, 2);
  assert.match(document.text, /Inventory/); assert.match(document.text, /Second page/);
  assert.deepEqual(document.usage, { inputTokens: 0, outputTokens: 0, totalTokens: 0 });
  assert.equal(document.pages[0]!.tables[0]!.structure, 'geometry-heuristic');
  assert.equal(document.pages[0]!.tables[0]!.cells.length, 4);
  const normalized = documentForIndex(document);
  await createDocumentIndex([normalized]);
  assert.equal(Object.hasOwn(normalized, 'pages'), false);
  for (const block of document.blocks!) {
    assert.equal(block.source!.provenance, 'native-text'); assert.equal(block.source!.geometryUnit, 'pdf-points');
    const source = document.pages[block.source!.pageNumber! - 1]!.blocks.find(({ id }) => id === block.id)!;
    assert.equal(document.text.slice(block.startOffset, block.endOffset), source.text);
    assert.ok(source.bounds[0] >= 0 && source.bounds[1] >= 0 && source.bounds[2] > 0 && source.bounds[3] > 0);
  }
});

test('PDF enforces bytes, pages, geometry and cancellation', async () => {
  await assert.rejects(extractPdf(nativePdfFixture(), { maxBytes: 16 }), /byte bounds/);
  await assert.rejects(extractPdf(nativePdfFixture(), { maxPages: 1 }), /page limit/);
  await assert.rejects(extractPdf(nativePdfFixture(), { maxPagePixels: 100 }), /geometry/);
  await assert.rejects(extractPdf(nativePdfFixture(), { signal: AbortSignal.abort() }), { name: 'AbortError' });
  await assert.rejects(extractPdf(nativePdfFixture(), { ocr: 'scanned' }), /inferStructured/);
});

test('real scanned PDF renders owned pixels and marks structured OCR provenance and usage', async () => {
  let calls = 0;
  const document = await extractPdf(scannedPdfFixture(), { ocr: 'scanned', infer: async (options) => {
    calls++; assert.ok(options.image && typeof options.image === 'object' && 'data' in options.image);
    assert.ok(options.image.data.some((value) => value < 32), 'rendered scan contains dark glyphs');
    return { value: { blocks: [{ text: 'Invoice 1042', kind: 'text', x: 20, y: 20, width: 200, height: 40, table: -1, row: -1, column: -1, rowSpan: 1, columnSpan: 1, header: false }] }, usage: { inputTokens: 100, outputTokens: 40, totalTokens: 140 } };
  } });
  assert.equal(calls, 1); assert.equal(document.pages[0]!.readingOrder, 'model-ocr-untrusted');
  assert.equal(document.blocks![0]!.source!.kind, 'ocr'); assert.equal(document.blocks![0]!.source!.provenance, 'model-ocr-untrusted');
  assert.equal(document.pages[0]!.ocr!.accuracy, 'not-verified'); assert.equal(document.usage!.totalTokens, 140);
});

test('standalone raster OCR rejects model geometry and overlapping cell coordinates', async () => {
  const { png } = rasterOcrFixture();
  const block = { text: 'Apples', kind: 'table-cell', x: 10, y: 10, width: 80, height: 30, table: 0, row: 0, column: 0, rowSpan: 1, columnSpan: 1, header: false };
  const output = await ocrImage(png, async () => ({ value: { blocks: [block] } }));
  assert.equal(output.provenance, 'model-ocr-untrusted'); assert.equal(output.usage, null); assert.equal(output.tables[0]!.cells.length, 1);
  await createDocumentIndex([documentForIndex(output)]);
  await assert.rejects(ocrImage(png, async () => ({ value: { blocks: [{ ...block, x: 9999 }] } })), { code: 'STRUCTURED_OUTPUT' });
  await assert.rejects(ocrImage(png, async () => ({ value: { blocks: [block, { ...block, text: '7' }] } })), { code: 'STRUCTURED_OUTPUT' });
  await assert.rejects(ocrImage(png, async () => ({ value: { blocks: [{ ...block, kind: 'text' }] } })), { code: 'STRUCTURED_OUTPUT' });
});
