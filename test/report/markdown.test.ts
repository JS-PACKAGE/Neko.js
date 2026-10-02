import assert from 'node:assert/strict';
import test from 'node:test';
import { renderMarkdown, validateStructuredReport } from '../../src/report/index.js';
import type { Page, StructuredReport } from '../../src/types.js';

const page: Page = {
  url: 'https://example.test/article',
  paragraphs: [{ id: 'p1', text: 'Body', source: { kind: 'html', startOffset: 10, endOffset: 20 } }],
  images: [{ id: 'i1', url: 'https://example.test/photo.png', alt: 'A photo', discoveredBy: ['img'] }],
};
const report: StructuredReport = {
  language: 'zh-TW',
  page: { url: page.url, summary: '頁面摘要內容' },
  sections: [{ heading: '內文', keyPoints: ['重點'], paragraphIds: ['p1'] }],
  images: [{ imageId: 'i1', url: page.images[0]!.url, description: 'A described scene.', source: { kind: 'image', imageId: 'i1' }, alt: 'A photo' }],
  conclusion: '結論',
};


test('rejects image descriptions without genuine descriptions or matching source IDs', () => {
  assert.throws(() => renderMarkdown({ ...report, images: [{ ...report.images[0]!, description: '' }] }), /missing its generated description/);
  assert.throws(() => renderMarkdown({ ...report, images: [{ ...report.images[0]!, source: { kind: 'image', imageId: 'other' } }] }), /inconsistent provenance/);
});

test('validates report paragraph and image references against the extracted page', () => {
  assert.doesNotThrow(() => validateStructuredReport(report, page));
  assert.throws(() => validateStructuredReport({ ...report, sections: [{ ...report.sections[0]!, paragraphIds: ['missing'] }] }, page), /unknown paragraph/);
  assert.throws(() => validateStructuredReport({ ...report, images: [] }, page), /every extracted image/);
  assert.throws(() => validateStructuredReport({ ...report, images: [{ ...report.images[0]!, url: 'https://elsewhere.test/photo.png' }] }, page), /inconsistent provenance/);
});
