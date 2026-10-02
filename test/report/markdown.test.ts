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
  imageFailurePolicy: 'error',
  page: { url: page.url, summary: '頁面摘要內容' },
  sections: [{ heading: '內文', keyPoints: ['重點'], paragraphIds: ['p1'] }],
  images: [{ status: 'described', imageId: 'i1', url: page.images[0]!.url, description: '圖片呈現街道景色。', source: { kind: 'image', imageId: 'i1' }, alt: 'A photo' }],
  conclusion: '結論',
};


test('rejects image descriptions without genuine descriptions or matching source IDs', () => {
  const image = report.images[0]!; assert.ok(image.status === 'described');
  assert.throws(() => renderMarkdown({ ...report, images: [{ ...image, description: '' }] }), /missing its generated description/);
  assert.throws(() => renderMarkdown({ ...report, images: [{ ...report.images[0]!, source: { kind: 'image', imageId: 'other' } }] }), /inconsistent provenance/);
});

test('validates report paragraph and image references against the extracted page', () => {
  assert.doesNotThrow(() => validateStructuredReport(report, page));
  assert.throws(() => validateStructuredReport({ ...report, sections: [{ ...report.sections[0]!, paragraphIds: ['missing'] }] }, page), /unknown paragraph/);
  assert.throws(() => validateStructuredReport({ ...report, images: [] }, page), /every extracted image/);
  assert.throws(() => validateStructuredReport({ ...report, images: [{ ...report.images[0]!, url: 'https://elsewhere.test/photo.png' }] }, page), /inconsistent provenance/);
  assert.throws(() => validateStructuredReport({ ...report, sections: [] }, page), /every extracted paragraph/);
  assert.throws(() => validateStructuredReport({ ...report, page: { ...report.page, summary: 42 } }, page), /non-empty string/);
  assert.throws(() => validateStructuredReport({ ...report, sections: [{ paragraphIds: ['p1'], keyPoints: [] }] }, page), /non-empty string array/);
});

test('rejects mismatched generated language in every report field, not source metadata', () => {
  for (const invalid of [
    { ...report, page: { ...report.page, summary: 'A complete English summary.' } },
    { ...report, sections: [{ ...report.sections[0]!, keyPoints: ['The exhibit displays a red square.'] }] },
    { ...report, images: [{ ...report.images[0]!, description: 'A red square.' }] },
    { ...report, conclusion: 'An English conclusion.' },
    { ...report, conclusion: '这个图片显示红色方形。' },
  ]) assert.throws(() => validateStructuredReport(invalid, page), (error: unknown) => typeof error === 'object' && error !== null && 'code' in error && error.code === 'LANGUAGE_MISMATCH');
  const english: StructuredReport = { ...report, language: 'en', page: { ...report.page, summary: 'A street scene.' }, sections: [{ keyPoints: ['People walk along a street.'], paragraphIds: ['p1'] }], images: [{ status: 'described', imageId: 'i1', url: page.images[0]!.url, description: 'A street with people.', source: { kind: 'image', imageId: 'i1' }, alt: 'A photo' }], conclusion: 'The scene depicts everyday street life.' };
  assert.throws(() => validateStructuredReport({ ...english, conclusion: '圖片呈現街道景色。' }, page), /English language check/);
});

test('partial reports retain failed-image provenance without claiming a description', () => {
  const partial: StructuredReport = { ...report, imageFailurePolicy: 'omit', images: [{ status: 'failed', imageId: 'i1', url: page.images[0]!.url, alt: 'A photo', source: { kind: 'image', imageId: 'i1' }, error: { stage: 'image', code: 'OPERATION_FAILED', message: '<decoder failed>' } }] };
  validateStructuredReport(partial, page);
  const markdown = renderMarkdown(partial);
  assert.match(markdown, /i1/);
  assert.match(markdown, /&lt;decoder failed&gt;/);
  assert.throws(() => validateStructuredReport({ ...partial, imageFailurePolicy: 'error' }, page), /failure policy/);
  assert.throws(() => validateStructuredReport({ ...partial, images: [{ ...partial.images[0]!, description: 'fake success' }] }, page), /cannot claim/);
});
