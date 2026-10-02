import assert from 'node:assert/strict';
import test from 'node:test';
import { renderMarkdown, validateStructuredReport } from '../../src/report/index.js';
import { snapshotPage } from '../../src/web/source.js';
import { getModelProfile } from '../../src/cache/manifest.js';
import { NekoError } from '../../src/errors.js';
import type { Page, StructuredReport } from '../../src/types.js';

async function fixture(language = 'en', imageUrl = 'https://example.test/photo.png'): Promise<StructuredReport> {
  const page: Page = { url: 'about:blank', title: '<Source title>', paragraphs: [{ id: 'p1', text: 'The sky is blue.', source: { kind: 'html', startOffset: 10, endOffset: 26 } }, { id: 'p2', text: 'Unquoted navigation.', source: { kind: 'html' } }], images: [{ id: 'i1', url: imageUrl, alt: '<img src=x onerror=alert(1)>', discoveredBy: ['img'] }] };
  const snapshot = await snapshotPage(page);
  const profile = getModelProfile('all-q4');
  const text = language === 'zh-TW' ? '天空呈現藍色。' : 'The sky is blue.';
  const description = language === 'zh-TW' ? '圖片呈現紅色方形。' : 'A red square.';
  const quote = { kind: 'quote' as const, snapshotId: snapshot.id, paragraphId: 'p1', versionId: snapshot.paragraphs[0]!.versionId, startOffset: 0, endOffset: 16, quote: 'The sky is blue.' };
  const versionId = 'a'.repeat(64);
  return {
    language, imageFailurePolicy: 'error', page: { url: page.url, title: page.title!, summary: text }, sections: [{ keyPoints: [text], paragraphIds: ['p1', 'p2'] }],
    images: [{ status: 'described', imageId: 'i1', url: imageUrl, alt: page.images[0]!.alt!, source: { kind: 'image', imageId: 'i1' }, description, observation: { versionId, width: 32, height: 32, verification: 'model-observation' } }],
    conclusion: text, snapshot,
    claims: [
      ...['page.summary', 'sections[0].keyPoints[0]', 'conclusion'].map((target, index) => ({ id: `c${index}`, target, startOffset: 0, endOffset: text.length, citations: [quote], verification: 'references-validated' as const })),
      { id: 'image', target: 'images[0].description', startOffset: 0, endOffset: description.length, citations: [{ kind: 'image-observation', snapshotId: snapshot.id, imageId: 'i1', versionId }], verification: 'references-validated' },
    ],
    metadata: { model: { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: { ...profile.dtype } }, backend: { runtime: 'node', device: 'cpu', executionProviders: ['cpu'], gpuMemoryBytes: null, adapterEvidence: null, supported: true, capabilityEvidence: 'model-runtime-compatibility', providerEvidence: 'loaded-session-configuration', sessions: Object.entries(profile.dtype).map(([name, dtype]) => ({ name, dtype, device: 'cpu' })) }, usage: { inputTokens: 100, outputTokens: 10, totalTokens: 110 }, timings: { loadMs: 1, preprocessMs: 2, generationMs: 3, totalMs: 6, queueWaitMs: 0 }, memory: { jsHeapBytes: null, gpuBytes: null }, execution: { mode: 'inline', runtime: 'node' }, resumedStages: 0, evidence: 'references-validated-not-fact-checked' },
  };
}

test('persisted inline reports audit uncited source text without refetching', async () => {
  const report = JSON.parse(JSON.stringify(await fixture())) as StructuredReport;
  await validateStructuredReport(report);
  report.snapshot.source.paragraphs[1]!.text = 'Replaced uncited source';
  await assert.rejects(validateStructuredReport(report), TypeError);
});

test('quote citations reject altered text, offsets, versions and external source changes', async () => {
  const report = await fixture();
  for (const change of [{ quote: 'The sky is red.' }, { startOffset: 1 }, { versionId: 'b'.repeat(64) }]) {
    const changed = structuredClone(report); Object.assign(changed.claims[0]!.citations[0]!, change);
    await assert.rejects(validateStructuredReport(changed), TypeError);
  }
  const changedSource = structuredClone(report.snapshot.source); changedSource.paragraphs[0]!.text = 'The sky is red.';
  await assert.rejects(validateStructuredReport(report, changedSource), TypeError);
});

test('atomic spans reject unsupported generated suffixes, overlapping claims and wrong image versions', async () => {
  const report = await fixture();
  const unsupported = structuredClone(report); unsupported.page.summary += ' A fabricated detail.';
  await assert.rejects(validateStructuredReport(unsupported), TypeError);
  const overlap = structuredClone(report); overlap.claims.push({ ...overlap.claims[0]!, id: 'overlap' });
  await assert.rejects(validateStructuredReport(overlap), TypeError);
  const wrongImage = structuredClone(report); const citation = wrongImage.claims[3]!.citations[0]!; assert.ok(citation.kind === 'image-observation'); citation.versionId = 'b'.repeat(64);
  await assert.rejects(validateStructuredReport(wrongImage), TypeError);
});

test('selected paragraph/image coverage and loaded profile evidence cannot silently drift', async () => {
  const report = await fixture();
  const missing = structuredClone(report); missing.sections[0]!.paragraphIds = ['p1'];
  await assert.rejects(validateStructuredReport(missing), TypeError);
  const altered = structuredClone(report); altered.images[0]!.url = 'https://elsewhere.test/image.png';
  await assert.rejects(validateStructuredReport(altered), TypeError);
  const wrongProfile = structuredClone(report); wrongProfile.metadata.backend.sessions.find(({ name }) => name === 'vision_encoder')!.dtype = 'fp16';
  await assert.rejects(validateStructuredReport(wrongProfile), TypeError);
});

test('language checks cover every generated field while source metadata remains unchanged', async () => {
  const report = await fixture('zh-TW');
  await validateStructuredReport(report);
  const variants = [structuredClone(report), structuredClone(report), structuredClone(report), structuredClone(report)];
  variants[0]!.page.summary = 'An English summary.'; variants[1]!.sections[0]!.keyPoints[0] = 'An English key point.';
  const image = variants[2]!.images[0]!; assert.ok(image.status === 'described'); image.description = 'An English image description.';
  variants[3]!.conclusion = '这个图片显示红色方形。';
  for (const invalid of variants) await assert.rejects(validateStructuredReport(invalid), (error: unknown) => error instanceof NekoError && error.code === 'LANGUAGE_MISMATCH');
});

test('failed-image reports preserve errors but cannot claim observations or success under error policy', async () => {
  const report = await fixture(); report.imageFailurePolicy = 'omit';
  report.images = [{ status: 'failed', imageId: 'i1', url: report.images[0]!.url, alt: report.images[0]!.alt!, source: { kind: 'image', imageId: 'i1' }, error: { stage: 'image', code: 'OPERATION_FAILED', message: '<decoder failed>' } }];
  report.claims = report.claims.filter(({ target }) => target !== 'images[0].description');
  await validateStructuredReport(report);
  assert.ok(renderMarkdown(report).includes('&lt;decoder failed&gt;'));
  await assert.rejects(validateStructuredReport({ ...report, imageFailurePolicy: 'error' }), TypeError);
  await assert.rejects(validateStructuredReport({ ...report, images: [{ ...report.images[0]!, description: 'A fake success.' }] }), TypeError);
});

test('Markdown escapes source HTML and refuses active-scheme provenance links', async () => {
  const report = await fixture(); await validateStructuredReport(report); report.images[0]!.url = 'javascript:alert(1)';
  const markdown = renderMarkdown(report);
  assert.ok(markdown.includes('&lt;Source title&gt;'));
  assert.ok(!markdown.includes('<img'));
  assert.doesNotMatch(markdown, /\]\(javascript:/i);
});
