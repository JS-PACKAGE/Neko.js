import assert from 'node:assert/strict';
import test from 'node:test';
import { askDocuments, createDocumentIndex, documentFromPage, importDocumentIndex, type DocumentInput, type DocumentQueryHost } from '../../src/documents/index.js';
import type { InferencePlan, InferencePlanOptions, StructuredInferOptions } from '../../src/core/engine.js';
import { extractPage } from '../../src/web/extract.js';
import { getRegisteredModelProfile } from '../../src/cache/registry.js';
import type { DocumentIndexSnapshot } from '../../src/documents/index.js';

const registered = getRegisteredModelProfile();
const model: InferencePlan['model'] = { id: registered.id, revision: registered.revision, profile: registered.profile, dtype: registered.dtype };
// Public host seam: deterministic fixtures validate retrieval/ownership, not real model quality.
function plan(options: InferencePlanOptions, limit = 4096): InferencePlan {
  const inputTokens = Math.ceil((options.prompt?.length ?? 0) / 4); const maxNewTokens = options.maxNewTokens ?? 512;
  return { inputTokens, maxNewTokens, contextLimit: limit, availableOutputTokens: Math.max(0, limit - inputTokens), fits: inputTokens + maxNewTokens <= limit, model };
}
function evidence(options: StructuredInferOptions): { id: string; documentId: string; text: string }[] {
  const payload: { chunks: { id: string; documentId: string; text: string }[] } = JSON.parse(options.prompt!.slice(options.prompt!.indexOf('\n') + 1));
  return payload.chunks;
}

test('long-document BM25 retrieves relevant late paragraphs with exact UTF-16 provenance', async () => {
  const prefix = Array.from({ length: 2000 }, (_, n) => `Ordinary garden record ${n}: the grass grows near the fence.\n\n`).join('');
  const target = '🌌 The violet comet telemetry arrived at 03:17.';
  const index = await createDocumentIndex([{ id: 'long', text: prefix + target }], { chunkSize: 128 });
  const result = index.search('violet comet telemetry', { topK: 2 });
  assert.equal(result.hits[0]!.chunk.documentId, 'long');
  assert.ok(result.hits[0]!.chunk.text.includes('violet comet telemetry'));
  const quote = index.quote(result.hits[0]!.chunk.id, 'violet comet telemetry');
  assert.equal(quote.startOffset, (prefix + target).indexOf('violet comet telemetry'));
  assert.equal((prefix + target).slice(quote.startOffset, quote.endOffset), quote.quote);
  assert.equal(index.validateCitation(quote), true);
  assert.equal(result.coverage.exhaustive, false);
});

test('multi-document indexes and persistent reconstruction are deterministic including CJK retrieval', async () => {
  const documents: DocumentInput[] = [{ id: 'z', text: '庭園には四本の樫の木があります。' }, { id: 'a', text: '銀河観測の開始日は十月三日です。' }];
  const one = await createDocumentIndex(documents, { chunkSize: 64 });
  const two = await createDocumentIndex([...documents].reverse(), { chunkSize: 64 });
  assert.deepEqual(one.exportSnapshot(), two.exportSnapshot());
  assert.deepEqual(one.search('銀河観測'), two.search('銀河観測'));
  assert.equal(one.search('銀河観測').hits[0]!.chunk.documentId, 'a');
  const restored = await importDocumentIndex(JSON.parse(JSON.stringify(one.exportSnapshot())));
  assert.deepEqual(restored.search('銀河観測'), one.search('銀河観測'));
  assert.equal(restored.validateCitation(one.search('銀河観測').hits[0]!.citation), true);
  assert.deepEqual(one.search('樫の木', { documentIds: ['a'] }).hits.filter(({ chunk }) => chunk.documentId === 'z'), []);
});

test('source, version, format and limit corruption are rejected on index import', async () => {
  const index = await createDocumentIndex([{ id: 'x', text: 'Original evidence.' }]);
  type MutableSnapshot = Omit<DocumentIndexSnapshot, 'version'> & { version: number; postings?: unknown };
  const change = (mutate: (snapshot: MutableSnapshot) => void) => {
    const snapshot: MutableSnapshot = JSON.parse(JSON.stringify(index.exportSnapshot())); mutate(snapshot); return snapshot;
  };
  await assert.rejects(importDocumentIndex(change((snapshot) => { snapshot.documents[0]!.text = 'Changed evidence.'; })), /integrity/);
  await assert.rejects(importDocumentIndex(change((snapshot) => { snapshot.documents[0]!.versionId = '0'.repeat(64); })), /integrity/);
  await assert.rejects(importDocumentIndex(change((snapshot) => { snapshot.version = 2; })), /Unsupported/);
  await assert.rejects(importDocumentIndex(change((snapshot) => { snapshot.revision = 1; })), /integrity/);
  await assert.rejects(importDocumentIndex(change((snapshot) => { snapshot.options.maxChunks = 1_000_000; })), /maxChunks/);
  await assert.rejects(importDocumentIndex(change((snapshot) => { snapshot.postings = {}; })), /unknown field/);
});

test('update, replace and remove invalidate stale citations and queued updates are atomic', async () => {
  const input = { id: 'x', text: 'Alpha observation.' };
  const creating = createDocumentIndex([input]); input.text = 'Caller mutation must not enter the index.';
  const index = await creating;
  assert.equal(index.getDocument('x')!.text, 'Alpha observation.');
  const citation = index.search('Alpha').hits[0]!.citation;
  await Promise.all([index.updateDocument({ id: 'x', text: 'Beta observation.' }), index.updateDocument({ id: 'y', text: 'Gamma observation.' })]);
  assert.equal(index.revision, 2); assert.equal(index.documentCount, 2); assert.equal(index.validateCitation(citation), false);
  const beta = index.search('Beta').hits[0]!.citation;
  await index.removeDocument('y'); assert.equal(index.validateCitation(beta), false);
  const before = index.id;
  await assert.rejects(index.replaceDocuments([{ id: 'same', text: 'One' }, { id: 'same', text: 'Two' }]), /Duplicate/);
  assert.equal(index.id, before);
  await index.replaceDocuments([{ id: 'only', text: 'Delta observation.' }]);
  assert.equal(index.documentCount, 1); assert.equal(index.getDocument('x'), undefined);
  assert.equal(index.search('Alpha').hits.length, 0);
});

test('candidate limits are explicit and rare terms keep late matches eligible', async () => {
  const documents = Array.from({ length: 20 }, (_, n) => ({ id: `${n}`.padStart(2, '0'), text: 'Common observation.' }));
  documents.push({ id: 'last', text: 'Common zirconium observation.' });
  const index = await createDocumentIndex(documents);
  const result = index.search('common zirconium', { maxScoredChunks: 1, topK: 1 });
  assert.equal(result.hits[0]!.chunk.documentId, 'last');
  assert.equal(result.coverage.scoredChunks, 1); assert.equal(result.coverage.candidateLimitReached, true); assert.equal(result.coverage.exhaustive, false);
  assert.throws(() => index.search('common', { topK: 101 }), /topK/);
  assert.throws(() => index.search('common', { documentIds: ['missing'] }), /known/);
});

test('block provenance and page normalization preserve exact source text and astral boundaries', async () => {
  const page = await extractPage('<p>🌳 Orchard evidence.</p><p>Late paragraph evidence.</p>');
  const input = documentFromPage(page, 'page'); const index = await createDocumentIndex([input], { chunkSize: 64 });
  const hit = index.search('Orchard').hits[0]!;
  assert.equal(hit.chunk.source?.kind, 'page'); assert.equal(hit.chunk.source?.paragraphId, page.paragraphs[0]!.id);
  const quote = index.quote(hit.chunk.id, '🌳'); assert.equal(index.validateCitation(quote), true);
  assert.throws(() => index.quote(hit.chunk.id, '\ud83c'), /substring/);
  assert.equal(index.validateCitation({ ...quote, quote: '\ud83c', endOffset: quote.startOffset + 1 }), false);
  assert.equal(index.validateCitation({ ...quote, source: { kind: 'ocr' } }), false);
  await assert.rejects(createDocumentIndex([{ id: 'bad', text: '🌳', blocks: [{ id: 'half', startOffset: 0, endOffset: 1 }] }]), /surrogate/);
});

test('askDocuments selects bounded chunks with planning, exact evidence, streaming and actual usage', async () => {
  const index = await createDocumentIndex([{ id: 'a', text: 'The violet comet telemetry arrived at 03:17.' }, { id: 'b', text: 'Violet comet history. '.repeat(20) }], { chunkSize: 128 });
  const seen: StructuredInferOptions[] = []; const tokens: string[] = []; let planned = 0;
  let contextLimit: number | undefined;
  const usage = { inputTokens: 345, outputTokens: 22, totalTokens: 367 };
  const host: DocumentQueryHost = {
    async planInference(options) { planned++; const candidate = plan(options); contextLimit ??= candidate.inputTokens + candidate.maxNewTokens; return plan(options, contextLimit); },
    async inferStructured(options) {
      seen.push(options); options.onToken?.('{');
      const chunks = evidence(options); const found = chunks.find(({ text }) => text.includes('03:17'))!;
      assert.ok(found); return { value: { status: 'answered', claims: [{ text: '03:17', chunkIds: [found.id] }] }, usage, model };
    },
  };
  const answer = await askDocuments(host, index, 'When did violet comet telemetry arrive?', { maxNewTokens: 32, onToken: (text) => tokens.push(text), search: { topK: 4 } });
  assert.equal(answer.status, 'answered'); assert.equal(answer.answer, '03:17'); assert.deepEqual(answer.usage, usage);
  assert.equal(seen.length, 1); assert.deepEqual(tokens, ['{']); assert.equal(answer.retrieval.planningCalls, planned);
  assert.ok(answer.retrieval.plan!.inputTokens + 32 <= answer.retrieval.plan!.contextLimit);
  assert.equal(answer.retrieval.contextLimited, true); assert.ok(answer.retrieval.contextOmittedChunkIds.length);
  assert.equal(index.validateCitation(answer.claims[0]!.citations[0]), true);
  assert.equal(answer.retrieval.exhaustive, false);
});

test('no lexical evidence or no fitting chunk skips inference honestly with zero usage', async () => {
  const index = await createDocumentIndex([{ id: 'x', text: 'Garden evidence.' }]); let calls = 0;
  const host: DocumentQueryHost = { async planInference(options) { return plan(options, 32); }, async inferStructured() { calls++; throw new Error('must not infer'); } };
  for (const question of ['zirconium', 'garden']) {
    const answer = await askDocuments(host, index, question, { maxNewTokens: 32 });
    assert.equal(answer.status, 'insufficient-evidence'); assert.deepEqual(answer.usage, { inputTokens: 0, outputTokens: 0, totalTokens: 0 });
    assert.equal(answer.claims.length, 0); assert.match(answer.answer, /unselected/);
  }
  assert.equal(calls, 0);
});

test('fabricated, paraphrased and unknown citations never become supported answers', async () => {
  const index = await createDocumentIndex([{ id: 'x', text: 'The garden contains 4 oak trees.' }]);
  const chunkId = index.search('oak').hits[0]!.chunk.id;
  const host = (text: string, ids: string[]): DocumentQueryHost => ({ async planInference(options) { return plan(options); }, async inferStructured() { return { value: { status: 'answered', claims: [{ text, chunkIds: ids }] } }; } });
  assert.equal((await askDocuments(host('The garden has four oaks.', [chunkId]), index, 'oak')).status, 'insufficient-evidence');
  await assert.rejects(askDocuments(host('4 oak trees', ['unknown']), index, 'oak'), { code: 'STRUCTURED_OUTPUT' });
  await assert.rejects(askDocuments(host('4 oak trees', [chunkId, chunkId]), index, 'oak'), { code: 'STRUCTURED_OUTPUT' });
});

test('mutation during inference rejects stale evidence instead of returning an answer', async () => {
  const index = await createDocumentIndex([{ id: 'x', text: 'Garden evidence.' }]);
  const host: DocumentQueryHost = { async planInference(options) { return plan(options); }, async inferStructured(options) {
    const chunk = evidence(options)[0]!; await index.updateDocument({ id: 'x', text: 'Replacement evidence.' });
    return { value: { status: 'answered', claims: [{ text: 'Garden', chunkIds: [chunk.id] }] } };
  } };
  await assert.rejects(askDocuments(host, index, 'Garden'), { code: 'INVALID_INPUT' });
});

test('cancellation awaits native host inference ownership before rejecting', async () => {
  const index = await createDocumentIndex([{ id: 'x', text: 'Garden evidence.' }]);
  const controller = new AbortController(); const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<{ value: unknown }>(); let settled = false;
  const host: DocumentQueryHost = { async planInference(options) { return plan(options); }, async inferStructured() { entered.resolve(); return release.promise; } };
  const operation = askDocuments(host, index.exportSnapshot(), 'Garden', { signal: controller.signal });
  void operation.then(() => { settled = true; }, () => { settled = true; });
  await entered.promise; controller.abort(); await Promise.resolve(); assert.equal(settled, false);
  release.resolve({ value: { status: 'insufficient-evidence', claims: [] } }); await assert.rejects(operation, { name: 'AbortError' });
});
