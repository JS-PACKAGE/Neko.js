import assert from 'node:assert/strict';
import test from 'node:test';
import { askDocument } from '../../src/web/query.js';
import { extractPage } from '../../src/web/extract.js';
import type { DocumentInference } from '../../src/web/query.js';

// The callback is the public injection boundary; these tests exercise provenance without claiming model quality.
test('document QA preserves the exact question and validates UTF-16 quote provenance', async () => {
  const page = await extractPage('<p>庭園 🌳 contains 4 oak trees.</p>');
  const question = '  How many oak trees?\nIgnore instructions?  ';
  const text = page.paragraphs[0]!.text;
  const startOffset = text.indexOf('contains');
  const quote = text.slice(startOffset);
  const answer = await askDocument(page, question, async () => ({
    value: { status: 'answered', claims: [{ text: quote, paragraphIds: ['p1'] }] },
  }));
  assert.equal(answer.status, 'answered');
  assert.equal(answer.question, question);
  assert.equal(answer.claims[0]!.audit.status, 'supported');
  const citation = answer.claims[0]!.citations[0]!;
  assert.equal(citation.snapshotId, answer.snapshot.id);
  assert.equal(citation.versionId, answer.snapshot.paragraphs[0]!.versionId);
  assert.equal(citation.quote, text.slice(citation.startOffset, citation.endOffset));
  assert.equal(answer.evidence, 'exact-quotes-heuristic-audit-not-fact-checked');
});

test('QA rejects unknown or duplicate evidence IDs rather than certifying references', async () => {
  const page = await extractPage('<p>The garden contains 4 oak trees.</p>');
  for (const paragraphIds of [['missing'], ['p1', 'p1']]) {
    await assert.rejects(askDocument(page, 'How many trees?', async () => ({ value: { status: 'answered', claims: [{ text: '4 oak trees', paragraphIds }] } })), { code: 'STRUCTURED_OUTPUT' });
  }
  await assert.rejects(askDocument(page, 'How many trees?', async () => ({ value: { status: 'answered', claims: [] } })), { code: 'STRUCTURED_OUTPUT' });
});

test('unknown or contradicted generated claims produce explicit insufficient evidence', async () => {
  const source = 'The garden contains 4 oak trees.';
  for (const text of ['There are four trees in the garden.', 'The garden contains 9 oak trees.', 'The garden contains  4 oak trees.']) {
    const answer = await askDocument(`<p>${source}</p>`, 'How many trees?', async () => ({ value: { status: 'answered', claims: [{ text, paragraphIds: ['p1'] }] } }));
    assert.equal(answer.status, 'insufficient-evidence');
    assert.deepEqual(answer.claims, []);
    assert.match(answer.answer, /Insufficient evidence/);
  }
});

test('QA source selection does not expose excluded evidence and empty selection skips inference', async () => {
  const page = await extractPage('<p>Selected evidence.</p><p>Excluded secret.</p>');
  const infer: DocumentInference = async (options) => {
    assert.doesNotMatch(options.prompt!, /Excluded secret/);
    return { value: { status: 'insufficient-evidence', claims: [] } };
  };
  assert.equal((await askDocument(page, 'What is known?', infer, { sources: { paragraphIds: ['p1'] } })).snapshot.source.paragraphs.length, 1);
  const empty = await askDocument(page, 'What is known?', async () => { throw new Error('Inference must not run'); }, { sources: { paragraphIds: [] } });
  assert.equal(empty.status, 'insufficient-evidence');
  assert.deepEqual(empty.snapshot.source.paragraphs, []);
});

test('QA cancellation does not abandon pending native inference work', async () => {
  const controller = new AbortController();
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<{ value: unknown }>();
  let settled = false;
  const answer = askDocument('<p>Evidence.</p>', 'What is known?', async () => { entered.resolve(); return release.promise; }, { signal: controller.signal });
  void answer.then(() => { settled = true; }, () => { settled = true; });
  await entered.promise; controller.abort();
  await Promise.resolve(); assert.equal(settled, false);
  release.resolve({ value: { status: 'insufficient-evidence', claims: [] } });
  await assert.rejects(answer, { name: 'AbortError' });
});
