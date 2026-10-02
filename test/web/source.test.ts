import assert from 'node:assert/strict';
import test from 'node:test';
import { selectPage, snapshotPage } from '../../src/web/source.js';
import type { Page, Paragraph } from '../../src/types.js';

const page = (): Page => ({ url: 'https://example.test/article', title: 'Original', paragraphs: [{ id: 'p1', text: '庭園 contains oak trees 🌳.', source: { kind: 'html', startOffset: 0, endOffset: 32 } }, { id: 'p2', text: 'Navigation', source: { kind: 'html' } }], images: [{ id: 'i1', url: 'https://example.test/image.png', discoveredBy: ['img'] }] });

test('async selection owns the source snapshot despite concurrent caller changes and retained predicate references', async () => {
  const input = page();
  const original = await snapshotPage(await selectPage(input, { paragraphIds: ['p1'] }));
  const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>();
  let retained: Paragraph | undefined;
  const selection = selectPage(input, { paragraphIds: ['p1'], paragraph: async (source) => { retained = source as Paragraph; entered.resolve(); await release.promise; return true; } });
  await entered.promise;
  input.url = 'https://example.test/changed'; input.title = 'Changed'; input.paragraphs[0]!.text = 'Changed source'; input.images[0]!.discoveredBy.push('picture');
  release.resolve(); const accepted = await selection;
  assert.throws(() => { retained!.text = 'Injected evidence'; }, TypeError);
  assert.throws(() => { retained!.source.startOffset = 100; }, TypeError);
  assert.equal(accepted.paragraphs[0]!.text, '庭園 contains oak trees 🌳.');
  assert.deepEqual(await snapshotPage(accepted), original);
  assert.notEqual((await snapshotPage(input)).id, original.id);
});

test('source selectors reject unknown and duplicate IDs rather than silently losing coverage', async () => {
  await assert.rejects(selectPage(page(), { paragraphIds: ['missing'] }), TypeError);
  await assert.rejects(selectPage(page(), { imageIds: ['i1', 'i1'] }), TypeError);
  const input = page(); input.images[0]!.id = 'p1';
  await assert.rejects(selectPage(input), TypeError);
  assert.deepEqual((await selectPage(page(), { paragraphIds: ['p1'], imageIds: [] })).paragraphs.map(({ id }) => id), ['p1']);
});

test('a saved snapshot keeps uncited source text and metadata consistent with its digest during crypto awaits', async () => {
  const input = page(); const snapshot = snapshotPage(input);
  input.title = 'Changed later'; input.paragraphs[1]!.text = 'The uncited original was replaced'; input.paragraphs[0]!.source.startOffset = 12;
  const saved = await snapshot;
  assert.equal(saved.source.title, 'Original');
  assert.equal(saved.source.paragraphs[1]!.text, 'Navigation');
  assert.equal(saved.source.paragraphs[0]!.source.startOffset, 0);
  assert.throws(() => { saved.source.paragraphs[1]!.text = 'Overwritten'; }, TypeError);
  assert.equal((await snapshotPage(saved.source)).id, saved.id);
  assert.notEqual((await snapshotPage(input)).id, saved.id);
});
