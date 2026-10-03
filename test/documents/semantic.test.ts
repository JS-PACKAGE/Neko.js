import assert from 'node:assert/strict';
import test from 'node:test';
import { askDocuments, createDocumentIndex, type DocumentEmbedder, type DocumentQueryHost } from '../../src/documents/index.js';
import type { InferencePlan, InferencePlanOptions, StructuredInferOptions } from '../../src/core/engine.js';
import { getRegisteredModelProfile } from '../../src/cache/registry.js';

// Deterministic concept space: dimensions are vehicle, fruit, finance. Synonyms share a dimension but no words.
const concepts: Record<string, number> = { car: 0, automobile: 0, vehicle: 0, apple: 1, banana: 1, fruit: 1, invoice: 2, payment: 2, bill: 2 };
function counted(overrides: Partial<DocumentEmbedder> = {}) {
  const calls: { kind: string; texts: string[] }[] = [];
  const embedder: DocumentEmbedder = {
    id: 'fixture-concepts-v1', dimensions: 3,
    async embed(texts, { kind }) {
      calls.push({ kind, texts: [...texts] });
      return texts.map((text) => { const vector = [0, 0, 0.001]; for (const word of text.toLowerCase().match(/[a-z]+/g) ?? []) if (word in concepts) vector[concepts[word]!]! += 1; return vector; });
    },
    ...overrides,
  };
  return { embedder, calls, documentTexts: () => calls.filter(({ kind }) => kind === 'document').flatMap(({ texts }) => texts) };
}
const documents = [
  { id: 'garage', text: 'The car is parked outside.' },
  { id: 'kitchen', text: 'A banana sits in the bowl.' },
  { id: 'office', text: 'The invoice is overdue.' },
];

test('hybrid retrieval finds a synonym that shares no lexical term, with exact citations', async () => {
  const index = await createDocumentIndex(documents);
  assert.deepEqual(index.search('automobile').hits, []);
  const { embedder } = counted();
  const result = await index.searchHybrid('automobile', { embedder, topK: 1 });
  assert.equal(result.hits[0]!.chunk.documentId, 'garage');
  assert.equal(index.validateCitation(result.hits[0]!.citation), true);
  assert.equal(result.coverage.strategy, 'hybrid-bm25-vector-rrf-v1');
  assert.deepEqual(result.coverage.semantic, { embedderId: 'fixture-concepts-v1', dimensions: 3, embeddedChunks: 3, cachedChunks: 0, scoredChunks: 3, minSimilarity: null });
  assert.equal(result.coverage.exhaustive, false);
});

test('an exact lexical hit and a semantic neighbour both survive fusion', async () => {
  const index = await createDocumentIndex([...documents, { id: 'notes', text: 'Quartz inventory ledger entry.' }]);
  const { embedder } = counted();
  const result = await index.searchHybrid('quartz automobile', { embedder, topK: 2 });
  assert.deepEqual(result.hits.map(({ chunk }) => chunk.documentId).sort(), ['garage', 'notes']);
});

test('chunk vectors are cached by content, so a mutation only embeds changed text', async () => {
  const index = await createDocumentIndex(documents);
  const probe = counted();
  await index.searchHybrid('automobile', { embedder: probe.embedder });
  assert.equal(probe.documentTexts().length, 3);
  const again = await index.searchHybrid('automobile', { embedder: probe.embedder });
  assert.equal(probe.documentTexts().length, 3);
  assert.equal(again.coverage.semantic!.embeddedChunks, 0);
  assert.equal(again.coverage.semantic!.cachedChunks, 3);
  await index.updateDocument({ id: 'kitchen', text: 'An apple sits in the bowl.' });
  await index.searchHybrid('automobile', { embedder: probe.embedder });
  assert.deepEqual(probe.documentTexts().slice(3), ['An apple sits in the bowl.']);
  // A different embedder identity never reuses another model's vectors.
  const other = counted({ id: 'fixture-concepts-v2' });
  await index.searchHybrid('automobile', { embedder: other.embedder });
  assert.equal(other.documentTexts().length, 3);
});

test('embedder calls are batched, distinguish query from document, and respect document filters', async () => {
  const index = await createDocumentIndex(documents);
  const probe = counted();
  await index.searchHybrid('vehicle', { embedder: probe.embedder, batchSize: 2, documentIds: ['garage', 'office'] });
  assert.deepEqual(probe.calls.map(({ kind, texts }) => [kind, texts.length]), [['query', 1], ['document', 2]]);
  assert.deepEqual(probe.documentTexts().sort(), ['The car is parked outside.', 'The invoice is overdue.']);
});

test('minSimilarity drops weak semantic candidates without hiding exact lexical matches', async () => {
  const index = await createDocumentIndex(documents);
  const { embedder } = counted();
  const result = await index.searchHybrid('automobile', { embedder, minSimilarity: 0.9 });
  assert.deepEqual(result.hits.map(({ chunk }) => chunk.documentId), ['garage']);
  assert.equal(result.coverage.semantic!.scoredChunks, 1);
  // "bowl" has no concept dimension: only the lexical match can bring the kitchen chunk in.
  const lexical = await index.searchHybrid('bowl', { embedder, minSimilarity: 0.99, topK: 3 });
  assert.ok(lexical.hits.some(({ chunk }) => chunk.documentId === 'kitchen'));
});

test('untrusted embedder output and failures are rejected without caching bad vectors', async () => {
  const index = await createDocumentIndex(documents);
  const bad: [string, DocumentEmbedder['embed']][] = [
    ['wrong dimensions', async (texts) => texts.map(() => [1, 0])],
    ['non-finite', async (texts) => texts.map(() => [Number.NaN, 0, 0])],
    ['zero vector', async (texts) => texts.map(() => [0, 0, 0])],
    ['wrong count', async () => [[1, 0, 0]]],
  ];
  for (const [label, embed] of bad) {
    const { embedder } = counted({ embed });
    await assert.rejects(index.searchHybrid('car', { embedder }), { code: 'MODEL_OUTPUT' }, label);
  }
  await assert.rejects(index.searchHybrid('car', { embedder: counted({ embed: async () => { throw new Error('model offline'); } }).embedder }), { code: 'OPERATION_FAILED', message: 'model offline' });
  const good = counted();
  await index.searchHybrid('car', { embedder: good.embedder });
  assert.equal(good.documentTexts().length, 3);
});

test('limits, invalid settings and cancellation fail before unbounded embedding work', async () => {
  const index = await createDocumentIndex(documents);
  const probe = counted();
  await assert.rejects(index.searchHybrid('car', { embedder: probe.embedder, maxEmbeddedChunks: 2 }), { code: 'INVALID_INPUT' });
  assert.equal(probe.calls.length, 0);
  await assert.rejects(index.searchHybrid('car', { embedder: { ...probe.embedder, dimensions: 0 } }), { code: 'INVALID_INPUT' });
  await assert.rejects(index.searchHybrid('car', { embedder: probe.embedder, batchSize: 0 }), { code: 'INVALID_INPUT' });
  await assert.rejects(index.searchHybrid('car', { embedder: probe.embedder, minSimilarity: 2 }), { code: 'INVALID_INPUT' });
  const controller = new AbortController();
  const slow = counted({ async embed(texts, { kind }) { if (kind === 'document') controller.abort(new Error('stop')); return texts.map(() => [1, 0, 0]); } });
  await assert.rejects(index.searchHybrid('car', { embedder: slow.embedder, signal: controller.signal, batchSize: 1 }), { code: 'ABORTED' });
});

test('askDocuments retrieves semantically and only sends selected chunks to the host', async () => {
  const registered = getRegisteredModelProfile();
  const model: InferencePlan['model'] = { id: registered.id, revision: registered.revision, profile: registered.profile, dtype: registered.dtype };
  const prompts: string[] = [];
  const host: DocumentQueryHost = {
    async planInference(options: InferencePlanOptions) { return { inputTokens: 10, maxNewTokens: options.maxNewTokens ?? 512, contextLimit: 4096, availableOutputTokens: 4000, fits: true, model }; },
    async inferStructured(options: StructuredInferOptions) {
      prompts.push(options.prompt!);
      const payload: { chunks: { id: string; text: string }[] } = JSON.parse(options.prompt!.slice(options.prompt!.indexOf('\n') + 1));
      return { value: { status: 'answered', claims: [{ text: 'The car is parked outside.', chunkIds: [payload.chunks[0]!.id] }] } };
    },
  };
  const index = await createDocumentIndex(documents);
  const { embedder } = counted();
  const answer = await askDocuments(host, index, 'Where is the automobile?', { embedder, search: { topK: 1 } });
  assert.equal(answer.status, 'answered');
  assert.equal(answer.retrieval.strategy, 'hybrid-bm25-vector-rrf-v1');
  assert.equal(answer.claims[0]!.citations[0]!.documentId, 'garage');
  assert.ok(!prompts[0]!.includes('banana'));
  await assert.rejects(askDocuments(host, index, 'Where?', { embedding: { batchSize: 4 } }), /require an embedder/);
});
