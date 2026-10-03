import assert from 'node:assert/strict';
import test from 'node:test';
import { Tensor } from '@huggingface/transformers';
import { VisionEngine } from '../../src/core/engine.js';
import { getModelProfile } from '../../src/cache/manifest.js';
import { generateReport, ReportError, validateDescribeOptions, type ReportCheckpoint } from '../../src/report/generate.js';
import { parseReportCheckpoint, serializeReportCheckpoint, parseStructuredReport, serializeStructuredReport, renderMarkdown } from '../../src/report/index.js';
import { hashValue } from '../../src/web/source.js';
import { NekoError } from '../../src/errors.js';
import type { Page, StructuredReport } from '../../src/types.js';

// Deterministic tokens exercise the real engine, schemas, budgets and report planner without a model download.
function fixture(lossySections = false) {
  const prompts: string[] = [];
  const encoded = (text: string) => Array.from(text, (character) => BigInt(character.codePointAt(0)!));
  const processor = Object.assign((text: string) => ({ input_ids: new Tensor('int64', BigInt64Array.from(encoded(text)), [1, text.length]) }), {
    apply_chat_template: (messages: { role: string; content: { text?: string }[] }[]) => messages.map((message) => `${message.role}:${message.content.map((part) => part.text ?? '').join('')}\n`).join('') + 'assistant:',
    tokenizer: { all_special_ids: [1000], encode: encoded, decode: (ids: bigint[]) => ids.filter((id) => id !== 1000n).map((id) => String.fromCodePoint(Number(id))).join('') },
  });
  const profile = getModelProfile();
  const model = {
    sessions: Object.fromEntries(Object.entries(profile.dtype).map(([name, dtype]) => [name, { config: { device: 'cpu', dtype } }])),
    generation_config: { eos_token_id: 1000 },
    async generate(options: { input_ids: Tensor; max_new_tokens: number; streamer: { put(ids: bigint[][]): void; end(): void }; stopping_criteria: { interrupted: boolean } }) {
      const ids = Array.from(options.input_ids.data, BigInt);
      const prompt = ids.map((id) => String.fromCodePoint(Number(id))).join(''); prompts.push(prompt);
      const source = JSON.parse(prompt.split('Source JSON:\n')[1]!.split('\nNow output')[0]!) as { evidenceId?: string; quote?: string; text?: string; evidenceIds?: string[] }[];
      const section = prompt.includes('Give separate key points');
      const facts = section && !lossySections ? source : source.slice(0, 1);
      const claims = facts.map((fact) => ({ text: (fact.quote ?? fact.text)!.trim(), evidenceIds: fact.evidenceId ? [fact.evidenceId] : fact.evidenceIds! }));
      const multiple = prompt.includes('with claims (an array');
      const output = encoded(JSON.stringify(multiple ? { claims } : claims[0]));
      options.streamer.put([ids]);
      if (!options.stopping_criteria.interrupted) { const tokens = output.slice(0, options.max_new_tokens); ids.push(...tokens); options.streamer.put([tokens]); }
      options.streamer.end();
      return new Tensor('int64', BigInt64Array.from(ids), [1, ids.length]);
    },
    async dispose() {},
  };
  const Construct = VisionEngine as unknown as new (...args: unknown[]) => VisionEngine;
  const engine = new Construct(model, processor, { runtime: 'node', device: 'cpu', executionProviders: ['cpu'] }, 0, false, 8192, 2000, { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype }, undefined, true);
  return { engine, prompts };
}

const page: Page = { url: 'about:blank', paragraphs: [{ id: 'p1', text: 'Admission costs 7 dollars. Doors open at 09:30. The route spans 2-4 km. Pets are not allowed. Parking is free.', source: { kind: 'html' } }], images: [] };

async function report(engine: VisionEngine, options: Parameters<typeof generateReport>[2] = {}): Promise<StructuredReport> {
  const result = await generateReport(engine, page, { maxNewTokens: 512, contextWindowTokens: 8192, ...options });
  assert.notEqual(typeof result, 'string'); return result as StructuredReport;
}

test('multi-fact paragraphs produce bounded multi-point sections with complete independent source retention', async () => {
  const { engine, prompts } = fixture();
  try {
    const result = await report(engine);
    assert.deepEqual(result.sections.map(({ keyPoints }) => keyPoints.length), [4, 1]);
    assert.equal(result.sourceFacts.map(({ citation }) => citation.quote).join(''), page.paragraphs[0]!.text);
    assert.equal(result.metadata.coverage.retainedTextCharacters, page.paragraphs[0]!.text.length);
    assert.equal(result.metadata.coverage.modelCitedFactIds.length, 5);
    assert.deepEqual(result.metadata.coverage.summaryCitedFactIds, ['q1']);
    assert.equal(result.metadata.coverage.semanticRetention, 'not-measured');
    assert.equal(result.metadata.coverage.conclusionBasis, 'retained-source');
    assert.ok(prompts.at(-1)!.includes('Pets are not allowed.'));
    assert.ok(!result.page.summary.includes('Pets'));
    assert.ok(prompts.slice(0, 2).every((prompt) => prompt.includes('with claims (an array')));
    assert.ok(prompts.slice(2).every((prompt) => prompt.includes('one concise evidence-backed passage') && !prompt.includes('with claims (an array')));
    const loaded = await parseStructuredReport(await serializeStructuredReport(result));
    assert.deepEqual(loaded.sourceFacts, result.sourceFacts);
    assert.ok(renderMarkdown(loaded).includes('Pets are not allowed'));
  } finally { await engine.dispose(); }
});

test('lossy generated points never erase persisted source quotes or inflate semantic coverage', async () => {
  const { engine } = fixture(true);
  try {
    const result = await report(engine);
    assert.deepEqual(result.metadata.coverage.modelCitedFactIds, ['q1', 'q5']);
    assert.equal(result.sourceFacts.length, 5);
    assert.equal(result.sourceFacts.map(({ citation }) => citation.quote).join(''), page.paragraphs[0]!.text);
    assert.equal(result.metadata.evidence, 'references-validated-not-fact-checked');
  } finally { await engine.dispose(); }
});

test('versioned checkpoints round-trip, reuse deterministic stages and fail closed on corrupt provenance', async () => {
  const { engine, prompts } = fixture(); let checkpoint: ReportCheckpoint | undefined;
  try {
    const original = await report(engine, { onCheckpoint: (value) => { checkpoint = value; } });
    assert.ok(checkpoint);
    const saved = await parseReportCheckpoint(await serializeReportCheckpoint(checkpoint));
    const count = prompts.length;
    const resumed = await report(engine, { resume: saved });
    assert.equal(prompts.length, count);
    assert.equal(resumed.metadata.resumedStages, Object.keys(saved.completed).length);
    assert.deepEqual(resumed.sourceFacts, original.sourceFacts);
    for (const version of [1, 3]) await assert.rejects(parseReportCheckpoint(JSON.stringify({ ...saved, version })), (error: unknown) => error instanceof NekoError && error.code === 'CHECKPOINT_INVALID');
    const corrupt = structuredClone(saved); corrupt.sourceFacts[0]!.citation.quote = 'Tampered';
    corrupt.checksum = await hashValue({ ...corrupt, checksum: undefined });
    await assert.rejects(parseReportCheckpoint(JSON.stringify(corrupt)), (error: unknown) => error instanceof NekoError && error.code === 'CHECKPOINT_INVALID');
    const outside = structuredClone(saved); outside.completed['section:0']!.claims[0]!.evidenceIds = ['q999'];
    outside.checksum = await hashValue({ ...outside, checksum: undefined });
    await assert.rejects(parseReportCheckpoint(JSON.stringify(outside)), (error: unknown) => error instanceof NekoError && error.code === 'CHECKPOINT_INVALID');
    const wrongGroup = structuredClone(saved); wrongGroup.completed['section:0']!.claims[0]!.evidenceIds = ['q5'];
    wrongGroup.checksum = await hashValue({ ...wrongGroup, checksum: undefined });
    await assert.rejects(parseReportCheckpoint(JSON.stringify(wrongGroup)), (error: unknown) => error instanceof NekoError && error.code === 'CHECKPOINT_INVALID');
    const gapPlan = structuredClone(saved); gapPlan.sectionPlan.pop();
    gapPlan.checksum = await hashValue({ ...gapPlan, checksum: undefined });
    await assert.rejects(parseReportCheckpoint(JSON.stringify(gapPlan)), (error: unknown) => error instanceof NekoError && error.code === 'CHECKPOINT_INVALID');
  } finally { await engine.dispose(); }
});

test('failed budgets retain a serializable quote ledger; cancellation does not run later stages', async () => {
  const { engine, prompts } = fixture();
  try {
    let failed: ReportError | undefined;
    await assert.rejects(report(engine, { budget: { maxTotalTokens: 1 } }), (error: unknown) => {
      assert.ok(error instanceof ReportError); assert.equal(error.code, 'BUDGET_EXCEEDED'); failed = error; return true;
    });
    assert.ok(failed);
    await parseReportCheckpoint(await serializeReportCheckpoint(failed.checkpoint));
    assert.equal(failed.checkpoint.sourceFacts.map(({ citation }) => citation.quote).join(''), page.paragraphs[0]!.text);
    const controller = new AbortController();
    await assert.rejects(report(engine, { signal: controller.signal, onEvent: (event) => { if (event.type === 'stage-complete') controller.abort(); } }), (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
    assert.equal(prompts.length, 1);
  } finally { await engine.dispose(); }
});

test('cheap report validation rejects malformed options without invoking a model', () => {
  for (const options of [{ language: 'not_a_language' }, { maxNewTokens: 0 }, { contextWindowTokens: 1 }, { budget: { maxTotalTokens: -1 } }, { generation: { temperature: 0 } }]) assert.throws(() => validateDescribeOptions(options), (error: unknown) => error instanceof NekoError && error.stage === 'preprocess' && error.code === 'INVALID_INPUT');
});
