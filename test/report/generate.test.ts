import assert from 'node:assert/strict';
import test from 'node:test';
import { Tensor, TokenizersBackend } from '@huggingface/transformers';
import { VisionEngine } from '../../src/core/engine.js';
import { getRegisteredModelProfile } from '../../src/cache/registry.js';
import { generateReport, planReport, ReportError, validateDescribeOptions, type ReportCheckpoint } from '../../src/report/generate.js';
import { parseReportCheckpoint, serializeReportCheckpoint, parseStructuredReport, serializeStructuredReport, renderMarkdown } from '../../src/report/index.js';
import { hashValue } from '../../src/web/source.js';
import { NekoError } from '../../src/errors.js';
import type { Page, StructuredReport } from '../../src/types.js';
import { generateExtractiveReport, planExtractiveReport } from '../../src/report/extractive.js';

// Deterministic tokens exercise the real engine, schemas, budgets and report planner without a model download.
function fixture(lossySections = false, firstSectionFailure?: 'STRUCTURED_OUTPUT' | 'POLICY_DENIED') {
  const prompts: string[] = [];
  const encoded = (text: string) => Array.from(text, (character) => BigInt(character.codePointAt(0)!));
  let invisible = 256;
  const vocab = Object.fromEntries(Array.from({ length: 256 }, (_, byte) => [
    String.fromCharCode(byte >= 33 && byte <= 126 || byte >= 161 && byte <= 172 || byte >= 174 ? byte : invisible++), byte,
  ]));
  const tokenizer = new TokenizersBackend({ model: { type: 'BPE', vocab, merges: [] }, decoder: { type: 'ByteLevel' }, added_tokens: [], normalizer: null, pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: false }, post_processor: null }, {});
  const processor = Object.assign((text: string) => ({ input_ids: new Tensor('int64', BigInt64Array.from(encoded(text)), [1, text.length]) }), {
    apply_chat_template: (messages: { role: string; content: { text?: string }[] }[]) => messages.map((message) => `${message.role}:${message.content.map((part) => part.text ?? '').join('')}\n`).join('') + 'assistant:',
    tokenizer,
  });
  const profile = getRegisteredModelProfile();
  const model = {
    sessions: Object.fromEntries(Object.entries(profile.dtype).map(([name, dtype]) => [name, { config: { device: 'cpu', dtype } }])),
    generation_config: { eos_token_id: 1000 },
    async generate(options: { input_ids: Tensor; max_new_tokens: number; streamer: { put(ids: bigint[][]): void; end(): void }; stopping_criteria: { interrupted: boolean } }) {
      const ids = Array.from(options.input_ids.data, BigInt);
      const prompt = ids.map((id) => String.fromCodePoint(Number(id))).join(''); prompts.push(prompt);
      const source = JSON.parse(prompt.split('Source JSON:\n')[1]!.split('\nNow output')[0]!) as { evidenceId?: string; quote?: string; text?: string; evidenceIds?: string[] }[];
      const section = prompt.includes('Give separate key points');
      if (section && firstSectionFailure && prompts.filter((value) => value.includes('Give separate key points')).length === 1) throw new NekoError('Fixture first stage output failure', 'generate', firstSectionFailure);
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
  const { engine } = fixture();
  try {
    const result = await report(engine);
    assert.deepEqual(result.sections.map(({ keyPoints }) => keyPoints.length), [4, 1]);
    assert.equal(result.sourceFacts.map(({ citation }) => citation.quote).join(''), page.paragraphs[0]!.text);
    assert.equal(result.metadata.coverage.retainedTextCharacters, page.paragraphs[0]!.text.length);
    assert.equal(result.metadata.coverage.modelCitedFactIds.length, 5);
    assert.deepEqual(result.metadata.coverage.summaryCitedFactIds, ['q1']);
    assert.equal(result.metadata.coverage.semanticRetention, 'not-measured');
    assert.equal(result.metadata.coverage.conclusionBasis, 'retained-source');
    assert.ok(!result.page.summary.includes('Pets'));
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
    assert.equal(result.metadata.evidence, 'references-validated-heuristic-audit');
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
    for (const version of [1, 2, 4]) await assert.rejects(parseReportCheckpoint(JSON.stringify({ ...saved, version })), (error: unknown) => error instanceof NekoError && error.code === 'CHECKPOINT_INVALID');
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

test('increased cumulative authorization resumes completed stages without changing report identity', async () => {
  const { engine, prompts } = fixture();
  try {
    const plan = await planReport(engine, page, { maxNewTokens: 512, contextWindowTokens: 8192 });
    let failed: ReportError | undefined;
    await assert.rejects(report(engine, { budget: { maxTotalTokens: plan.stages[0]!.inputTokens! + 512 } }), (error: unknown) => { assert.ok(error instanceof ReportError); failed = error; return error.code === 'BUDGET_EXCEEDED'; });
    assert.ok(failed); assert.equal(failed.partial.completedStages.length, 1);
    const saved = await parseReportCheckpoint(await serializeReportCheckpoint(failed.checkpoint));
    const before = prompts.length;
    const resumed = await report(engine, { resume: saved, budget: { maxTotalTokens: 100_000 } });
    assert.equal(resumed.metadata.resumedStages, 1);
    assert.equal(prompts.length - before, plan.stages.length - 1);
    assert.equal(resumed.claims[0]!.audit.status, 'supported');
  } finally { await engine.dispose(); }
});

test('only explicitly authorized failed output stages retry and account for failed input tokens', async () => {
  const { engine, prompts } = fixture(false, 'STRUCTURED_OUTPUT');
  try {
    let checkpoint: ReportCheckpoint | undefined;
    const result = await report(engine, { retries: { section: 1 }, onCheckpoint: (value) => { checkpoint = value; } });
    assert.ok(checkpoint); assert.equal(checkpoint.attempts[0]!.outcome, 'error');
    assert.ok(checkpoint.attempts[0]!.usage.inputTokens > 0);
    assert.ok(prompts[1]!.includes('Retry 1:'));
    assert.ok(prompts.slice(2).every((prompt) => !prompt.includes('Retry 1:')));
    assert.equal(result.metadata.retryAttempts, 1);
    assert.equal(checkpoint.attempts.reduce((sum, attempt) => sum + attempt.usage.totalTokens, 0), result.metadata.usage.totalTokens);
    await serializeReportCheckpoint(checkpoint);
  } finally { await engine.dispose(); }
  const denied = fixture(false, 'POLICY_DENIED');
  try { await assert.rejects(report(denied.engine, { retries: { section: 3 } }), (error: unknown) => error instanceof ReportError && error.code === 'POLICY_DENIED'); assert.equal(denied.prompts.length, 1); }
  finally { await denied.engine.dispose(); }
});

test('aggregate budget truncation preserves accounting and resumes without authorizing output retries', async () => {
  const { engine } = fixture();
  const options = { maxNewTokens: 512, contextWindowTokens: 8192, retries: { section: 0 } };
  try {
    const plan = await planReport(engine, page, options);
    let saved: ReportCheckpoint | undefined;
    await assert.rejects(report(engine, { ...options, budget: { maxTotalTokens: plan.stages[0]!.inputTokens! + 1 } }), (error: unknown) => {
      assert.ok(error instanceof ReportError);
      assert.equal(error.code, 'BUDGET_EXCEEDED');
      saved = error.checkpoint;
      return true;
    });
    assert.ok(saved);
    assert.equal(saved.attempts[0]!.usage.outputTokens, 1);
    assert.equal(saved.attempts[0]!.errorCode, 'BUDGET_EXCEEDED');
    let final: ReportCheckpoint | undefined;
    const resumed = await report(engine, { ...options, budget: { maxTotalTokens: 100000 }, resume: saved, onCheckpoint: (value) => { final = value; } });
    assert.ok(final);
    assert.deepEqual(final.attempts[0], saved.attempts[0]);
    assert.equal(resumed.metadata.retryAttempts, 0);
    assert.equal(resumed.metadata.usage.totalTokens, final.attempts.reduce((sum, attempt) => sum + attempt.usage.totalTokens, 0));
    await parseReportCheckpoint(await serializeReportCheckpoint(final));
  } finally { await engine.dispose(); }
});

test('extractive reporting retains exact multilingual text and image metadata without an engine', async () => {
  const profile = getRegisteredModelProfile();
  const input: Page = { url: 'about:blank', paragraphs: [{ id: 'p1', text: '臺灣價格為 7 元。 No translation. <script>inert</script>', source: { kind: 'html' } }], images: [{ id: 'i1', url: 'https://example.test/not-fetched.png', discoveredBy: ['img'] }] };
  const plan = await planExtractiveReport(input, { mode: 'extractive' });
  assert.equal(plan.knownInputTokens, 0); assert.deepEqual(plan.stages, []);
  const result = await generateExtractiveReport(input, { mode: 'extractive' }, { model: { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype }, execution: { mode: 'inline', runtime: 'node' } });
  assert.notEqual(typeof result, 'string'); const report = result as StructuredReport;
  assert.equal(report.language, 'und'); assert.equal(report.metadata.backend, null); assert.equal(report.metadata.usage.totalTokens, 0);
  assert.equal(report.page.summary, input.paragraphs[0]!.text); assert.equal(report.conclusion, input.paragraphs[0]!.text);
  assert.equal(report.images[0]!.status, 'retained'); assert.ok(report.claims.every(({ audit }) => audit.status === 'supported'));
  assert.deepEqual(report.metadata.coverage.modelCitedFactIds, []);
  await parseStructuredReport(await serializeStructuredReport(report));
  assert.ok(!renderMarkdown(report).includes('<script>'));
});
