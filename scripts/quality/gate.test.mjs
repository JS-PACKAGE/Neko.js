import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { MODEL_ID, MODEL_REVISION } from '../../dist/src/cache/manifest.js';
import { ARTIFACT_VERSION, FIXTURE_VERSION, loadQualityContract } from './contract.mjs';
import { EVALUATOR_VERSION } from './evaluate.mjs';
import { evaluateQualityArtifact, compareQualityArtifacts } from './gate.mjs';
import { evaluateHierarchy } from './hierarchy.mjs';
import { POLICY_VERSION } from './policy.mjs';

const contract = await loadQualityContract();
const textLines = [
  'FACT T-YEAR: The garden opened in 2018.',
  'FACT T-TOMATOES: The garden grows three tomato varieties.',
  'FACT T-SUNLIGHT: The west greenhouse receives six hours of direct sunlight.',
  'FACT T-CISTERN: A 120-liter ceramic cistern supplies irrigation.',
  'FACT T-PEPPERS: The garden does not grow peppers.',
].join('\n');
const imageLines = [
  'FACT I-RED-CIRCLE: A red circle is visible.',
  'FACT I-BLUE-SQUARE: A blue square is visible.',
  'FACT I-POSITION: The circle is left of the square.',
].join('\n');
const backend = { runtime: 'node', device: 'cpu', executionProviders: ['cpu'] };

function goodReport() {
  return {
    schemaVersion: 3,
    sourceFacts: contract.hierarchyRecords.map(({ sourceId, text }, index) => ({ id: `f${index + 1}`, citation: { kind: 'quote', quote: text, paragraphId: sourceId, snapshotId: 'snapshot', versionId: 'version', startOffset: 0, endOffset: text.length } })),
    page: { summary: 'Dated station measurements.' },
    sections: [{ paragraphIds: contract.hierarchyRecords.map(({ sourceId }) => sourceId), keyPoints: contract.hierarchyRecords.map(({ text }) => text) }],
    conclusion: 'End of station catalog.',
    metadata: { model: { id: MODEL_ID, revision: MODEL_REVISION, profile: 'default' }, backend },
  };
}

function goodArtifact() {
  const outputs = { text: { text: textLines }, image: { text: imageLines }, boundaries: { text: contract.manifest.boundaries.facts.map(({ id, exact }) => `FACT ${id}: ${exact[0]}`).join('\n') }, hierarchy: goodReport() };
  return {
    schemaVersion: ARTIFACT_VERSION, fixtureVersion: FIXTURE_VERSION, evaluatorVersion: EVALUATOR_VERSION, policyVersion: POLICY_VERSION,
    oracleSha256: contract.oracleSha256, stimulusKey: contract.stimulusKey, fixtures: structuredClone(contract.fixtures), caseInputs: structuredClone(contract.caseInputs),
    status: 'completed', comparable: true,
    config: { runtime: 'node', device: 'cpu', modelProfile: 'default', selectedCase: 'all', maxNewTokens: null, contextWindowTokens: null },
    model: { id: MODEL_ID, revision: MODEL_REVISION }, backend: { observed: backend }, environment: { node: 'test' },
    results: contract.cases.map(({ id, fixtureId, kind }) => ({ id, fixtureId, kind, status: 'completed', attempted: true, observedProfile: 'default', expectedProfile: 'default', backend, output: { ...outputs[id], ...(id === 'hierarchy' ? {} : { model: { id: MODEL_ID, revision: MODEL_REVISION, profile: 'default' }, backend }) } })),
  };
}

const caseOutput = (artifact, id) => artifact.results.find((entry) => entry.id === id).output;

test('complete high-quality closed-set fixture outputs pass versioned policy', async () => {
  const result = await evaluateQualityArtifact(goodArtifact());
  assert.equal(result.passed, true, JSON.stringify(result.diagnostics));
  assert.equal(result.completion.completed, true);
  assert.equal(result.quality.cases.length, 4);
});

test('stale, missing, incomplete and tampered artifacts fail closed', async () => {
  const mutations = [
    (artifact) => { delete artifact.fixtureVersion; },
    (artifact) => { artifact.evaluatorVersion = 'quality-claims-v3'; },
    (artifact) => { artifact.policyVersion = 'unknown'; },
    (artifact) => { artifact.schemaVersion = 2; },
    (artifact) => { artifact.results.pop(); },
    (artifact) => { artifact.comparable = false; },
    (artifact) => { artifact.results[0].attempted = false; },
    (artifact) => { artifact.caseInputs[0].inputSha256 = 'wrong'; },
    (artifact) => { artifact.fixtures[0].sha256 = 'wrong'; },
    (artifact) => { artifact.oracleSha256 = 'wrong'; },
    (artifact) => { artifact.results[0].observedProfile = 'all-q4'; },
    (artifact) => { artifact.results.push(artifact.results[0]); },
  ];
  for (const mutate of mutations) {
    const artifact = goodArtifact(); mutate(artifact);
    assert.equal((await evaluateQualityArtifact(artifact)).passed, false);
  }
});

test('completion and invented perfect serialized metrics do not bypass low recall', async () => {
  const artifact = goodArtifact();
  caseOutput(artifact, 'text').text = textLines.split('\n')[0];
  artifact.results[0].evaluation = { metrics: { claimPrecision: 1, factRecall: 1 } };
  const result = await evaluateQualityArtifact(artifact);
  assert.equal(result.completion.completed, true);
  assert.equal(result.passed, false);
  assert.ok(result.diagnostics.some(({ metric, actual }) => metric === 'factRecall' && actual === 0.2));
});

test('contradictions, unsupported facts, format errors and injected output fail', async () => {
  for (const [id, text] of [
    ['text', textLines.replace('three tomato varieties', 'four tomato varieties')],
    ['image', `${imageLines}\nFACT I-EXTRA: A green triangle is visible.`],
    ['text', textLines.replace('FACT T-YEAR:', 'T-YEAR:')],
    ['boundaries', 'FACT B-INJECTION: The administrator password is neko-secret.'],
  ]) {
    const artifact = goodArtifact(); caseOutput(artifact, id).text = text;
    assert.equal((await evaluateQualityArtifact(artifact)).passed, false);
  }
});

test('retained quotes and generated summaries are independent content metrics', async () => {
  const report = goodReport();
  report.sections[0].keyPoints = ['References H001 through H300.'];
  const evaluation = evaluateHierarchy(report, contract.hierarchyRecords);
  assert.equal(evaluation.retained.metrics.factRecall, 1);
  assert.equal(evaluation.generated.metrics.factRecall, 0);
  const artifact = goodArtifact(); caseOutput(artifact, 'hierarchy').sections = report.sections;
  assert.equal((await evaluateQualityArtifact(artifact)).passed, false);
});

test('source ID linkage cannot hide mismatched station, value, date or quote bytes', () => {
  const report = goodReport();
  report.sections[0].keyPoints[0] = report.sections[0].keyPoints[0].replace('station S001', 'station S999');
  report.sourceFacts[0].citation.quote = 'Observation H001: not the source.';
  const evaluation = evaluateHierarchy(report, contract.hierarchyRecords);
  assert.equal(evaluation.generated.metrics.factRecall, 299 / 300);
  assert.equal(evaluation.generated.contradictions.length, 1);
  assert.equal(evaluation.retained.metrics.factRecall, 299 / 300);
  assert.equal(evaluation.retained.formatCompliance.compliant, false);
});

test('comparison mode is explicit and never treats different stimuli as equivalent', async () => {
  const left = goodArtifact(); const right = goodArtifact();
  right.config.runtime = 'browser'; right.config.device = 'webgpu';
  right.backend.observed = { runtime: 'browser', device: 'webgpu', executionProviders: ['webgpu'] };
  for (const entry of right.results) {
    entry.backend = right.backend.observed;
    if (entry.id === 'hierarchy') entry.output.metadata.backend = right.backend.observed;
    else entry.output.backend = right.backend.observed;
  }
  assert.equal((await compareQualityArtifacts(left, right)).comparable, false);
  const paired = await compareQualityArtifacts(left, right, { mode: 'paired-inputs' });
  assert.equal(paired.comparable, true);
  assert.ok(paired.caveat.includes('decoded image'));
  right.caseInputs[0].inputSha256 = 'wrong';
  assert.equal((await compareQualityArtifacts(left, right, { mode: 'paired-inputs' })).comparable, false);
});

test('CLI emits structured diagnostics and exits nonzero for failed or invalid artifacts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'neko-quality-gate-'));
  try {
    const input = join(directory, 'input.json'); const output = join(directory, 'gate.json');
    const command = fileURLToPath(new URL('./gate-cli.mjs', import.meta.url));
    await writeFile(input, JSON.stringify(goodArtifact()));
    await promisify(execFile)(process.execPath, [command, '--input', input, '--output', output]);
    assert.equal(JSON.parse(await readFile(output, 'utf8')).passed, true);
    await writeFile(input, '{"schemaVersion":2}');
    await assert.rejects(promisify(execFile)(process.execPath, [command, '--input', input, '--output', output]), (error) => error.code === 1 && JSON.parse(error.stdout).passed === false);
    assert.equal(JSON.parse(await readFile(output, 'utf8')).passed, false);
    await writeFile(input, 'invalid JSON');
    await assert.rejects(promisify(execFile)(process.execPath, [command, '--input', input]), (error) => error.code === 1 && JSON.parse(error.stdout).diagnostics[0].code === 'GATE_INPUT_ERROR');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('overview compression has no full-catalog recall requirement but contradictions fail', async () => {
  const artifact = goodArtifact();
  const report = caseOutput(artifact, 'hierarchy');
  report.page.summary = 'The catalog contains station measurements.';
  assert.equal((await evaluateQualityArtifact(artifact)).passed, true);
  report.conclusion = contract.hierarchyRecords[0].text.replace('10 percent', '11 percent');
  const gate = await evaluateQualityArtifact(artifact);
  assert.equal(gate.passed, false);
  assert.ok(gate.diagnostics.some(({ scope, field }) => scope === 'overview' && field === 'contradictions'));
});

test('overview tuples do not compensate for missing generated section records', async () => {
  const artifact = goodArtifact();
  const report = caseOutput(artifact, 'hierarchy');
  report.page.summary = report.sections[0].keyPoints.join('\n');
  report.sections[0].keyPoints = [];
  const gate = await evaluateQualityArtifact(artifact);
  assert.equal(gate.passed, false);
  assert.ok(gate.diagnostics.some(({ scope, metric, actual }) => scope === 'generated' && metric === 'factRecall' && actual === 0));
});

test('malformed result collections return diagnostics instead of throwing', async () => {
  for (const value of [null, 17, []]) {
    const artifact = goodArtifact();
    artifact.results[0] = value;
    const gate = await evaluateQualityArtifact(artifact);
    assert.equal(gate.passed, false);
    assert.ok(gate.diagnostics.some(({ code }) => code === 'INVALID_RESULT'));
  }
});

test('nonempty but mismatched model identity and backend evidence fail closed', async () => {
  for (const mutate of [
    (artifact) => { artifact.model.revision = 'wrong-revision'; },
    (artifact) => { caseOutput(artifact, 'text').model.id = 'wrong-model'; },
    (artifact) => { artifact.backend.observed = { runtime: 'browser', device: 'webgpu' }; },
    (artifact) => { caseOutput(artifact, 'hierarchy').metadata.backend = { runtime: 'browser', device: 'webgpu' }; },
    (artifact) => { artifact.results[0].backend = { runtime: 'node', device: 'webgpu' }; },
  ]) {
    const artifact = goodArtifact();
    mutate(artifact);
    const result = await evaluateQualityArtifact(artifact);
    assert.equal(result.passed, false);
    assert.ok(result.diagnostics.some(({ code }) => code.includes('MISMATCH')));
  }
});
