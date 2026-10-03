import { isDeepStrictEqual } from 'node:util';
import { MODEL_ID, MODEL_REVISION } from '../../dist/src/cache/manifest.js';
import { ARTIFACT_VERSION, CASE_IDS, FIXTURE_VERSION, loadQualityContract } from './contract.mjs';
import { EVALUATOR_VERSION, evaluateClaims } from './evaluate.mjs';
import { evaluateHierarchy } from './hierarchy.mjs';
import { POLICY_VERSION, QUALITY_POLICY } from './policy.mjs';

/** Recompute scores from raw SDK results; serialized metrics are never trusted. */
export async function evaluateQualityArtifact(artifact, { requiredCases = QUALITY_POLICY.requiredCases } = {}) {
  const diagnostics = [];
  const add = (code, message, fields = {}) => diagnostics.push({ code, message, ...fields });
  const scoredCases = [];
  const completion = { completed: false, comparable: false, caseCount: Array.isArray(artifact?.results) ? artifact.results.length : 0 };
  const finish = () => ({ schemaVersion: 1, policyVersion: POLICY_VERSION, evaluatorVersion: EVALUATOR_VERSION, fixtureVersion: FIXTURE_VERSION, passed: diagnostics.length === 0, completion, quality: { passed: diagnostics.length === 0, cases: scoredCases }, diagnostics });
  if (!artifact || typeof artifact !== 'object' || Array.isArray(artifact)) {
    add('INVALID_ARTIFACT', 'Expected a benchmark artifact object.');
    return finish();
  }
  for (const [key, expected] of Object.entries({ schemaVersion: ARTIFACT_VERSION, fixtureVersion: FIXTURE_VERSION, evaluatorVersion: EVALUATOR_VERSION, policyVersion: POLICY_VERSION })) {
    if (artifact[key] !== expected) add('VERSION_MISMATCH', `Expected ${key}=${expected}.`, { field: key, actual: artifact[key] ?? null, expected });
  }
  if (!Array.isArray(requiredCases) || !requiredCases.length || requiredCases.some((id) => !CASE_IDS.includes(id)) || new Set(requiredCases).size !== requiredCases.length) {
    add('INVALID_REQUIRED_CASES', 'Required cases must be unique known fixture IDs.');
    return finish();
  }
  if (artifact.status !== 'completed' || artifact.comparable !== true || artifact.runnerError || artifact.networkPolicyViolation) add('INCOMPLETE_RUN', 'Run must complete, be comparable, and have no runner/network failure.');
  const config = artifact.config;
  if (!config || !['node', 'browser'].includes(config.runtime) || !['cpu', 'webgpu'].includes(config.device) || !['default', 'all-q4'].includes(config.modelProfile)) add('INVALID_CONFIG', 'Runtime, device and profile evidence must be present.');
  if (!artifact.model?.id || !artifact.model?.revision || !artifact.backend?.observed) add('MISSING_RUNTIME_EVIDENCE', 'Model identity/revision and observed backend are required.');
  if (artifact.model?.id !== MODEL_ID || artifact.model?.revision !== MODEL_REVISION) add('MODEL_IDENTITY_MISMATCH', 'Artifact must report the pinned model ID and revision.');
  if (artifact.backend?.observed?.runtime !== config?.runtime || artifact.backend?.observed?.device !== config?.device) add('BACKEND_CONFIG_MISMATCH', 'Observed run backend runtime/device must match the declared configuration.');
  let contract;
  try { contract = await loadQualityContract(config ?? {}); }
  catch (error) { add('INVALID_STIMULUS', error.message); return finish(); }
  if (contract.fixtureMismatch) add('LOCAL_FIXTURE_INTEGRITY', 'Committed fixtures do not match their oracle.');
  for (const [key, expected] of Object.entries({ oracleSha256: contract.oracleSha256, stimulusKey: contract.stimulusKey, fixtures: contract.fixtures, caseInputs: contract.caseInputs })) {
    if (!isDeepStrictEqual(artifact[key], expected)) add('STIMULUS_MISMATCH', `Artifact ${key} does not match current fixtures and declared configuration.`, { field: key });
  }
  const results = Array.isArray(artifact.results) ? artifact.results : [];
  if (results.some((entry) => !entry || typeof entry !== 'object' || Array.isArray(entry))) {
    add('INVALID_RESULT', 'Every case result must be an object.');
    return finish();
  }
  const expectedCases = contract.cases.map(({ id }) => id);
  if (!isDeepStrictEqual(results.map(({ id }) => id), expectedCases)) add('CASE_SET_MISMATCH', 'Exactly one result per selected fixture, in declared order, is required.');
  for (const id of requiredCases) if (!results.some((entry) => entry.id === id)) add('MISSING_REQUIRED_CASE', `Missing required fixture ${id}.`, { caseId: id });
  completion.completed = artifact.status === 'completed' && results.length > 0 && results.every((entry) => entry.status === 'completed' && entry.attempted === true);
  completion.comparable = completion.completed && artifact.comparable === true && !artifact.networkPolicyViolation;
  const checkScore = (caseId, scope, score, policy) => {
    for (const [metric, threshold] of [['claimPrecision', policy.minPrecision], ['factRecall', policy.minRecall]]) {
      if (threshold === undefined) continue;
      const value = score.metrics[metric];
      if (!Number.isFinite(value) || value < threshold) add('QUALITY_THRESHOLD', `${scope}.${metric} must be >= ${threshold}.`, { caseId, scope, metric, actual: value, minimum: threshold });
    }
    for (const [field, maximum] of [['contradictions', policy.maxContradictions], ['unsupported', policy.maxUnsupported], ['formatErrors', policy.maxFormatErrors]]) {
      const entries = field === 'formatErrors' ? score.formatCompliance.errors : score[field];
      if (entries.length > maximum) add('QUALITY_FINDINGS', `${scope}.${field} exceeds ${maximum}.`, { caseId, scope, field, count: entries.length, findings: entries });
    }
  };
  for (const entry of results) {
    const testCase = contract.cases.find(({ id }) => id === entry.id);
    if (!testCase) continue;
    if (entry.status !== 'completed' || entry.attempted !== true || entry.fixtureId !== testCase.fixtureId || entry.kind !== testCase.kind || entry.observedProfile !== config?.modelProfile || entry.expectedProfile !== config?.modelProfile || !entry.output || !entry.backend) {
      add('INVALID_CASE_EVIDENCE', 'Case must have a completed raw output, matching fixture/kind/profile, and backend evidence.', { caseId: entry.id });
      continue;
    }
    const rawModel = entry.id === 'hierarchy' ? entry.output.metadata?.model : entry.output.model;
    const rawBackend = entry.id === 'hierarchy' ? entry.output.metadata?.backend : entry.output.backend;
    if (rawModel?.profile !== config?.modelProfile) add('OUTPUT_PROFILE_MISMATCH', 'Raw SDK output profile does not match declared profile.', { caseId: entry.id });
    if (rawModel?.id !== MODEL_ID || rawModel?.revision !== MODEL_REVISION) add('OUTPUT_MODEL_MISMATCH', 'Raw SDK output must identify the pinned model and revision.', { caseId: entry.id });
    if (!isDeepStrictEqual(entry.backend, rawBackend) || rawBackend?.runtime !== config?.runtime || rawBackend?.device !== config?.device) add('OUTPUT_BACKEND_MISMATCH', 'Raw SDK backend and case backend must agree with declared runtime/device.', { caseId: entry.id });
    let evaluation;
    if (entry.id === 'hierarchy') {
      evaluation = evaluateHierarchy(entry.output, contract.hierarchyRecords);
      if (!evaluation.reportVersionValid) add('REPORT_VERSION_MISMATCH', 'Hierarchy requires report schemaVersion 2.', { caseId: entry.id });
      checkScore(entry.id, 'retained', evaluation.retained, QUALITY_POLICY.hierarchyRetained);
      checkScore(entry.id, 'generated', evaluation.generated, QUALITY_POLICY.hierarchyGenerated);
      checkScore(entry.id, 'overview', evaluation.overview, QUALITY_POLICY.hierarchyOverview);
    } else if (typeof entry.output.text !== 'string') add('MISSING_RAW_TEXT', 'Raw inference text is required.', { caseId: entry.id });
    else {
      evaluation = evaluateClaims(entry.output.text, contract.manifest[entry.id]);
      checkScore(entry.id, 'claims', evaluation, QUALITY_POLICY.claims);
    }
    if (evaluation) scoredCases.push({ id: entry.id, evaluation });
  }
  return finish();
}

/** Quality score comparisons may cross backend/profile only via explicit paired mode. */
export async function compareQualityArtifacts(left, right, { mode = 'strict' } = {}) {
  const [leftGate, rightGate] = await Promise.all([evaluateQualityArtifact(left), evaluateQualityArtifact(right)]);
  const diagnostics = [];
  const add = (field) => diagnostics.push({ code: 'NONCOMPARABLE', field });
  if (!['strict', 'paired-inputs'].includes(mode)) add('mode');
  for (const key of ['schemaVersion', 'fixtureVersion', 'evaluatorVersion', 'policyVersion', 'oracleSha256', 'stimulusKey', 'caseInputs']) if (!isDeepStrictEqual(left?.[key], right?.[key])) add(key);
  if (!leftGate.completion.comparable || !rightGate.completion.comparable) add('completion');
  const structuralCodes = new Set(['QUALITY_THRESHOLD', 'QUALITY_FINDINGS']);
  if (leftGate.diagnostics.some(({ code }) => !structuralCodes.has(code)) || rightGate.diagnostics.some(({ code }) => !structuralCodes.has(code))) add('artifact-integrity');
  if (!isDeepStrictEqual(left?.model, right?.model)) add('model');
  if (mode === 'strict') {
    for (const key of ['runtime', 'device', 'modelProfile']) if (!isDeepStrictEqual(left?.config?.[key], right?.config?.[key])) add(`config.${key}`);
    if (!isDeepStrictEqual(left?.backend?.observed, right?.backend?.observed)) add('backend.observed');
    if (!isDeepStrictEqual(left?.environment, right?.environment)) add('environment');
    if (!isDeepStrictEqual(left?.browser, right?.browser)) add('browser');
  }
  return { schemaVersion: 1, mode, comparable: diagnostics.length === 0, passed: diagnostics.length === 0 && leftGate.passed && rightGate.passed, caveat: mode === 'paired-inputs' ? 'Paired encoded inputs only; runtime, profile, backend and decoded image representations may differ. This is not an isolated quantization/platform experiment.' : null, diagnostics, left: leftGate, right: rightGate };
}
