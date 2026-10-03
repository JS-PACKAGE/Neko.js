import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { compareQualityArtifacts, evaluateQualityArtifact } from './gate.mjs';
import { POLICY_VERSION } from './policy.mjs';

let outputPath;
let result;
try {
  const { values } = parseArgs({ options: {
    input: { type: 'string' }, output: { type: 'string' },
    baseline: { type: 'string' }, 'comparison-mode': { type: 'string', default: 'strict' },
    cases: { type: 'string' }, help: { type: 'boolean', default: false },
  } });
  if (values.help) {
    console.log('Usage: npm run quality:gate -- --input artifact.json [--output diagnostics.json] [--cases text,image,boundaries,hierarchy] [--baseline artifact.json --comparison-mode strict|paired-inputs]\nDefault gate requires all cases. --cases is a scoped diagnostic gate, not full release acceptance. Comparison requires both artifacts to pass quality; paired-inputs explicitly permits runtime/profile differences, not model/input differences.');
  } else {
    outputPath = values.output && resolve(values.output);
    if (!values.input) throw new TypeError('--input is required');
    if (values.baseline && values.cases) throw new TypeError('--cases is not supported with --baseline');
    const artifact = JSON.parse(await readFile(resolve(values.input), 'utf8'));
    result = values.baseline
      ? await compareQualityArtifacts(JSON.parse(await readFile(resolve(values.baseline), 'utf8')), artifact, { mode: values['comparison-mode'] })
      : await evaluateQualityArtifact(artifact, values.cases ? { requiredCases: values.cases.split(',') } : undefined);
  }
} catch (error) {
  result = { schemaVersion: 1, policyVersion: POLICY_VERSION, passed: false, diagnostics: [{ code: 'GATE_INPUT_ERROR', message: error instanceof Error ? error.message : String(error) }] };
}
if (result) {
  const json = `${JSON.stringify(result, null, 2)}\n`;
  if (outputPath) {
    try { await mkdir(dirname(outputPath), { recursive: true }); await writeFile(outputPath, json); }
    catch (error) { result.passed = false; result.diagnostics.push({ code: 'GATE_OUTPUT_ERROR', message: error.message }); }
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.passed) process.exitCode = 1;
}
