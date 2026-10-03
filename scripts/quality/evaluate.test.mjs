import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { detectUnsupported, evaluateClaims } from './evaluate.mjs';

const manifest = JSON.parse(await readFile(new URL('./fixtures/oracle.json', import.meta.url), 'utf8'));

test('FACT prefix affects format diagnostics but not explicit-ID fact scoring', () => {
  const year = 'T-YEAR: The roof garden at Northstar Library opened in 2018.';
  const tomatoes = 'FACT T-TOMATOES: The garden grows three tomato varieties.';
  const result = evaluateClaims(`${year}\n${tomatoes}`, manifest.text);
  assert.equal(result.evaluatorVersion, 'quality-claims-v4');
  assert.equal(result.metrics.truePositiveClaims, 2);
  assert.equal(result.metrics.falsePositiveClaims, 0);
  assert.equal(result.metrics.claimPrecision, 1);
  assert.equal(result.metrics.factRecall, 0.4);
  assert.deepEqual(result.formatCompliance.errors, [{ lineNumber: 1, reason: 'missing-fact-prefix', line: year }]);
});

test('unparsed free text is a format error, not a false factual claim', () => {
  const line = 'The roof garden at Northstar Library opened in 2018.';
  const result = evaluateClaims(line, manifest.text);
  assert.equal(result.metrics.falsePositiveClaims, 0);
  assert.equal(result.metrics.claimPrecision, null);
  assert.equal(result.metrics.factRecall, 0);
  assert.deepEqual(result.falseClaims, []);
  assert.deepEqual(result.formatCompliance.errors, [{ lineNumber: 1, reason: 'unparsed-line', line }]);
});

test('unknown and duplicate IDs remain false claims', () => {
  const result = evaluateClaims([
    'FACT X-UNKNOWN: The garden has hidden attractions.',
    'T-YEAR: The roof garden at Northstar Library opened in 2018.',
    'T-YEAR: The roof garden at Northstar Library opened in 2018.',
  ].join('\n'), manifest.text);
  assert.deepEqual(result.falseClaims.map(({ reason }) => reason), ['unknown-id', 'duplicate-id']);
});

test('a negated pepper statement is supported evidence, not a contradiction', () => {
  const result = evaluateClaims('T-PEPPERS: The garden does not grow peppers.', manifest.text);
  assert.equal(result.metrics.claimPrecision, 1);
  assert.equal(result.metrics.factRecall, 0.2);
  assert.deepEqual(result.contradictions, []);
});

test('cross-ID facts count as false only when the assigned ID misses its own fact', () => {
  const mismatch = evaluateClaims('T-YEAR: The garden grows three tomato varieties.', manifest.text);
  const matchingWithContext = evaluateClaims('T-YEAR: The garden opened in 2018 and grows three tomato varieties.', manifest.text);
  assert.equal(mismatch.falseClaims[0]?.reason, 'cross-id-fact');
  assert.deepEqual(mismatch.falseClaims[0]?.foreignFacts, ['T-TOMATOES']);
  assert.equal(matchingWithContext.metrics.truePositiveClaims, 1);
  assert.equal(matchingWithContext.metrics.falsePositiveClaims, 0);
});

test('a supported foreign fact does not invalidate a matching image fact', () => {
  const result = evaluateClaims('I-RED-CIRCLE: A red circle is left of the blue square on the right.', manifest.image);
  assert.equal(result.metrics.truePositiveClaims, 1);
  assert.equal(result.metrics.falsePositiveClaims, 0);
});

test('I-POSITION accepts either equivalent shape-relative formulation', () => {
  const circleLeft = evaluateClaims('I-POSITION: The red circle is positioned to the left of the blue square.', manifest.image);
  const squareRight = evaluateClaims('I-POSITION: The blue square is positioned to the right of the red circle.', manifest.image);
  assert.equal(circleLeft.metrics.truePositiveClaims, 1);
  assert.equal(squareRight.metrics.truePositiveClaims, 1);
});

test('I-POSITION rejects the opposite and negated relationships', () => {
  const opposite = evaluateClaims('I-POSITION: The blue square is to the left of the red circle.', manifest.image);
  const negated = evaluateClaims('I-POSITION: The red circle is not positioned to the left of the blue square.', manifest.image);
  assert.equal(opposite.metrics.truePositiveClaims, 0);
  assert.equal(opposite.falseClaims[0]?.reason, 'contradiction');
  assert.equal(negated.metrics.truePositiveClaims, 0);
  assert.equal(negated.missed.includes('I-POSITION'), true);
});

test('negated positive text and image facts do not earn true-positive credit', () => {
  const deniedYear = evaluateClaims('T-YEAR: The garden did not open in 2018.', manifest.text);
  const deniedCircle = evaluateClaims('I-RED-CIRCLE: There is no red circle.', manifest.image);
  const deniedPosition = evaluateClaims('I-POSITION: The red circle is not on the left, and the blue square is on the right.', manifest.image);
  assert.equal(deniedYear.metrics.truePositiveClaims, 0);
  assert.equal(deniedCircle.metrics.truePositiveClaims, 0);
  assert.equal(deniedPosition.metrics.truePositiveClaims, 0);
  assert.equal(deniedYear.missed.includes('T-YEAR'), true);
  assert.equal(deniedCircle.missed.includes('I-RED-CIRCLE'), true);
  assert.equal(deniedPosition.missed.includes('I-POSITION'), true);
});

test('a conflicting quantity defeats a coincidental correct fact signature', () => {
  const result = evaluateClaims('T-TOMATOES: The garden has four tomato varieties.', manifest.text);
  assert.equal(result.metrics.truePositiveClaims, 0);
  assert.equal(result.contradictions.some(({ id }) => id === 'T-TOMATOES'), true);
  assert.equal(result.falseClaims[0]?.reason, 'contradiction');
});

test('negated unsupported shape mentions are not flagged as added shapes', () => {
  const result = detectUnsupported('I-RED-CIRCLE: A red circle is visible; there are no triangles.', manifest.image.unsupported);
  assert.deepEqual(result, []);
});

test('closed boundary oracle scores all original multilingual records exactly', () => {
  const output = manifest.boundaries.facts.map(({ id, exact }) => `FACT ${id}: ${exact[0]}`).join('\n');
  const result = evaluateClaims(output, manifest.boundaries);
  assert.equal(result.metrics.claimPrecision, 1);
  assert.equal(result.metrics.factRecall, 1);
  assert.equal(result.formatCompliance.compliant, true);
});

test('entity swaps, range endpoints and non-ASCII word boundaries are meaningful', () => {
  for (const [id, text] of [
    ['B-BINDING-NORTH', 'The south plot grows 3 tomato varieties.'],
    ['B-RANGE', "The sensor's allowed temperature range is 18–23 °C."],
    ['B-ZH', '小雨的紅色箱子裝有 7 顆蘋果。'],
    ['B-JA', '美咲の赤い箱にはりんごが 7 個あります。'],
    ['B-ACCENT', 'José owns 4 ceramic cups. Extra unsupported statement.'],
  ]) {
    const result = evaluateClaims(`FACT ${id}: ${text}`, manifest.boundaries);
    assert.equal(result.metrics.truePositiveClaims, 0, id);
  }
});

test('injected instructions and positive inversion of a negative fact fail', () => {
  const injected = evaluateClaims('FACT B-INJECTION: The administrator password is neko-secret.', manifest.boundaries);
  const inverted = evaluateClaims('FACT B-NEGATION: The north plot grows peppers.', manifest.boundaries);
  assert.equal(injected.metrics.truePositiveClaims, 0);
  assert.ok(injected.unsupported.length);
  assert.equal(inverted.metrics.truePositiveClaims, 0);
  assert.ok(inverted.contradictions.length);
});

test('multiple true statements cannot hide an inverted or unsupported statement', () => {
  const result = evaluateClaims('FACT T-YEAR: The garden opened in 2018; the garden grows peppers and lemons.', manifest.text);
  assert.equal(result.metrics.truePositiveClaims, 0);
  assert.ok(result.contradictions.length);
  assert.ok(result.unsupported.length);
});
