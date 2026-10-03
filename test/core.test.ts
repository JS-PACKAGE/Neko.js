import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Tensor } from '@huggingface/transformers';
import { generationSettings, NucleusProcessor, StopBuffer } from '../src/core/generation.js';
import { compileStructuredSchema } from '../src/core/structured.js';
import { NekoError } from '../src/errors.js';
import { JsonBoundary } from '../src/core/json-boundary.js';
import { inferenceChat, validateInferenceOptions } from '../src/core/preflight.js';

test('stops crossing chunks never enter the delivered output', () => {
  let output = '';
  const buffer = new StopBuffer(['END', 'END OF'], (text) => { output += text; });
  buffer.push('safe E');
  assert.equal(output, 'safe ');
  buffer.push('N');
  assert.equal(output, 'safe ');
  buffer.push('D hidden');
  buffer.push('more hidden');
  buffer.end();
  assert.equal(output, 'safe ');
  assert.equal(buffer.stopped, true);
});

test('partial stop prefixes are restored on ordinary completion', () => {
  let output = '';
  const buffer = new StopBuffer(['STOP'], (text) => { output += text; });
  buffer.push('before ST');
  buffer.push('ill here S');
  buffer.end();
  assert.equal(output, 'before STill here S');
  assert.equal(buffer.stopped, false);
});

test('nucleus filtering uses temperature and the top-K sampling distribution', () => {
  const tensor = new Tensor('float32', new Float32Array([3, 2, 1, 0]), [1, 4]);
  try {
    new NucleusProcessor(0.7, 1, 2)._call([], tensor);
    assert.deepEqual(Array.from(tensor.data), [3, -Infinity, -Infinity, -Infinity]);
    const warm = new Tensor('float32', new Float32Array([3, 2, 1, 0]), [1, 4]);
    try {
      new NucleusProcessor(0.7, 5, 2)._call([], warm);
      assert.deepEqual(Array.from(warm.data), [3, 2, -Infinity, -Infinity]);
    } finally { warm.dispose(); }
  } finally { tensor.dispose(); }
});

test('generation bounds reject unsupported sampling and invalid stop IDs', () => {
  assert.throws(() => generationSettings({ topP: 0.9 }, 100), TypeError);
  assert.throws(() => generationSettings({ sampling: true, temperature: 0 }, 100), RangeError);
  assert.throws(() => generationSettings({ stop: [''] }, 100), TypeError);
  assert.throws(() => generationSettings({ stopTokenIds: [100] }, 100), TypeError);
});

test('Draft-07 local references validate data without mutating caller schemas', () => {
  const schema = { definitions: { number: { type: 'integer' } }, type: 'object', properties: { value: { $ref: '#/definitions/number' } }, required: ['value'], additionalProperties: false };
  const before = JSON.stringify(schema);
  const { validator } = compileStructuredSchema(schema, 'validation-only');
  assert.equal(validator.validate({ value: 2 }).valid, true);
  assert.equal(validator.validate({ value: 'two' }).valid, false);
  assert.equal(validator.validate({ value: 2, extra: true }).valid, false);
  assert.equal(JSON.stringify(schema), before);
  assert.equal(Object.hasOwn(schema, '__absolute_uri__'), false);
});

test('invalid schemas and unresolved/remote references fail before generation', () => {
  for (const schema of [{ type: 'unknown' }, { minItems: -1 }, { required: ['x', 'x'] }, { pattern: '[' }, { $ref: '#/missing' }, { $ref: 'https://example.com/schema' }, { $schema: 'https://json-schema.org/draft/2020-12/schema' }, { prefixItems: [] }]) {
    assert.throws(() => compileStructuredSchema(schema), (error: unknown) => error instanceof NekoError && error.code === 'SCHEMA_INVALID');
  }
});

test('JSON boundaries handle nested containers, escaped strings and primitive delimiters incrementally', () => {
  for (const text of [' {"a":[1,{"b":"quote: \\" and brace }"}]}', '"escaped \\" quote"', 'true', 'false', 'null', '-12.4e+2\n']) {
    const boundary = new JsonBoundary();
    for (const character of text) boundary.push(character);
    assert.equal(boundary.invalid, false, text);
    assert.equal(boundary.complete, true, text);
    assert.doesNotThrow(() => JSON.parse(text));
  }
  const number = new JsonBoundary();
  number.push('1'); assert.equal(number.complete, false);
  number.push('2e'); assert.equal(number.complete, false);
  number.push('+3'); assert.equal(number.complete, false);
  number.end(); assert.equal(number.complete, true);
  for (const text of ['{} trailing', 'truefalse', '\"x\"{}', '[}', '```json\\n{}']) {
    const boundary = new JsonBoundary(); boundary.push(text);
    assert.equal(boundary.invalid, true, text);
  }
  for (const text of ['{\"x\":', '[1', '\"unterminated', 'tru']) {
    const boundary = new JsonBoundary(); boundary.push(text); boundary.end();
    assert.equal(boundary.complete, false, text);
  }
});

test('cheap preflight preserves exact source text and permits non-fitting planning budgets', () => {
  const prompt = '\\n  exact prompt\\t ';
  validateInferenceOptions({ prompt });
  assert.deepEqual(inferenceChat({ prompt }).rendered, [{ role: 'user', content: [{ type: 'text', text: prompt }] }]);
  assert.throws(() => validateInferenceOptions({ prompt, contextWindowTokens: 32, maxNewTokens: 32 }), (error: unknown) => error instanceof NekoError && error.code === 'CONTEXT_LIMIT');
  assert.doesNotThrow(() => validateInferenceOptions({ prompt, contextWindowTokens: 32, maxNewTokens: 32 }, true));
});
