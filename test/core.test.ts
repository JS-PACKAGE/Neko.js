import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Tensor } from '@huggingface/transformers';
import { generationSettings, NucleusProcessor, StopBuffer } from '../src/core/generation.js';
import { compileStructuredSchema } from '../src/core/structured.js';
import { NekoError } from '../src/errors.js';

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
  const { validator } = compileStructuredSchema(schema);
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
