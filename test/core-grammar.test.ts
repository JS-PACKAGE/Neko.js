import assert from 'node:assert/strict';
import test from 'node:test';
import { LogitsProcessorList, Tensor, TokenizersBackend, Qwen3VLProcessor } from '@huggingface/transformers';
import { JsonGrammarState, JsonGrammarProcessor, TokenByteTrie, tokenizerByteTrie } from '../src/core/json-grammar.js';
import { compileStructuredSchema, validateStructuredValue, type SchemaValue } from '../src/core/structured.js';
import { PromptTokenCache, renderInferenceChat } from '../src/core/tokenizer.js';
import { NekoError } from '../src/errors.js';

const schema = { type: 'object', additionalProperties: false, required: ['text', 'ids'], properties: {
  text: { type: 'string', minLength: 1 }, ids: { type: 'array', minItems: 1, maxItems: 2, uniqueItems: true, items: { type: 'string', minLength: 1 } },
} } as const;
const encoder = new TextEncoder();

test('schema grammar enforces closed objects, required keys and array/string bounds across every byte split', () => {
  const grammar = compileStructuredSchema(schema).grammar!;
  const valid = '{"ids":["a","b"],"text":"漢字 😀 \\u0061"}';
  const bytes = encoder.encode(valid);
  for (let split = 0; split <= bytes.length; split++) {
    const state = new JsonGrammarState(grammar);
    assert.equal(state.push(bytes.slice(0, split)), true);
    assert.equal(state.push(bytes.slice(split)), true);
    assert.equal(state.complete, true);
  }
  for (const invalid of ['{}', '{"text":"","ids":["a"]}', '{"text":"x","ids":[]}', '{"text":"x","ids":["a","a"]}', '{"text":"x","ids":["a","b","c"]}', '{"text":"x","extra":1}', '{"text":"x","ids":["a"],}', '{"text":"x","ids":["a",]}']) {
    const state = new JsonGrammarState(grammar);
    assert.equal(state.push(encoder.encode(invalid)), false, invalid);
  }
});

test('integer constraints accept integral exponent values and reject fractional values', () => {
  for (const value of [1e21, -1e21]) {
    const state = new JsonGrammarState(compileStructuredSchema({ type: 'integer', const: value }).grammar!);
    assert.equal(state.push(encoder.encode(JSON.stringify(value))), true);
    assert.equal(state.complete, true);
  }
  const fractional = new JsonGrammarState(compileStructuredSchema({ type: 'integer' }).grammar!);
  fractional.push(encoder.encode('1.5'));
  assert.equal(fractional.complete, false);
});

test('finite unique domains and closed objects reject prefixes with no possible continuation', () => {
  const cases = [
    [{ type: 'array', uniqueItems: true, items: { type: 'string', enum: ['p1', 'p2'] } }, '["p1","p1', '["p1","p2"]', '["p1","p2",'],
    [{ type: 'array', uniqueItems: true, items: { type: 'integer', enum: [1, 2] } }, '[1,1', '[1,2]', '[1,2,'],
    [{ type: 'array', uniqueItems: true, items: { type: 'boolean', enum: [true, false] } }, '[true,t', '[true,false]', '[true,false,'],
    [{ type: 'object', additionalProperties: false, properties: { p: { type: 'null' } } }, '{"p":null,"p', '{"p":null}', '{"p":null,'],
  ] as const;
  for (const [schema, duplicate, valid, exhausted] of cases) {
    const grammar = compileStructuredSchema(schema).grammar!;
    assert.equal(new JsonGrammarState(grammar).push(encoder.encode(duplicate)), false, duplicate);
    const complete = new JsonGrammarState(grammar);
    assert.equal(complete.push(encoder.encode(valid)), true, valid);
    assert.equal(complete.complete, true, valid);
    assert.equal(new JsonGrammarState(grammar).push(encoder.encode(exhausted)), false, exhausted);
  }
});

test('ordinary string token masks preserve all UTF-8 boundary classes and reject structural payloads', () => {
  const pieces: [number, Uint8Array][] = [
    [0, encoder.encode('abc漢')], [1, Uint8Array.of(0xc3)], [2, Uint8Array.of(0xa9)], [3, Uint8Array.of(0x80)],
    [4, encoder.encode('"')], [5, encoder.encode('" extra')], [6, encoder.encode('\\u0061')],
    [7, Uint8Array.of(0xed, 0xa0)], [8, Uint8Array.of(0xe0, 0x80)], [9, Uint8Array.of(0xf4, 0x90)], [10, Uint8Array.of(10)],
  ];
  const trie = new TokenByteTrie(pieces);
  for (const [prefix, expected] of [
    [[34, 97], [0, 1, 4, 6]], [[34, 0xc3], [2, 3]], [[34, 0xe1], [2, 3]], [[34, 0xf1], [2, 3]],
    [[34, 0xe0], [2]], [[34, 0xed], [3]], [[34, 0xf0], [2]], [[34, 0xf4], [3]],
  ] as const) {
    const state = new JsonGrammarState({ type: 'string', minLength: 1 });
    assert.equal(state.push(Uint8Array.from(prefix)), true);
    const allowed: number[] = [];
    trie.allowed(state, (id) => allowed.push(id));
    assert.deepEqual(allowed.sort((a, b) => a - b), expected);
  }
});

test('ByteLevel token pieces preserve partial UTF-8 bytes instead of replacement characters', () => {
  const tokenizer = new TokenizersBackend({ model: { type: 'BPE', vocab: { '"': 0, 'Ã': 1, '©': 2, x: 3 }, merges: [] },
    decoder: { type: 'ByteLevel' }, added_tokens: [], normalizer: null, pre_tokenizer: null, post_processor: null }, {});
  const trie = tokenizerByteTrie(tokenizer);
  assert.deepEqual(Array.from(trie.pieces.get(1)!), [0xc3]);
  assert.deepEqual(Array.from(trie.pieces.get(2)!), [0xa9]);
  const processor = new JsonGrammarProcessor(compileStructuredSchema({ type: 'string', enum: ['é'] }).grammar!, trie, 1, [4]);
  const list = new LogitsProcessorList(); list.push(processor);
  const scores = new Tensor('float32', new Float32Array(5).fill(1), [1, 5]);
  try {
    list._call([[99n, 0n]], scores);
    assert.equal(scores.data[1], 1); assert.equal(scores.data[3], -Infinity); assert.equal(scores.data[4], -Infinity);
    (scores.data as Float32Array).fill(1);
    list._call([[99n, 0n, 1n]], scores);
    assert.equal(scores.data[2], 1); assert.equal(scores.data[0], -Infinity);
    (scores.data as Float32Array).fill(1);
    list._call([[99n, 0n, 1n, 2n, 0n]], scores);
    assert.equal(scores.data[4], 1); assert.equal(scores.data[1], -Infinity);
  } finally { scores.dispose(); }
});

test('logits grammar masks same-token trailing payload and premature EOS using the actual callable list API', () => {
  const trie = new TokenByteTrie([[0, encoder.encode('{}')], [1, encoder.encode('{} prose')], [2, encoder.encode('{')], [3, encoder.encode('}')]]);
  const list = new LogitsProcessorList();
  list.push(new JsonGrammarProcessor(compileStructuredSchema({ type: 'object', additionalProperties: false }).grammar!, trie, 1, [4]));
  const scores = new Tensor('float32', new Float32Array(5).fill(1), [1, 5]);
  try {
    list._call([[100n]], scores);
    assert.equal(scores.data[0], 1); assert.equal(scores.data[1], -Infinity); assert.equal(scores.data[3], -Infinity); assert.equal(scores.data[4], -Infinity);
    (scores.data as Float32Array).fill(1); list._call([[100n, 0n]], scores);
    assert.equal(scores.data[4], 1); assert.equal(scores.data[0], -Infinity);
  } finally { scores.dispose(); }
});

test('UTF-8 grammar rejects overlong encodings, surrogates and incompatible enum prefixes before token selection', () => {
  for (const bytes of [[0xe0, 0x80], [0xed, 0xa0], [0xf0, 0x80], [0xf4, 0x90], [0x80]]) {
    const state = new JsonGrammarState(compileStructuredSchema({ type: 'string' }).grammar!);
    assert.equal(state.push(encoder.encode('"')), true); assert.equal(state.push(Uint8Array.from(bytes)), false);
  }
  const state = new JsonGrammarState(compileStructuredSchema({ type: 'string', enum: ['x'] }).grammar!);
  assert.equal(state.push(encoder.encode('"')), true); assert.equal(state.push(Uint8Array.of(0xc3)), false);
  const escaped = new JsonGrammarState(compileStructuredSchema({ type: 'string', enum: ['x'] }).grammar!);
  assert.equal(escaped.push(encoder.encode('"\\u1')), false);
});

test('constrained subset rejects unsupported schemas explicitly; validation-only supports Draft07 and validates schema-derived types', () => {
  for (const unsupported of [{ type: 'string', pattern: '^x$' }, { type: 'object' }, { type: 'number', minimum: 0 }, { anyOf: [{ type: 'string' }] }, false]) {
    assert.throws(() => compileStructuredSchema(unsupported), (error: unknown) => error instanceof NekoError && error.code === 'SCHEMA_UNSUPPORTED');
  }
  const full = compileStructuredSchema({ type: 'string', pattern: '^x$' }, 'validation-only');
  assert.equal(validateStructuredValue(full, 'x'), 'x'); assert.throws(() => validateStructuredValue(full, 'y'));
  const value: SchemaValue<typeof schema> = validateStructuredValue(compileStructuredSchema(schema), { text: 'yes', ids: ['a'] });
  assert.equal(value.text.toUpperCase(), 'YES'); assert.equal(value.ids[0], 'a');
});

test('exact prompt preprocessing caches token tensors without retaining caller-owned buffers and remains bounded', async () => {
  const cache = new PromptTokenCache(); let builds = 0;
  const build = async () => { builds++; return { input_ids: new Tensor('int64', BigInt64Array.from([1n, 2n]), [1, 2]), attention_mask: new Tensor('int64', BigInt64Array.from([1n, 1n]), [1, 2]) }; };
  const first = await cache.inputs(' exact prompt ', build);
  assert.equal(first.reuse.hit, false);
  (first.inputs.input_ids as Tensor).data[0] = 9n;
  const second = await cache.inputs(' exact prompt ', build);
  assert.equal(builds, 1); assert.equal(second.reuse.hit, true); assert.equal(second.reuse.reusedTokens, 2); assert.equal(second.reuse.kvReuse, false);
  assert.equal((second.inputs.input_ids as Tensor).data[0], 1n);
  for (const value of Object.values(first.inputs)) if (value instanceof Tensor) value.dispose();
  for (const value of Object.values(second.inputs)) if (value instanceof Tensor) value.dispose();
  for (let index = 0; index < 20; index++) { const entry = await cache.inputs(`prompt ${index}`, build); for (const value of Object.values(entry.inputs)) if (value instanceof Tensor) value.dispose(); }
  assert.equal(cache.info().entries, 16); cache.clear(); assert.equal(cache.info().entries, 0);
});

test('region marker expansion preserves exact text and ordered multimodal message placement', () => {
  const tokenizer = new TokenizersBackend({ model: { type: 'BPE', vocab: { x: 0 }, merges: [] },
    decoder: { type: 'ByteLevel' }, added_tokens: [], normalizer: null, pre_tokenizer: null, post_processor: null }, {});
  const template = '{% for message in messages %}{{ message.role }}:{% for item in message.content %}{% if item.type == \"image\" %}<|image_pad|>{% else %}{{ item.text }}{% endif %}{% endfor %}\\n{% endfor %}assistant:';
  const processor = new Qwen3VLProcessor({}, { tokenizer }, template);
  const image = { data: new Uint8Array([1, 2, 3]), width: 1, height: 1, channels: 3 as const };
  const messages = [{ role: 'user' as const, content: [{ type: 'text' as const, text: '  before\\t' }, { type: 'image' as const, image }, { type: 'text' as const, text: '\\nafter  ' }] }];
  const original = messages[0]!.content.length;
  const rendered = renderInferenceChat(processor, { messages }, undefined, [2]);
  assert.equal(rendered.text, 'user:  before\\t<|image_pad|><|image_pad|>\\nafter  \\nassistant:');
  assert.equal(rendered.images.length, 1); assert.equal(messages[0]!.content.length, original);
  assert.throws(() => renderInferenceChat(processor, { messages }, undefined, [0]), TypeError);
  assert.throws(() => renderInferenceChat(processor, { messages }, undefined, [65]), TypeError);
});

test('flat token trie exposes exact bytes for sparse IDs, skips empty pieces and shares prefixes without losing siblings', () => {
  const trie = new TokenByteTrie([[900, encoder.encode('ab')], [3, encoder.encode('a')], [7, new Uint8Array()], [12, encoder.encode('ac')], [13, encoder.encode('ab')]]);
  assert.equal(trie.pieces.size, 4);
  assert.deepEqual(Array.from(trie.pieces.get(900)!), [97, 98]);
  assert.deepEqual(Array.from(trie.pieces.get(13)!), [97, 98]);
  for (const missing of [7, 8, 1000, -1]) assert.equal(trie.pieces.get(missing), undefined);
  const state = new JsonGrammarState({ type: 'any' });
  const allowed: number[] = [];
  trie.allowed(state, (id) => allowed.push(id));
  // A top-level value cannot begin with a letter, so no letter-led piece may be offered.
  assert.deepEqual(allowed, []);
  const string = new JsonGrammarState({ type: 'string' });
  string.pushByte(34);
  const inString: number[] = [];
  trie.allowed(string, (id) => inString.push(id));
  assert.deepEqual(inString.sort((a, b) => a - b), [3, 12, 13, 900]);
});
