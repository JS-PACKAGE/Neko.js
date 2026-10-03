import assert from 'node:assert/strict';
import test from 'node:test';
import { defineTool, executeToolCalls, inferTools, type ToolCall, type ToolConversationMessage } from '../src/core/tools.js';
import type { Neko } from '../src/index.js';
import type { StructuredInferOptions, StructuredInferenceResult } from '../src/core/engine.js';
import type { SchemaValue } from '../src/core/structured.js';
import { getRegisteredModelProfile } from '../src/cache/registry.js';
import { NekoError } from '../src/errors.js';

const profile = getRegisteredModelProfile();
const add = defineTool({ name: 'add', description: 'Add two integers.', parameters: {
  type: 'object', properties: { a: { type: 'integer' }, b: { type: 'integer' } }, required: ['a', 'b'], additionalProperties: false,
}, result: { type: 'integer' } });
const echo = defineTool({ name: 'echo', parameters: { type: 'string' } });
const tools = [add, echo] as const;
const valid: ToolCall<typeof tools> = { id: 'call_1', name: 'add', arguments: { a: 2, b: 3 } };
const code = (expected: string) => (error: unknown): boolean => error instanceof NekoError && error.code === expected;

// Schema-derived calls are discriminated by name; arbitrary caller-selected types are unavailable.
// @ts-expect-error add arguments derive integer fields from its schema
const invalidTyped: ToolCall<typeof tools> = { id: 'invalid', name: 'add', arguments: { a: 'two', b: 3 } };
void invalidTyped;

function hostFixture(values: unknown[]) {
  const inputs: StructuredInferOptions[] = [];
  const host: Pick<Neko, 'inferStructured'> = {
    async inferStructured<S>(input: StructuredInferOptions<S>): Promise<StructuredInferenceResult<SchemaValue<S>>> {
      inputs.push(input);
      return {
        text: 'fixture JSON', value: values.shift() as SchemaValue<S>, finishReason: 'stop', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        model: { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype },
        backend: { runtime: 'node', device: 'cpu', executionProviders: ['cpu'], gpuMemoryBytes: null, adapterEvidence: null, supported: true, capabilityEvidence: 'model-runtime-compatibility', sessions: [], providerEvidence: 'loaded-session-configuration' },
        timings: { loadMs: 0, preprocessMs: 0, firstTokenMs: null, generationMs: 0, totalMs: 0 }, memory: { jsHeapBytes: null, gpuBytes: null },
        structured: { mode: 'tokenizer-constrained-runtime-validation', dialect: 'draft-07' },
      };
    },
  };
  return { host, inputs };
}

// Hosts deliberately supply both valid and hostile structured values; no model quality is claimed.
test('model selection and typed result roundtrip keep tool results out of system authority', async () => {
  const fixture = hostFixture([{ message: 'I will add.', calls: [valid] }, { message: 'The sum is 5.', calls: [] }]);
  const messages: ToolConversationMessage<typeof tools>[] = [{ role: 'system', content: 'Be concise.' }, { role: 'user', content: 'Add 2 and 3.' }];
  const selection = await inferTools(fixture.host, { tools, messages });
  assert.deepEqual(selection.toolCalls, [valid]); assert.equal(selection.message.role, 'assistant');
  let approved = 0; let executed = 0;
  const results = await executeToolCalls(tools, selection.toolCalls, { approve: async () => { approved++; return true; }, handlers: {
    add: ({ a, b }) => { executed++; return a + b; }, echo: (text) => text,
  } });
  assert.deepEqual(results, [{ role: 'tool', id: 'call_1', name: 'add', status: 'ok', value: 5 }]);
  const final = await inferTools(fixture.host, { tools, messages: [...messages, selection.message, ...results] });
  assert.deepEqual(final.toolCalls, []); assert.equal(final.value.message, 'The sum is 5.');
  assert.equal(approved, 1); assert.equal(executed, 1);
  const chat = fixture.inputs[1]!.messages!;
  assert.equal(chat[0]!.role, 'system'); assert.equal(chat[chat.length - 1]!.role, 'user');
  assert.match(chat[chat.length - 1]!.content as string, /Untrusted tool result JSON/);
});

test('denied calls are never executed and approval arguments cannot be mutated', async () => {
  let executed = 0;
  const output = await executeToolCalls(tools, [valid], { approve: (call) => {
    assert.equal(Object.isFrozen(call), true); assert.equal(Object.isFrozen(call.arguments), true); return false;
  }, handlers: { add: ({ a, b }) => { executed++; return a + b; }, echo: (text) => text } });
  assert.equal(executed, 0); assert.equal(output[0]!.status, 'denied');
});

test('unknown names, duplicate IDs, schema-invalid arguments and invalid envelopes fail closed', async () => {
  for (const value of [
    { message: 'bad', calls: [{ ...valid, name: 'unknown' }] },
    { message: 'bad', calls: [valid, valid] },
    { message: 'bad', calls: [{ ...valid, arguments: { a: '2', b: 3 } }] },
    { message: 'bad', calls: [{ ...valid, arguments: { a: 2, b: 3, hidden: true } }] },
    { message: 'bad', calls: [{ ...valid, id: 'bad id' }] },
    { message: 'bad', calls: [valid], extra: true },
  ]) {
    await assert.rejects(inferTools(hostFixture([value]).host, { tools, messages: [{ role: 'user', content: 'choose' }] }), code('STRUCTURED_OUTPUT'));
  }
});

test('execution validates the entire call list before any approval or side effect', async () => {
  let approvals = 0;
  await assert.rejects(executeToolCalls(tools, [valid, valid], { approve: () => { approvals++; return true; }, handlers: { add: ({ a, b }) => a + b, echo: (text) => text } }), code('STRUCTURED_OUTPUT'));
  assert.equal(approvals, 0);
});

test('conversation IDs and results cannot be forged, duplicated, omitted or reused', async () => {
  const messages: ToolConversationMessage<typeof tools>[] = [{ role: 'user', content: 'add' }, { role: 'assistant', content: 'calling', toolCalls: [valid] }];
  const result = { role: 'tool', id: valid.id, name: 'add', status: 'ok', value: 5 } as const;
  for (const history of [messages, [...messages, { ...result, id: 'unknown' }], [...messages, result, result], [...messages, { ...result, value: 'not integer' }]]) {
    const fixture = hostFixture([]);
    await assert.rejects(inferTools(fixture.host, { tools, messages: history as ToolConversationMessage<typeof tools>[] })); assert.equal(fixture.inputs.length, 0);
  }
  await assert.rejects(inferTools(hostFixture([{ message: 'reuse', calls: [valid] }]).host, { tools, messages: [...messages, result] }), code('STRUCTURED_OUTPUT'));
});

test('approval and handler failures produce separate safe results; output schema is validated', async () => {
  const calls: ToolCall<typeof tools>[] = [valid, { ...valid, id: 'call_2' }, { id: 'call_3', name: 'echo', arguments: 'hello' }];
  const output = await executeToolCalls(tools, calls, {
    approve: (call) => { if (call.id === 'call_1') throw new Error('private application secret'); return true; },
    handlers: { add: () => { throw new Error('private handler secret'); }, echo: (text) => text },
  });
  assert.deepEqual(output.map((item) => item.status), ['error', 'error', 'ok']);
  assert.doesNotMatch(JSON.stringify(output), /private/);
  const wrongOutput = await executeToolCalls(tools, [valid], { approve: () => true, handlers: {
    // @ts-expect-error result schema derives an integer return type
    add: () => 'invalid result', echo: (text) => text,
  } });
  assert.equal(wrongOutput[0]!.status, 'error');
});

test('cancelled approval cannot run a handler and running handlers retain ownership until completion', async () => {
  const abort = new AbortController(); const entered = Promise.withResolvers<void>(); const approval = Promise.withResolvers<boolean>(); let executed = 0;
  const output = executeToolCalls(tools, [valid], { signal: abort.signal, approve: () => { entered.resolve(); return approval.promise; }, handlers: {
    add: () => { executed++; return 5; }, echo: (text) => text,
  } });
  await entered.promise; abort.abort(); const cancelled = await output; approval.resolve(true);
  assert.equal(cancelled[0]!.status, 'cancelled'); assert.equal(executed, 0);
  const activeAbort = new AbortController(); const started = Promise.withResolvers<void>(); const release = Promise.withResolvers<number>(); let settled = false;
  const active = executeToolCalls(tools, [valid], { signal: activeAbort.signal, approve: () => true, handlers: {
    add: async (_args, { signal }) => { assert.strictEqual(signal, activeAbort.signal); started.resolve(); return release.promise; }, echo: (text) => text,
  } }).then((value) => { settled = true; return value; });
  await started.promise; activeAbort.abort(); await Promise.resolve(); assert.equal(settled, false);
  release.resolve(5); assert.equal((await active)[0]!.status, 'cancelled');
});

test('tool schemas are snapshotted and unsupported grammar keywords require explicit validation-only', () => {
  const parameters = { type: 'object', properties: { value: { type: 'string' } }, required: ['value'], additionalProperties: false } as const;
  const definition = defineTool({ name: 'snapshot', parameters }); assert.notStrictEqual(definition.parameters, parameters); assert.equal(Object.isFrozen(definition.parameters.properties), true);
  assert.throws(() => defineTool({ name: 'bounded', parameters: { type: 'integer', minimum: 0 } }), code('SCHEMA_UNSUPPORTED'));
  assert.doesNotThrow(() => defineTool({ name: 'bounded', parameters: { type: 'integer', minimum: 0 }, structuredMode: 'validation-only' }));
  assert.throws(() => defineTool({ name: '__proto__', parameters: { type: 'string' } }), code('INVALID_INPUT'));
});
