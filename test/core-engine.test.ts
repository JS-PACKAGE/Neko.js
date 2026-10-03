import assert from 'node:assert/strict';
import test from 'node:test';
import { Tensor } from '@huggingface/transformers';
import { VisionEngine } from '../src/core/engine.js';
import { getRegisteredModelProfile } from '../src/cache/registry.js';
import { NekoError } from '../src/errors.js';

// A deterministic token source isolates generation control flow from model quality/downloads.
function fixture(chunks: string[], eos = true) {
  let generated = 0;
  let calls = 0;
  const encoded = (text: string) => Array.from(text, (character) => BigInt(character.codePointAt(0)!));
  const processor = Object.assign((text: string) => ({ input_ids: new Tensor('int64', BigInt64Array.from(encoded(text)), [1, text.length]) }), {
    apply_chat_template: (messages: { role: string; content: { text?: string }[] }[]) => messages.map((message) => `${message.role}:${message.content.map((part) => part.text ?? '').join('')}\n`).join('') + 'assistant:',
    tokenizer: { all_special_ids: [1000], encode: encoded, decode: (ids: bigint[]) => ids.filter((id) => id !== 1000n).map((id) => String.fromCodePoint(Number(id))).join('') },
  });
  const model = {
    sessions: { decoder: { config: { device: 'cpu', dtype: 'q4' } } },
    generation_config: { eos_token_id: 1000 },
    async generate(options: { input_ids: Tensor; max_new_tokens: number; streamer: { put(ids: bigint[][]): void; end(): void }; stopping_criteria: { interrupted: boolean } }) {
      calls++;
      const ids = Array.from(options.input_ids.data, BigInt);
      options.streamer.put([ids]);
      let count = 0;
      for (const chunk of chunks) {
        if (options.stopping_criteria.interrupted || count >= options.max_new_tokens) break;
        generated++;
        const tokens = encoded(chunk).slice(0, options.max_new_tokens - count);
        ids.push(...tokens); count += tokens.length;
        options.streamer.put([tokens]);
      }
      if (eos && !options.stopping_criteria.interrupted && count < options.max_new_tokens) { ids.push(1000n); options.streamer.put([[1000n]]); }
      options.streamer.end();
      return new Tensor('int64', BigInt64Array.from(ids), [1, ids.length]);
    },
    async dispose() {},
  };
  const profile = getRegisteredModelProfile();
  const Construct = VisionEngine as unknown as new (...args: unknown[]) => VisionEngine;
  const engine = new Construct(model, processor, { runtime: 'node', device: 'cpu', executionProviders: ['cpu'] }, 0, false, 4096, 2000,
    { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype }, undefined, true);
  return { engine, generated: () => generated, calls: () => calls };
}

test('structured inference stops raw token generation at one JSON boundary and streams exactly that text', async () => {
  for (const chunks of [['{', '"a":[', '"escaped \\\" }",', 'true]}', ' prose'], ['"quoted ', '\\\" text"', ' trailing'], ['t', 'ru', 'e', ' false'], ['n', 'ull', '{}'], ['1', '2', 'e', '+3', '\n', ' trailing']]) {
    const { engine, generated } = fixture(chunks);
    let streamed = '';
    try {
      const result = await engine.inferStructured({ prompt: 'Return JSON.', schema: true, structuredMode: 'validation-only', maxNewTokens: 128, onToken: (text) => { streamed += text; } });
      assert.equal(result.text, chunks.slice(0, -1).join(''));
      assert.equal(streamed, result.text);
      assert.deepEqual(result.value, JSON.parse(result.text));
      assert.equal(generated(), chunks.length - 1);
      assert.equal(result.finishReason, 'stop');
    } finally { await engine.dispose(); }
  }
});

test('structured output rejects same-token trailing payload, malformed/truncated text and schema violations', async () => {
  for (const [chunks, schema, eos] of [
    [['{} prose'], true, true], [['truefalse'], true, true], [['{"x":}'], true, true],
    [['```json\n{}\n```'], true, true], [['{"x":'], true, true], [['1e'], true, true],
    [['{}'], { type: 'object', required: ['x'] }, true], [['1'], true, false], [[], true, true],
  ] as [string[], unknown, boolean][]) {
    const { engine } = fixture(chunks, eos);
    try {
      await assert.rejects(engine.inferStructured({ prompt: 'Return JSON.', schema, structuredMode: 'validation-only', maxNewTokens: 128 }), (error: unknown) => error instanceof NekoError && error.code === 'STRUCTURED_OUTPUT');
    } finally { await engine.dispose(); }
  }
});

test('planning uses the same chat/schema preprocessing without generation and reports non-fitting requests', async () => {
  const { engine, calls } = fixture(['{}']);
  try {
    const options = { prompt: '\n  preserve exactly\t', schema: { type: 'object' }, structuredMode: 'validation-only' as const, maxNewTokens: 16 };
    const plan = await engine.planInference(options);
    assert.equal(calls(), 0);
    const result = await engine.inferStructured(options);
    assert.equal(plan.inputTokens, result.usage.inputTokens);
    assert.equal(plan.maxNewTokens, 16);
    assert.equal(plan.contextLimit, 4096);
    assert.equal(plan.availableOutputTokens, 4096 - plan.inputTokens);
    const overflow = await engine.planInference({ prompt: options.prompt, maxNewTokens: 32, contextWindowTokens: 32 });
    assert.equal(overflow.fits, false);
    assert.equal(calls(), 1);
  } finally { await engine.dispose(); }
});
