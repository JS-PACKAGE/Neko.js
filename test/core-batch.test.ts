import assert from 'node:assert/strict';
import test from 'node:test';
import { getEventListeners } from 'node:events';
import { Tensor, TokenizersBackend } from '@huggingface/transformers';
import { VisionEngine, type InferenceBudget, type StructuredBatchRequest, type StructuredInferenceResult } from '../src/core/engine.js';
import { getRegisteredModelProfile } from '../src/cache/registry.js';
import { getGenerationDiagnostic } from '../src/core/diagnostics.js';

// The decoder samples real processed logits on every row, including rows already stopped.
function fixture(outputs: string[], hook?: (step: number) => void, temptInvalid = false) {
  const encoded = (text: string) => Array.from(text, (character) => BigInt(character.codePointAt(0)!));
  let invisible = 256;
  const vocab = Object.fromEntries(Array.from({ length: 256 }, (_, byte) => [
    String.fromCharCode(byte >= 33 && byte <= 126 || byte >= 161 && byte <= 172 || byte >= 174 ? byte : invisible++), byte,
  ]));
  const tokenizer = new TokenizersBackend({ model: { type: 'BPE', vocab, merges: [] }, decoder: { type: 'ByteLevel' }, added_tokens: [], normalizer: null, pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: false }, post_processor: null }, {});
  const inputs: { tensor: Tensor; disposed: boolean }[] = [];
  const track = (tensor: Tensor) => {
    const entry = { tensor, disposed: false }; inputs.push(entry);
    const dispose = tensor.dispose.bind(tensor);
    tensor.dispose = () => { entry.disposed = true; dispose(); };
    return tensor;
  };
  const processor = Object.assign((text: string) => ({ input_ids: track(new Tensor('int64', BigInt64Array.from(encoded(text)), [1, text.length])) }), {
    apply_chat_template: (messages: { role: string; content: { text?: string }[] }[]) => messages.map((message) => `${message.role}:${message.content.map((part) => part.text ?? '').join('')}\n`).join('') + 'assistant:',
    tokenizer,
  });
  let calls = 0;
  const sampled: bigint[][][] = [];
  const originalForward = async () => ({});
  const model = {
    sessions: { decoder: { config: { device: 'cpu', dtype: 'q4' } } },
    generation_config: { eos_token_id: 1000, pad_token_id: 0 },
    forward: originalForward,
    async generate(options: {
      input_ids: Tensor; attention_mask?: Tensor; max_new_tokens: number;
      streamer: { put(ids: bigint[][]): void; end(): void };
      logits_processor: (ids: bigint[][], logits: Tensor) => Tensor;
      stopping_criteria: ((ids: bigint[][], scores: number[][]) => boolean[]) & { interrupted?: boolean };
    }) {
      calls++;
      const width = options.input_ids.dims[1]!;
      const rows = Array.from({ length: options.input_ids.dims[0]! }, (_, row) => Array.from(options.input_ids.data.slice(row * width, (row + 1) * width), BigInt));
      const texts = rows.map((_, row) => encoded(outputs[row] ?? outputs[0]!));
      options.streamer.put(rows.map((row) => [...row]));
      for (let step = 0; step < options.max_new_tokens; step++) {
        if (options.stopping_criteria.interrupted) break;
        const logits = new Tensor('float32', new Float32Array(rows.length * 1001).fill(-100), [rows.length, 1001]);
        texts.forEach((text, row) => { logits.data[row * 1001 + Number(text[step] ?? 1000n)] = 100; });
        if (temptInvalid && step === 0) for (let row = 0; row < rows.length; row++) logits.data[row * 1001 + 93] = 200;
        const processed = options.logits_processor(rows, logits);
        const tokens = rows.map((row, index) => {
          let best = 0;
          for (let id = 1; id < 1001; id++) if (Number(processed.data[index * 1001 + id]) > Number(processed.data[index * 1001 + best])) best = id;
          row.push(BigInt(best)); return [BigInt(best)];
        });
        sampled.push(tokens);
        options.streamer.put(tokens);
        hook?.(step);
        if (options.stopping_criteria(rows, []).every(Boolean)) break;
      }
      options.streamer.end();
      return track(new Tensor('int64', BigInt64Array.from(rows.flat()), [rows.length, rows[0]!.length]));
    },
    async dispose() {},
  };
  const profile = getRegisteredModelProfile();
  const Construct = VisionEngine as unknown as new (...args: unknown[]) => VisionEngine;
  const engine = new Construct(model, processor, { runtime: 'node', device: 'cpu', executionProviders: ['cpu'] }, 0, false, 4096, 2000,
    { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype }, undefined, true);
  return { engine, sampled, inputs, calls: () => calls, restored: () => model.forward === originalForward };
}

async function requests(engine: VisionEngine, prompts = ['Short.', 'A substantially longer second prompt.'], extra: Partial<StructuredBatchRequest['options']> = {}): Promise<StructuredBatchRequest[]> {
  const deadline = Infinity;
  return Promise.all(prompts.map(async (prompt, index) => {
    const options = { prompt, schema: { type: 'object' }, structuredMode: 'validation-only' as const, maxNewTokens: 128, ...extra };
    const plan = await engine.planInference(options);
    const budget: InferenceBudget = { remainingTokens: plan.inputTokens + options.maxNewTokens, deadline, inputTokens: 0, outputTokens: 0, timings: { preprocessMs: 0, generationMs: 0 } };
    return { options, context: { budget, diagnosticStage: { id: `row:${index}`, attempt: 1 } } };
  }));
}

function fulfilled(result: PromiseSettledResult<StructuredInferenceResult>): asserts result is PromiseFulfilledResult<StructuredInferenceResult> {
  assert.equal(result.status, 'fulfilled');
}

test('unequal prompt rows stop independently and account only real input and terminal output', async () => {
  const texts = ['{}', '{"message":"the second row keeps generating"}'];
  const { engine, sampled } = fixture(texts);
  try {
    const rows = await requests(engine);
    const reservations = rows.map(({ context }) => context.budget!.remainingTokens);
    const result = await engine.inferStructuredBatch(rows); assert.ok(result);
    assert.equal(result.length, 2);
    result.forEach((item, index) => {
      fulfilled(item); assert.deepEqual(item.value.value, JSON.parse(texts[index]!));
      assert.equal(item.value.text, texts[index]);
      assert.equal(item.value.usage.inputTokens, reservations[index]! - 128);
      assert.equal(item.value.usage.outputTokens, texts[index]!.length);
      assert.equal(rows[index]!.context.budget!.outputTokens, texts[index]!.length);
      assert.equal(rows[index]!.context.budget!.remainingTokens, 128 - texts[index]!.length);
    });
    assert.ok(sampled.length > texts[0]!.length);
    assert.ok(sampled.slice(texts[0]!.length).every((step) => step[0]![0] === 1000n), 'finished rows must be EOS-masked, not sampled normally');
  } finally { await engine.dispose(); }
});

test('constrained batch grammar masks invalid logits separately for differently padded rows', async () => {
  const { engine } = fixture(['{}', '{"x":1}'], undefined, true);
  try {
    const rows = await requests(engine, undefined, { structuredMode: 'constrained', schema: { type: 'object', additionalProperties: false, properties: { x: { type: 'integer' } } } });
    const results = await engine.inferStructuredBatch(rows); assert.ok(results);
    fulfilled(results[0]!); fulfilled(results[1]!);
    assert.deepEqual(results.map((item) => { fulfilled(item); return item.value.value; }), [{}, { x: 1 }]);
  } finally { await engine.dispose(); }
});

test('malformed JSON rejects only its row with independent diagnostic usage', async () => {
  const { engine } = fixture(['{"x":}', '{"ok":true}']);
  try {
    const rows = await requests(engine, undefined, { diagnostics: true });
    const results = await engine.inferStructuredBatch(rows); assert.ok(results);
    const failed = results[0]!; assert.equal(failed.status, 'rejected');
    if (failed.status !== 'rejected') assert.fail('Expected malformed row rejection');
    const diagnostic = getGenerationDiagnostic(failed.reason); assert.ok(diagnostic);
    assert.equal(diagnostic.stageId, 'row:0'); assert.equal(diagnostic.code, 'STRUCTURED_OUTPUT');
    assert.equal(diagnostic.usage.outputTokens, rows[0]!.context.budget!.outputTokens);
    fulfilled(results[1]!); assert.deepEqual(results[1]!.value.value, { ok: true });
    assert.equal(rows[1]!.context.budget!.outputTokens, '{"ok":true}'.length);
  } finally { await engine.dispose(); }
});

test('global abort and deadline preserve completed rows and release generation ownership and inputs', async () => {
  for (const interruption of ['abort', 'deadline', 'runtime'] as const) {
    const controller = new AbortController();
    let rows: StructuredBatchRequest[] = [];
    let interrupt = true;
    const { engine, inputs, restored } = fixture(['{}', '{"long":"unfinished"}'], (step) => {
      if (!interrupt || step !== 2) return;
      if (interruption === 'abort') controller.abort();
      else if (interruption === 'deadline') rows.forEach(({ context }) => { context.budget!.deadline = -1; });
      else throw new Error('Decoder failure');
    });
    try {
      rows = await requests(engine, undefined, { signal: controller.signal, diagnostics: true });
      const results = await engine.inferStructuredBatch(rows); assert.ok(results);
      fulfilled(results[0]!);
      const failed = results[1]!; assert.equal(failed.status, 'rejected');
      if (failed.status !== 'rejected') assert.fail('Expected interrupted row rejection');
      const diagnostic = getGenerationDiagnostic(failed.reason); assert.ok(diagnostic);
      assert.equal(diagnostic.usage.outputTokens, 3);
      assert.equal(rows[0]!.context.budget!.outputTokens, 2);
      assert.equal(rows[1]!.context.budget!.outputTokens, 3);
      assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
      assert.ok(inputs.every(({ disposed }) => disposed));
      assert.ok(restored());
      interrupt = false;
      const next = await engine.inferStructuredBatch(await requests(engine)); assert.ok(next);
      next.forEach(fulfilled);
    } finally { await engine.dispose(); }
  }
});

test('unsupported streaming and sampling decline before dispatch without charging budgets', async () => {
  const { engine, calls } = fixture(['{}', '{}']);
  try {
    for (const extra of [{ onToken: () => {} }, { generation: { sampling: true, temperature: 0.7 } }]) {
      const rows = await requests(engine, undefined, extra);
      const before = rows.map(({ context }) => ({ ...context.budget }));
      assert.equal(await engine.inferStructuredBatch(rows), undefined);
      assert.equal(calls(), 0);
      rows.forEach(({ context }, index) => {
        assert.equal(context.budget!.remainingTokens, before[index]!.remainingTokens);
        assert.equal(context.budget!.inputTokens, 0); assert.equal(context.budget!.outputTokens, 0);
      });
    }
  } finally { await engine.dispose(); }
});

test('a row output cap and a split stop sequence cannot consume a healthy sibling budget', async () => {
  for (const generation of [undefined, { stop: ['END'] }]) {
    const text = generation ? '{"x":"ENDdiscarded"}' : '{"x":"this unfinished value is too long"}';
    const { engine } = fixture([text, '{}']);
    try {
      const rows = await requests(engine, undefined, { maxNewTokens: generation ? 16 : 8, ...(generation ? { generation } : {}) });
      const results = await engine.inferStructuredBatch(rows); assert.ok(results);
      assert.equal(results[0]!.status, 'rejected');
      fulfilled(results[1]!); assert.deepEqual(results[1]!.value.value, {});
      assert.equal(rows[1]!.context.budget!.outputTokens, 2);
      assert.equal(rows[0]!.context.budget!.outputTokens, generation ? 9 : 8);
    } finally { await engine.dispose(); }
  }
});
