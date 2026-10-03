import assert from 'node:assert/strict';
import test from 'node:test';
import { Tensor } from '@huggingface/transformers';
import { VisionEngine } from '../src/core/engine.js';
import { getGenerationDiagnostic } from '../src/core/diagnostics.js';
import { getRegisteredModelProfile } from '../src/cache/registry.js';
import { ReportError } from '../src/report/generate.js';
import { NekoError } from '../src/errors.js';

function truncatedEngine() {
  const tokenizer = { encode: (text: string) => Array.from(text, (c) => BigInt(c.codePointAt(0)!)), decode: (ids: bigint[]) => ids.map((id) => String.fromCodePoint(Number(id))).join(''), all_special_ids: [] };
  const processor = Object.assign((text: string) => ({ input_ids: new Tensor('int64', tokenizer.encode(text), [1, text.length]) }), { tokenizer, apply_chat_template: () => 'prompt' });
  const model = { sessions: { decoder: { config: { device: 'cpu', dtype: 'q4' } } }, generation_config: { eos_token_id: 1000 }, async generate(options: { input_ids: Tensor; streamer: { put(ids: bigint[][]): void; end(): void } }) {
    const prompt = Array.from(options.input_ids.data, BigInt); const output = tokenizer.encode('{"secret":"private');
    options.streamer.put([prompt]); options.streamer.put([output]); options.streamer.end();
    return new Tensor('int64', [...prompt, ...output], [1, prompt.length + output.length]);
  }, async dispose() {} };
  const profile = getRegisteredModelProfile();
  const Construct = VisionEngine as unknown as new (...args: unknown[]) => VisionEngine;
  return new Construct(model, processor, { runtime: 'node', device: 'cpu', executionProviders: ['cpu'] }, 0, false, 4096, 2000, { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype }, undefined, true);
}

test('failed structured diagnostics include termination, stage and failed usage without content by default', async () => {
  const engine = truncatedEngine();
  try {
    for (const diagnostics of [undefined, true, { capture: { maxCharacters: 8 } }] as const) {
      await assert.rejects(engine.inferStructured({ prompt: 'Return JSON', schema: true, structuredMode: 'validation-only', maxNewTokens: 18, ...(diagnostics ? { diagnostics } : {}) }, { diagnosticStage: { id: 'section:3', attempt: 2 } }), (error: unknown) => {
        assert.ok(error instanceof NekoError);
        const diagnostic = getGenerationDiagnostic(error);
        if (!diagnostics) { assert.equal(diagnostic, undefined); return true; }
        assert.ok(diagnostic);
        assert.equal(diagnostic.stageId, 'section:3'); assert.equal(diagnostic.attempt, 2);
        assert.equal(diagnostic.finishReason, 'length'); assert.equal(diagnostic.usage.outputTokens, 18);
        assert.equal(diagnostic.json?.position, 18);
        if (diagnostics === true) { assert.equal(diagnostic.capture, undefined); assert.ok(!JSON.stringify(diagnostic).includes('private')); }
        else { assert.equal(diagnostic.capture?.output, '{"secret'); assert.equal(diagnostic.capture?.truncated, true); }
        const wrapped = new ReportError(error, {} as never, {} as never);
        assert.deepEqual(getGenerationDiagnostic(wrapped), diagnostic);
        return true;
      });
    }
  } finally { await engine.dispose(); }
});
