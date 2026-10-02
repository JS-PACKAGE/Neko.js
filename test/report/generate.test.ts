import assert from 'node:assert/strict';
import test from 'node:test';
import { generateReport } from '../../src/report/generate.js';
import { atStage, NekoError } from '../../src/errors.js';
import type { VisionEngine, InferOptions } from '../../src/core/engine.js';

const html = '<article><p>A garden contains oak trees.</p><img src="https://example.test/garden.png"></article>';
const engine = (imageFailure?: NekoError) => ({
  contextLimit: () => 1024,
  countPrompt: () => 20,
  splitText: (text: string) => [text],
  infer: (options: InferOptions) => atStage('generate', options.signal, async () => {
    if (options.image && imageFailure) throw imageFailure;
    options.onToken?.('The garden contains oak trees.');
    return { text: 'The garden contains oak trees.', finishReason: 'stop' };
  }),
}) as unknown as VisionEngine;

test('omit preserves image failures but rejects application stream callback errors', async () => {
  const imageFailure = new NekoError('Unsupported image', 'image', 'INVALID_INPUT');
  const report = await generateReport(engine(imageFailure), html, { imageFailurePolicy: 'omit' });
  assert.notEqual(typeof report, 'string');
  assert.ok(typeof report !== 'string');
  assert.deepEqual(report.images, [{ imageId: 'i1', url: 'https://example.test/garden.png', source: { kind: 'image', imageId: 'i1' }, status: 'failed', error: { stage: 'image', code: 'INVALID_INPUT', message: imageFailure.message } }]);

  for (const callbackError of [new Error('Application stopped rendering'), undefined]) {
    await assert.rejects(generateReport(engine(), html, {
      imageFailurePolicy: 'omit',
      onToken: (_text, phase) => { if (phase === 'image') throw callbackError; },
    }), (error: unknown) => error instanceof NekoError && error.stage === 'generate' && error.code === 'OPERATION_FAILED' && error.cause === callbackError);
  }
});
