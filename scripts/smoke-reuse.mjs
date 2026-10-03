import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { Neko } from '../dist/node/index.js';

const { values } = parseArgs({ options: { device: { type: 'string', default: 'cpu' }, 'model-profile': { type: 'string', default: 'default' }, 'cache-dir': { type: 'string' }, 'allow-network': { type: 'boolean', default: false } } });
if (values.device !== 'cpu' || !['default', 'all-q4'].includes(values['model-profile'])) throw new TypeError('Node reuse smoke supports --device cpu --model-profile default|all-q4; browser WebGPU must be exercised separately');
const neko = await Neko.create({ device: values.device, modelProfile: values['model-profile'], localFilesOnly: !values['allow-network'], ...(values['cache-dir'] ? { cacheDir: values['cache-dir'] } : {}) });
const generation = { sampling: false };
try {
  const messages = [{ role: 'user', content: 'Name one primary color. Answer in one short sentence.' }];
  const first = await neko.infer({ messages, generation, maxNewTokens: 32, reuse: { retainState: true } });
  assert.ok(first.reuse?.state);
  const secondMessages = [...messages, { role: 'assistant', content: first.text }, { role: 'user', content: 'Name a different primary color. Answer in one short sentence.' }];
  const baseline = await neko.infer({ messages: secondMessages, generation, maxNewTokens: 32 });
  const cached = await neko.infer({ messages: secondMessages, generation, maxNewTokens: 32, reuse: { state: first.reuse.state, retainState: true } });
  assert.ok(cached.reuse.reusedDecoderTokens > 0);
  assert.equal(cached.text, baseline.text); assert.deepEqual(cached.usage, baseline.usage);
  // Branching the same handle must not change the original recurrent state.
  const branch = await neko.infer({ messages: secondMessages, generation, maxNewTokens: 32, reuse: { state: first.reuse.state } });
  assert.equal(branch.text, baseline.text);
  assert.equal(branch.reuse.reusedDecoderTokens, cached.reuse.reusedDecoderTokens);
  const thirdMessages = [...secondMessages, { role: 'assistant', content: cached.text }, { role: 'user', content: 'Name the remaining primary color. Answer in one short sentence.' }];
  const thirdBaseline = await neko.infer({ messages: thirdMessages, generation, maxNewTokens: 32 });
  const thirdCached = await neko.infer({ messages: thirdMessages, generation, maxNewTokens: 32, reuse: { state: cached.reuse.state } });
  assert.ok(thirdCached.reuse.reusedDecoderTokens > cached.reuse.reusedDecoderTokens);
  assert.equal(thirdCached.text, thirdBaseline.text); assert.deepEqual(thirdCached.usage, thirdBaseline.usage);
  const beforeAbort = await neko.reuseCacheInfo();
  const controller = new AbortController();
  await assert.rejects(neko.infer({ messages: secondMessages, generation, maxNewTokens: 32, signal: controller.signal, onToken: () => controller.abort(), reuse: { state: first.reuse.state, retainState: true } }), { code: 'ABORTED' });
  const afterAbort = await neko.reuseCacheInfo();
  assert.equal(afterAbort.stateEntries, beforeAbort.stateEntries); assert.equal(afterAbort.stateBytes, beforeAbort.stateBytes);
  const image = { width: 32, height: 32, channels: 3, data: new Uint8Array(32 * 32 * 3).fill(128) };
  const imageOptions = { image, prompt: 'Describe the image in one short sentence.', generation, maxNewTokens: 32 };
  const visionBaseline = await neko.infer(imageOptions);
  const visionFirst = await neko.infer({ ...imageOptions, reuse: { vision: true } });
  const visionSecond = await neko.infer({ ...imageOptions, reuse: { vision: true } });
  assert.ok(visionFirst.reuse.visionEncoderMisses > 0); assert.ok(visionSecond.reuse.visionEncoderHits > 0);
  assert.equal(visionFirst.text, visionBaseline.text); assert.equal(visionSecond.text, visionBaseline.text);
  await neko.releaseGenerationState(first.reuse.state);
  await assert.rejects(neko.infer({ messages: secondMessages, generation, maxNewTokens: 32, reuse: { state: first.reuse.state } }), { code: 'INVALID_INPUT' });
  await neko.clearReuseCaches();
  const info = await neko.reuseCacheInfo(); assert.equal(info.stateEntries, 0); assert.equal(info.visionEntries, 0);
  console.log(JSON.stringify({ passed: true, profile: values['model-profile'], backend: cached.backend.sessions, reusedDecoderTokens: cached.reuse.reusedDecoderTokens, visionEncoderHits: visionSecond.reuse.visionEncoderHits }));
} finally { await neko.dispose(); }
