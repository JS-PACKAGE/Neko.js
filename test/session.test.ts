import assert from 'node:assert/strict';
import test from 'node:test';
import sharp from 'sharp';
import { createConversationSession, type SessionHost, type SessionOptions } from '../src/core/session.js';
import type { ChatContent, InferOptions, InferencePlan, InferenceResult, ModelIdentity } from '../src/core/engine.js';
import { getRegisteredModelProfile } from '../src/cache/registry.js';
import { NekoError } from '../src/errors.js';

const profile = getRegisteredModelProfile();
const identity: ModelIdentity = { id: profile.id, revision: profile.revision, profile: profile.profile, dtype: profile.dtype };
function fixture(options: SessionOptions = {}, inference?: (options: InferOptions) => Promise<InferenceResult>) {
  const planned: InferOptions[] = []; const inferred: InferOptions[] = [];
  let fits: (options: InferOptions) => boolean = () => true;
  const result: InferenceResult = {
    text: '\n exact assistant\t', finishReason: 'stop', usage: { inputTokens: 4, outputTokens: 2, totalTokens: 6 }, model: identity,
    backend: { runtime: 'node', device: 'cpu', executionProviders: ['cpu'], gpuMemoryBytes: null, adapterEvidence: null, supported: true, capabilityEvidence: 'model-runtime-compatibility', sessions: [], providerEvidence: 'loaded-session-configuration' },
    timings: { loadMs: 0, preprocessMs: 0, firstTokenMs: null, generationMs: 0, totalMs: 0 }, memory: { jsHeapBytes: null, gpuBytes: null },
  };
  const host: SessionHost = {
    async planInference(input) {
      planned.push(input);
      const plan: InferencePlan = { inputTokens: 4, maxNewTokens: input.maxNewTokens ?? 128, contextLimit: 4096, availableOutputTokens: 4092, fits: fits(input), model: identity };
      return plan;
    },
    async infer(input) { inferred.push(input); return inference ? inference(input) : result; },
  };
  return { session: createConversationSession(host, identity, options), host, planned, inferred, result, setFits: (value: typeof fits) => { fits = value; } };
}

// These deterministic hosts exercise session state/ownership, not model output quality.
test('session serializes sends, preserves exact messages/results and plans unchanged generation input', async () => {
  const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>();
  let calls = 0;
  const setup = fixture({ system: '\n system \t' }, async () => { if (++calls === 1) { entered.resolve(); await release.promise; } return setup.result; });
  const first = setup.session.send('\n user one \t');
  await entered.promise;
  const second = setup.session.send('user two');
  assert.equal(setup.inferred.length, 1); assert.deepEqual(setup.session.history(), [{ role: 'system', content: '\n system \t' }]);
  release.resolve();
  assert.strictEqual(await first, setup.result); assert.strictEqual(await second, setup.result);
  assert.deepEqual(setup.inferred[1]!.messages, [
    { role: 'system', content: '\n system \t' }, { role: 'user', content: '\n user one \t' },
    { role: 'assistant', content: setup.result.text }, { role: 'user', content: 'user two' },
  ]);
  for (let index = 0; index < 2; index++) assert.deepEqual(setup.planned[index]!.messages, setup.inferred[index]!.messages);
});

test('failed and cancelled generation never append partial turns and later sends remain usable', async () => {
  const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>();
  let calls = 0; let streamed = '';
  const setup = fixture({}, async (options) => {
    options.onToken?.('partial');
    if (++calls === 1) throw new Error('generation failed');
    if (calls === 2) { entered.resolve(); await release.promise; }
    return setup.result;
  });
  await assert.rejects(setup.session.send('failed', { onToken: (text) => { streamed += text; } }), /generation failed/);
  assert.equal(streamed, 'partial'); assert.deepEqual(setup.session.history(), []);
  const abort = new AbortController(); const cancelled = setup.session.send('cancelled', { signal: abort.signal });
  await entered.promise;
  abort.abort('stop'); await assert.rejects(cancelled, (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
  release.resolve();
  await setup.session.send('success');
  assert.deepEqual(setup.session.history(), [{ role: 'user', content: 'success' }, { role: 'assistant', content: setup.result.text }]);
});

test('cancelled queued send never runs and does not overtake the active turn', async () => {
  const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>();
  const setup = fixture({}, async () => { entered.resolve(); await release.promise; return setup.result; });
  const first = setup.session.send('first'); await entered.promise;
  const abort = new AbortController(); const queued = setup.session.send('cancelled', { signal: abort.signal });
  abort.abort(); await assert.rejects(queued, (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
  const third = setup.session.send('third'); release.resolve(); await Promise.all([first, third]);
  assert.equal(setup.inferred.length, 2); assert.equal(setup.inferred[1]!.messages!.at(-1)!.content, 'third');
});

test('drop-oldest preserves system and complete turns and commits drops only after success', async () => {
  const setup = fixture({ system: 'system', contextPolicy: 'drop-oldest', maxHistoryMessages: 5 });
  await setup.session.send('one'); await setup.session.send('two'); await setup.session.send('three');
  assert.deepEqual(setup.session.history().map((message) => message.content), ['system', 'two', setup.result.text, 'three', setup.result.text]);
  const before = setup.session.history(); setup.setFits(() => false);
  await assert.rejects(setup.session.send('too large'), (error: unknown) => error instanceof NekoError && error.code === 'CONTEXT_LIMIT');
  assert.deepEqual(setup.session.history(), before);
  setup.setFits((options) => options.messages!.length <= 2);
  await setup.session.send('fits alone');
  assert.deepEqual(setup.session.history().map((message) => message.content), ['system', 'fits alone', setup.result.text]);
});

test('error context policy rejects overflow without dropping history or generating', async () => {
  const setup = fixture({ maxHistoryMessages: 2 }); await setup.session.send('first');
  await assert.rejects(setup.session.send('next'), (error: unknown) => error instanceof NekoError && error.code === 'CONTEXT_LIMIT');
  assert.equal(setup.inferred.length, 1); assert.equal(setup.session.history()[0]!.content, 'first');
  await setup.session.reset(); setup.setFits(() => false);
  await assert.rejects(setup.session.send('oversized'), (error: unknown) => error instanceof NekoError && error.code === 'CONTEXT_LIMIT');
  assert.deepEqual(setup.session.history(), []);
});

test('planning is nonmutating and branch/reset isolate histories while preserving system', async () => {
  const setup = fixture({ system: 'system' }); await setup.session.send('original');
  const before = setup.session.history(); await setup.session.plan('planned only'); assert.deepEqual(setup.session.history(), before);
  const branch = await setup.session.branch(); await branch.send('branch');
  assert.deepEqual(setup.session.history(), before); assert.equal(branch.history().length, 5);
  const clone = branch.history(); clone[0]!.content = 'mutated'; assert.equal(branch.history()[0]!.content, 'system');
  await branch.reset(); assert.deepEqual(branch.history(), [{ role: 'system', content: 'system' }]);
  await branch.dispose(); await setup.session.send('parent survives');
});

test('snapshots are portable immutable deep clones and importing validates version/model/complete turns', async () => {
  const setup = fixture({ system: 'system', generation: { stop: ['stop'] } }); await setup.session.send('exact\ntext');
  const snapshot = await setup.session.export();
  assert.equal(Object.isFrozen(snapshot), true); assert.equal(Object.isFrozen(snapshot.model.dtype), true); assert.equal(Object.isFrozen(snapshot.options.generation!.stop), true);
  const branch = createConversationSession(setup.host, identity);
  await branch.import(JSON.parse(JSON.stringify(snapshot))); assert.deepEqual(branch.history(), setup.session.history());
  const before = branch.history();
  for (const bad of [
    { ...snapshot, version: 2 }, { ...snapshot, model: { ...identity, revision: 'other' } },
    { ...snapshot, model: { ...identity, dtype: { ...identity.dtype, extra: 'q4' } } },
    { ...snapshot, messages: snapshot.messages.slice(0, -1) },
    { ...snapshot, messages: [{ role: 'system', content: 'different' }, ...snapshot.messages.slice(1)] },
    { ...snapshot, options: { ...snapshot.options, signal: {} } },
  ]) await assert.rejects(branch.import(bad), TypeError);
  assert.deepEqual(branch.history(), before);
  const reordered = { ...snapshot, model: { ...identity, dtype: { vision_encoder: identity.dtype.vision_encoder, decoder_model_merged: identity.dtype.decoder_model_merged, embed_tokens: identity.dtype.embed_tokens } } };
  await branch.import(reordered);
});

test('pixel inputs, queued content and exported/imported pixels cannot mutate session history', async () => {
  const setup = fixture();
  const pixels = { data: new Uint8Array([1, 2, 3]), width: 1, height: 1, channels: 3 as const };
  const content: ChatContent[] = [{ type: 'text', text: 'image exact' }, { type: 'image', image: pixels }];
  const send = setup.session.send(content); pixels.data[0] = 255; content[0] = { type: 'text', text: 'changed' };
  await send;
  const history = setup.session.history();
  assert.equal((history[0]!.content as ChatContent[])[0]!.type, 'text');
  const imagePart = (history[0]!.content as ChatContent[])[1]!; assert.equal(imagePart.type, 'image');
  if (imagePart.type !== 'image' || typeof imagePart.image !== 'object' || !('data' in imagePart.image)) throw new Error('Expected pixels');
  assert.equal(imagePart.image.data[0], 1); imagePart.image.data[0] = 99;
  const snapshot = await setup.session.export(); const branch = await setup.session.branch();
  await branch.import(JSON.parse(JSON.stringify(snapshot)));
  assert.deepEqual(branch.history(), setup.session.history());
  const invalid = JSON.parse(JSON.stringify(snapshot)); invalid.messages[0].content[1].image.data = 'AA==';
  await assert.rejects(branch.import(invalid), /length does not match/);
});

test('export refuses filesystem/network/URL references without invoking image fetch or filesystem IO', async () => {
  for (const image of ['/private/image.png', 'https://example.com/image.png', new URL('file:///private/image.png')]) {
    const setup = fixture(); await setup.session.send([{ type: 'text', text: 'image' }, { type: 'image', image }]);
    await assert.rejects(setup.session.export(), TypeError); assert.equal(setup.inferred.length, 1);
  }
});

test('dispose cancels active and pending sends but never disposes the shared host', async () => {
  const entered = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>();
  let disposed = false;
  const setup = fixture({}, async () => { entered.resolve(); await release.promise; return setup.result; });
  const host = Object.assign(setup.host, { async dispose() { disposed = true; } });
  const session = createConversationSession(host, identity);
  const first = session.send('active'); await entered.promise; const pending = session.send('pending');
  const firstRejected = assert.rejects(first, (error: unknown) => error instanceof NekoError && error.code === 'DISPOSED');
  const pendingRejected = assert.rejects(pending, (error: unknown) => error instanceof NekoError && error.code === 'DISPOSED');
  const disposal = session.dispose(); await Promise.all([firstRejected, pendingRejected]); release.resolve(); await disposal;
  assert.equal(disposed, false); assert.equal(setup.inferred.length, 1);
  await assert.rejects(session.send('closed'), (error: unknown) => error instanceof NekoError && error.code === 'DISPOSED');
  const survivor = createConversationSession(host, identity); await survivor.send('still usable');
});

test('embedded raster Blobs export as portable data and pixel overrides cannot produce invalid snapshots', async () => {
  const setup = fixture();
  const bytes = await sharp({ create: { width: 1, height: 1, channels: 3, background: '#ff0000' } }).png().toBuffer();
  const image = new Blob([new Uint8Array(bytes)], { type: 'image/png' });
  await setup.session.send([{ type: 'text', text: 'embedded' }, { type: 'image', image }]);
  const snapshot = await setup.session.export();
  const branch = createConversationSession(setup.host, identity);
  await branch.import(JSON.parse(JSON.stringify(snapshot)));
  const content = branch.history()[0]!.content as ChatContent[];
  assert.deepEqual(content[1], { type: 'image', image: `data:image/png;base64,${bytes.toString('base64')}` });
  const limited = fixture({ maxImageBytes: 2 });
  await limited.session.send([{ type: 'text', text: 'pixels' }, { type: 'image', image: { data: new Uint8Array([1, 2, 3]), width: 1, height: 1, channels: 3 } }], { maxImageBytes: 3 });
  await assert.rejects(limited.session.export(), /snapshot byte limit/);
});

test('session defaults and host arguments cannot mutate retained generation settings', async () => {
  const generation = { stop: ['original'] };
  const setup = fixture({ generation });
  generation.stop[0] = 'caller change';
  const host = setup.host;
  const originalPlan = host.planInference.bind(host);
  host.planInference = async (options) => { const result = await originalPlan(options); options.generation!.stop![0] = 'host change'; return result; };
  await setup.session.send('first');
  assert.deepEqual(setup.inferred[0]!.generation!.stop, ['original']);
  const snapshot = await setup.session.export();
  assert.deepEqual(snapshot.options.generation!.stop, ['original']);
});
