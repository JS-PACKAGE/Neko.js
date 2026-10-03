import assert from 'node:assert/strict';
import test from 'node:test';
import { RequestQueue } from '../src/core-queue.js';
import { queuedReadable } from '../src/queued-stream.js';
import { NekoError } from '../src/errors.js';

test('resource stream admission is lazy and remains owned until consumer cancellation', async () => {
  const queue = new RequestQueue();
  const stream = queuedReadable((signal, operation) => queue.enqueue('cache', signal, operation), () => new ReadableStream({
    pull(controller) { controller.enqueue(new Uint8Array([7])); },
  }, { highWaterMark: 0 }));
  assert.equal(queue.status().admitted, 0);
  const reader = stream.getReader();
  assert.deepEqual((await reader.read()).value, new Uint8Array([7]));
  let mutated = false;
  const mutation = queue.enqueue('cache', new AbortController().signal, async () => { mutated = true; });
  await Promise.resolve();
  assert.equal(mutated, false);
  await reader.cancel();
  await mutation;
  assert.equal(mutated, true);
  await queue.idle();
  assert.equal(queue.status().running, null);
});

test('lifetime cancellation errors the consuming stream and releases its admission', async () => {
  const queue = new RequestQueue();
  const lifetime = new AbortController();
  const stream = queuedReadable((signal, operation) => queue.enqueue('cache', AbortSignal.any([signal, lifetime.signal]), operation), () => new ReadableStream({
    pull(controller) { controller.enqueue(new Uint8Array([1])); },
  }, { highWaterMark: 0 }));
  const reader = stream.getReader();
  await reader.read();
  lifetime.abort('disposed');
  await assert.rejects(reader.read(), (error: unknown) => error instanceof NekoError && error.code === 'ABORTED');
  await queue.idle();
  assert.equal(queue.status().running, null);
});

test('resource read failure reaches consumer and does not strand queued operations', async () => {
  const queue = new RequestQueue();
  const stream = queuedReadable((signal, operation) => queue.enqueue('cache', signal, operation), () => new ReadableStream({
    pull(controller) { controller.error(new Error('fixture failed')); },
  }, { highWaterMark: 0 }));
  await assert.rejects(stream.getReader().read(), /fixture failed/);
  await queue.idle();
  await queue.enqueue('cache', new AbortController().signal, async () => undefined);
});
