import assert from 'node:assert/strict';
import test from 'node:test';
import { RequestQueue } from '../src/core-queue.js';
import { NekoError } from '../src/errors.js';

const signal = (): AbortSignal => new AbortController().signal;
const code = (expected: string) => (error: unknown): boolean => error instanceof NekoError && error.code === expected;

test('zero pending capacity permits the next request immediately after the previous result settles', async () => {
  const queue = new RequestQueue(0); let executions = 0;
  await queue.enqueue('generate', signal(), async () => { executions++; });
  await queue.enqueue('report', signal(), async () => { executions++; });
  assert.equal(executions, 2); assert.equal(queue.status().running, null);
});

test('queued cancellation frees capacity without executing cancelled work or overtaking active work', async () => {
  const queue = new RequestQueue(1); const started = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>();
  const order: string[] = [];
  const first = queue.enqueue('generate', signal(), async () => { order.push('first'); started.resolve(); await release.promise; });
  await started.promise;
  const cancelled = new AbortController();
  const pending = queue.enqueue('image', cancelled.signal, async () => { order.push('cancelled'); });
  await assert.rejects(queue.enqueue('report', signal(), async () => { order.push('overflow'); }), code('QUEUE_FULL'));
  cancelled.abort('caller stopped waiting'); await assert.rejects(pending, code('ABORTED'));
  const replacement = queue.enqueue('report', signal(), async () => { order.push('replacement'); });
  assert.deepEqual(order, ['first']); release.resolve(); await Promise.all([first, replacement]); await queue.idle();
  assert.deepEqual(order, ['first', 'replacement']); assert.equal(queue.status().pending, 0);
});

test('an active failure preserves its undefined cause and does not poison queued admission', async () => {
  const queue = new RequestQueue(1); const started = Promise.withResolvers<void>(); const release = Promise.withResolvers<void>(); let completed = false;
  const failure = queue.enqueue('generate', signal(), async () => { started.resolve(); await release.promise; throw undefined; });
  await started.promise;
  const next = queue.enqueue('report', signal(), async () => { completed = true; });
  const rejected = assert.rejects(failure, (error: unknown) => error instanceof NekoError && error.code === 'OPERATION_FAILED' && Object.hasOwn(error, 'cause') && error.cause === undefined);
  release.resolve(); await rejected; await next; assert.equal(completed, true); assert.equal(queue.status().running, null);
});
