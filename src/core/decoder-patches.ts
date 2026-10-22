import { Tensor, type PreTrainedModel } from '@huggingface/transformers';

interface DecoderValue { location?: string; data?: unknown }
interface DecoderSession { inputNames: readonly string[]; run(feeds: Record<string, { data: ArrayLike<unknown> }>, ...rest: unknown[]): Promise<unknown> }

/**
 * Transformers.js 4.2.0's `generic_text_to_text_forward` drops `num_logits_to_keep`, so `decoder_forward` sends 0 ("every position")
 * and the decoder projects the whole prompt onto the 248,320-token vocabulary during prefill (about 1 MB of logits per prompt token).
 * Generation only reads the last position, so this makes the given model's decoder session keep one; explicit nonzero values pass through.
 */
export function keepLastLogits(model: PreTrainedModel): void {
  const session = model.sessions.decoder_model_merged as DecoderSession | undefined;
  if (!session?.inputNames.includes('num_logits_to_keep')) return;
  const run = session.run;
  const last = new Tensor('int64', BigInt64Array.of(1n), []).ort_tensor as unknown as { data: ArrayLike<unknown> };
  Object.defineProperty(session, 'run', { configurable: true, writable: true, value: (feeds: Record<string, { data: ArrayLike<unknown> }>, ...rest: unknown[]) =>
    run.call(session, feeds.num_logits_to_keep?.data[0] === 0n ? { ...feeds, num_logits_to_keep: last } : feeds, ...rest) });
}

/** The buffer of a CPU tensor that owns all of it; anything shared or partial is left to the garbage collector. */
function ownedBuffer(value: unknown): ArrayBuffer | undefined {
  const { location, data } = (value ?? {}) as DecoderValue;
  if (location !== 'cpu' || !ArrayBuffer.isView(data) || !(data.buffer instanceof ArrayBuffer)) return undefined;
  return data.byteOffset === 0 && data.byteLength === data.buffer.byteLength && data.byteLength > 0 ? data.buffer : undefined;
}

/**
 * onnxruntime-node copies every output into a fresh ArrayBuffer, and a Qwen3.5 decode step returns about 21 MB of recurrent,
 * convolution and attention state (plus 1 MB of logits). It became garbage one step later, so V8's external-memory pressure forced
 * a full mark-compact every two or three tokens. Generation never reads state again once the decoder has consumed it as `past_*`
 * (cache checkouts and snapshots own clones), nor a run's logits once the next run starts (generation has copied them), so their
 * memory is released at that point instead. Only buffers this session itself returned are detached. Node CPU only.
 */
export function releaseConsumedDecoderOutputs(model: PreTrainedModel): void {
  const session = model.sessions.decoder_model_merged as DecoderSession | undefined;
  if (!session) return;
  const run = session.run;
  const produced = new WeakSet<ArrayBuffer>();
  let logits: ArrayBuffer | undefined;
  Object.defineProperty(session, 'run', { configurable: true, writable: true, value: async (feeds: Record<string, { data: ArrayLike<unknown> }>, ...rest: unknown[]) => {
    const outputs = await run.call(session, feeds, ...rest) as Record<string, unknown>;
    for (const [name, value] of Object.entries(feeds)) {
      const buffer = name.startsWith('past_') ? ownedBuffer(value) : undefined;
      if (buffer && produced.delete(buffer)) buffer.transfer(0);
    }
    if (logits && !logits.detached) logits.transfer(0);
    logits = ownedBuffer(outputs.logits);
    for (const [name, value] of Object.entries(outputs)) {
      const buffer = name.startsWith('present') ? ownedBuffer(value) : undefined;
      if (buffer) produced.add(buffer);
    }
    return outputs;
  } });
}
