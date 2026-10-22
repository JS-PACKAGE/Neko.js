import { Tensor, type PreTrainedModel } from '@huggingface/transformers';

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
