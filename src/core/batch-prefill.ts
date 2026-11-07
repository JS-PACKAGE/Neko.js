import { DynamicCache, Tensor } from '@huggingface/transformers';
import { InferenceReuseCache, type GenerationStateSnapshot } from './reuse.js';

export interface BatchPrefillRow { inputTokens: number; prefixTokens?: bigint[] | undefined; vision?: { pixel_values: Tensor; image_grid_thw: Tensor } | undefined; }

function dispose(tensors: Iterable<Tensor>): void {
  for (const tensor of new Set(tensors)) if (tensor.location !== 'none') tensor.dispose();
}

/** Prefill independent, unpadded hybrid states, then restore the padded layout used by batched decode. */
export async function prefillBatch(modelInputs: Record<string, unknown>, rows: readonly BatchPrefillRow[], forward: (inputs: Record<string, unknown>) => Promise<Record<string, unknown>>, reuse: InferenceReuseCache, staged: Map<string, GenerationStateSnapshot>, check: () => void): Promise<Record<string, unknown>> {
  const ids = modelInputs.input_ids; const positions = modelInputs.position_ids; const mask = modelInputs.attention_mask;
  if (!(ids instanceof Tensor) || !(positions instanceof Tensor) || !(mask instanceof Tensor)
    || ids.dims.length !== 2 || ids.dims[0] !== rows.length || !rows.length
    || positions.dims.length !== 3 || positions.dims[0] !== 3 || positions.dims[1] !== rows.length || positions.dims[2] !== ids.dims[1]
    || mask.dims.length !== 2 || mask.dims[0] !== rows.length || mask.dims[1] !== ids.dims[1]) throw new Error('Invalid batch prefill input layout');
  const width = ids.dims[1]!;
  const merged: Record<string, Tensor> = {};
  let outputNames: string[] | undefined;
  try {
    for (let index = 0; index < rows.length; index++) {
      check();
      const row = rows[index]!;
      if (!Number.isSafeInteger(row.inputTokens) || row.inputTokens < 1 || row.inputTokens > width) throw new Error('Invalid batch prefill row length');
      const temporary = new Set<Tensor>();
      const own = (tensor: Tensor) => { temporary.add(tensor); return tensor; };
      let cache = new DynamicCache();
      try {
        const start = width - row.inputTokens;
        const fullIds = own(ids.slice([index, index + 1], [start, width]));
        const fullPositions = own(positions.slice(null, [index, index + 1], [start, width]));
        const tokens = Array.from(fullIds.data, BigInt);
        const prefix = row.prefixTokens;
        const key = prefix?.join(',');
        let skipped = 0;
        if (prefix && key !== undefined) {
          if (!prefix.length || prefix.length >= tokens.length || !prefix.every((token, offset) => token === tokens[offset])) throw new Error('Batch prefill prefix does not match prompt');
          const checked = reuse.checkoutPrefix(key, tokens);
          if (checked) { cache = checked.cache; skipped = checked.tokens; }
          else {
            const snapshot = staged.get(key);
            if (snapshot && snapshot.tokens.length < tokens.length && snapshot.tokens.every((token, offset) => token === tokens[offset])) {
              const entries: Record<string, Tensor> = {};
              for (const [name, tensor] of Object.entries(snapshot.cache)) entries[name] = own(tensor.clone());
              cache = new DynamicCache(entries); skipped = snapshot.tokens.length;
            }
          }
        }
        for (const tensor of Object.values(cache)) temporary.add(tensor);
        const run = async (from: number, to: number): Promise<Record<string, unknown>> => {
          const inputIds = from === 0 && to === row.inputTokens ? fullIds : own(fullIds.slice(null, [from, to]));
          const inputPositions = from === 0 && to === row.inputTokens ? fullPositions : own(fullPositions.slice(null, null, [from, to]));
          const attention = own(new Tensor('int64', new BigInt64Array(to).fill(1n), [1, to]));
          const inputs: Record<string, unknown> = { ...modelInputs, input_ids: inputIds, position_ids: inputPositions, attention_mask: attention, past_key_values: cache };
          // No per-row preparation: the caller's full-batch RoPE deltas must remain untouched. Only the row's own image (never a
          // prefix: image rows have none) is encoded and merged, exactly as an unbatched prefill of that prompt does.
          if (row.vision) { inputs.pixel_values = row.vision.pixel_values; inputs.image_grid_thw = row.vision.image_grid_thw; }
          else { delete inputs.pixel_values; delete inputs.image_grid_thw; }
          check();
          const output = await forward(inputs);
          for (const tensor of Object.values(output)) if (tensor instanceof Tensor) temporary.add(tensor);
          check();
          return output;
        };
        if (prefix && key !== undefined && !skipped) {
          const output = await run(0, prefix.length);
          const entries: Record<string, Tensor> = {};
          for (const [name, tensor] of Object.entries(output)) if (name.startsWith('present') && tensor instanceof Tensor) entries[name.replace('present_conv', 'past_conv').replace('present_recurrent', 'past_recurrent').replace('present', 'past_key_values')] = tensor;
          cache.update(entries);
          // Only snapshotting is best-effort; a failed native prefill must propagate.
          let snapshot: GenerationStateSnapshot | undefined;
          try { snapshot = await reuse.snapshot(cache, prefix.slice(), ''); }
          catch { /* Internal prefix retention is an optional optimisation. */ }
          if (snapshot) {
            const previous = staged.get(key);
            if (previous) reuse.discardState(previous);
            staged.set(key, snapshot);
          }
          skipped = prefix.length;
        }
        const output = await run(skipped, row.inputTokens);
        const names = Object.keys(output).sort();
        if (!outputNames) {
          if (!(output.logits instanceof Tensor)) throw new Error('Batch prefill returned no logits');
          outputNames = names;
        } else if (names.length !== outputNames.length || names.some((name, offset) => name !== outputNames![offset])) throw new Error('Batch prefill output names differ between rows');
        for (const name of names) {
          const tensor = output[name];
          if (!(tensor instanceof Tensor) || tensor.dims[0] !== 1) throw new Error(`Invalid batch prefill output: ${name}`);
          const attention = /^present\.\d+\.(key|value)$/.test(name);
          if (attention && (tensor.dims.length !== 4 || tensor.dims[2] !== row.inputTokens)) throw new Error(`Invalid batch prefill attention layout: ${name}`);
          if (tensor.location !== 'cpu') await tensor.ort_tensor.getData(true);
          const data = tensor.data;
          if (!ArrayBuffer.isView(data) || data instanceof DataView) throw new Error(`Batch prefill requires numeric tensor storage: ${name}`);
          let target = merged[name];
          if (!target) {
            const dims = tensor.dims.slice(); dims[0] = rows.length;
            if (attention) dims[2] = width;
            const Storage = data.constructor as { new(length: number): typeof data };
            target = merged[name] = new Tensor(tensor.type, new Storage(dims.reduce((total, value) => total * value, 1)), dims);
          }
          if (target.type !== tensor.type || target.dims.length !== tensor.dims.length || tensor.dims.some((dim, axis) => axis !== 0 && !(attention && axis === 2) && dim !== target.dims[axis])) throw new Error(`Batch prefill output layout differs between rows: ${name}`);
          const targetData = target.data;
          if (!ArrayBuffer.isView(targetData)) throw new Error('Invalid merged tensor storage');
          const destination = new Uint8Array(targetData.buffer, targetData.byteOffset, targetData.byteLength);
          const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          if (attention) {
            const heads = tensor.dims[1]!;
            const sourceHeadBytes = source.byteLength / heads;
            const targetHeadBytes = sourceHeadBytes / row.inputTokens * width;
            for (let head = 0; head < heads; head++) destination.set(source.subarray(head * sourceHeadBytes, (head + 1) * sourceHeadBytes), (index * heads + head + 1) * targetHeadBytes - sourceHeadBytes);
          } else destination.set(source, index * source.byteLength);
        }
        // Copy immediately: the native session detaches its previous logits on the next forward.
      } finally {
        for (const tensor of Object.values(cache)) temporary.add(tensor);
        dispose(temporary);
      }
    }
    check();
    return merged;
  } catch (error) { dispose(Object.values(merged)); throw error; }
}
