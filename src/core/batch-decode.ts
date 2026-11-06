import { DynamicCache, Tensor } from '@huggingface/transformers';

function dispose(tensors: Iterable<Tensor>): void {
  for (const tensor of new Set(tensors)) if (tensor.location !== 'none') tensor.dispose();
}

/** Compact native decode rows without changing generation's logical batch or positions. */
export class BatchDecoder {
  private active: number[];
  private logits: Tensor | undefined;

  constructor(private readonly batchSize: number, private readonly isDone: (logicalRow: number) => boolean) {
    if (!Number.isSafeInteger(batchSize) || batchSize < 1) throw new Error('Invalid decoder batch size');
    this.active = Array.from({ length: batchSize }, (_, index) => index);
  }

  async forward(inputs: Record<string, unknown>, forward: (inputs: Record<string, unknown>) => Promise<Record<string, unknown>>): Promise<Record<string, unknown>> {
    const previous = this.active;
    let retained: number[] | undefined;
    for (let physical = 0; physical < previous.length; physical++) {
      if (this.isDone(previous[physical]!)) retained ??= Array.from({ length: physical }, (_, index) => index);
      else retained?.push(physical);
    }
    const active = retained ? retained.map((physical) => previous[physical]!) : previous;
    if (!active.length) { this.dispose(); throw new Error('Cannot decode an empty active batch'); }
    if (active.length === this.batchSize) return await forward(inputs);

    const temporary = new Set<Tensor>();
    let outputs: Record<string, unknown> | undefined;
    let returned = false;
    try {
      const gather = async (tensor: Tensor, axis: number, indices: readonly number[], expected: number): Promise<Tensor> => {
        if (tensor.dims[axis] !== expected || tensor.dims.some((dim) => !Number.isSafeInteger(dim) || dim < 0)) throw new Error('Invalid batch decoder tensor layout');
        // getData(true) also releases GPU storage: after download its location is CPU,
        // so DynamicCache.update would otherwise no longer release the native resource.
        if (tensor.location !== 'cpu') await tensor.ort_tensor.getData(true);
        const data = tensor.data;
        if (!ArrayBuffer.isView(data) || data instanceof DataView) throw new Error('Batch decoder requires numeric tensor storage');
        const dims = tensor.dims.slice(); dims[axis] = indices.length;
        const Storage = data.constructor as { new(length: number): typeof data };
        const result = new Tensor(tensor.type, new Storage(dims.reduce((total, dim) => total * dim, 1)), dims);
        temporary.add(result);
        const targetData = result.data;
        if (!ArrayBuffer.isView(targetData)) throw new Error('Invalid gathered decoder storage');
        const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const target = new Uint8Array(targetData.buffer, targetData.byteOffset, targetData.byteLength);
        const outer = tensor.dims.slice(0, axis).reduce((total, dim) => total * dim, 1);
        const rowBytes = source.byteLength / outer / expected;
        for (let group = 0; group < outer; group++) for (let row = 0; row < indices.length; row++) {
          const start = (group * expected + indices[row]!) * rowBytes;
          target.set(source.subarray(start, start + rowBytes), (group * indices.length + row) * rowBytes);
        }
        return result;
      };
      const ids = inputs.input_ids; const mask = inputs.attention_mask; const positions = inputs.position_ids;
      const cache = inputs.past_key_values;
      if (!(ids instanceof Tensor) || ids.dims.length !== 2 || ids.dims[1] !== 1
        || !(mask instanceof Tensor) || mask.dims.length !== 2
        || !(positions instanceof Tensor) || positions.dims.length !== 3 || positions.dims[0] !== 3 || positions.dims[2] !== 1
        || !(cache instanceof DynamicCache) || !Object.keys(cache).length) throw new Error('Invalid batch decode input layout');
      const compact: Record<string, unknown> = { ...inputs,
        input_ids: await gather(ids, 0, active, this.batchSize),
        attention_mask: await gather(mask, 0, active, this.batchSize),
        position_ids: await gather(positions, 1, active, this.batchSize),
      };
      if (retained) {
        const entries: Record<string, Tensor> = {};
        for (const [name, tensor] of Object.entries(cache)) {
          if (!(tensor instanceof Tensor) || tensor.dims.length < 1) throw new Error(`Invalid batch decode cache: ${name}`);
          entries[name] = await gather(tensor, 0, retained, previous.length);
        }
        compact.past_key_values = new DynamicCache(entries);
      } else {
        for (const tensor of Object.values(cache)) if (!(tensor instanceof Tensor) || tensor.dims[0] !== active.length) throw new Error('Invalid compacted decoder cache layout');
      }
      this.active = active;
      outputs = await forward(compact);
      const raw = outputs.logits;
      if (!(raw instanceof Tensor) || raw.dims.length !== 3 || raw.dims[0] !== active.length || raw.dims[1] !== 1 || !raw.dims[2]) throw new Error('Invalid compacted decoder logits layout');
      if (raw.location !== 'cpu') await raw.ort_tensor.getData(true);
      const data = raw.data;
      if (!ArrayBuffer.isView(data) || data instanceof DataView) throw new Error('Batch decoder requires numeric logits storage');
      if (!this.logits) {
        const Storage = data.constructor as { new(length: number): typeof data };
        this.logits = new Tensor(raw.type, new Storage(this.batchSize * raw.dims[2]!), [this.batchSize, 1, raw.dims[2]!]);
      }
      if (this.logits.type !== raw.type || this.logits.dims[2] !== raw.dims[2]) throw new Error('Batch decoder logits layout changed');
      const targetData = this.logits.data;
      if (!ArrayBuffer.isView(targetData)) throw new Error('Invalid logical decoder logits storage');
      const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const target = new Uint8Array(targetData.buffer, targetData.byteOffset, targetData.byteLength);
      const rowBytes = source.byteLength / active.length;
      for (let physical = 0; physical < active.length; physical++) target.set(source.subarray(physical * rowBytes, (physical + 1) * rowBytes), active[physical]! * rowBytes);
      // Generation copies the sampled position into separate score storage, so this backing store
      // remains independent of PreviousLogitsRelease and can be reused on the next step.
      const result = { ...outputs, logits: this.logits };
      if (!Object.entries(outputs).some(([name, tensor]) => name !== 'logits' && tensor === raw)) dispose([raw]);
      for (const tensor of Object.values(result)) if (tensor instanceof Tensor) temporary.delete(tensor);
      returned = true;
      return result;
    } catch (error) {
      this.dispose();
      throw error;
    } finally {
      if (!returned && outputs) for (const tensor of Object.values(outputs)) if (tensor instanceof Tensor) temporary.add(tensor);
      dispose(temporary);
    }
  }

  dispose(): void {
    if (this.logits) dispose([this.logits]);
    this.logits = undefined;
  }
}
