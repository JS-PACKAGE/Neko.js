import { ImageProcessor, Qwen2VLImageProcessor, Tensor, cat, type RawImage } from '@huggingface/transformers';

/** Strided copy equal to `Tensor.permute(...axes)` data, walking the output in order instead of dividing every input index per axis. */
export function permuteData<T extends Float32Array>(data: T, dims: readonly number[], axes: readonly number[]): T {
  const rank = dims.length;
  const inputStrides = new Array<number>(rank);
  for (let axis = rank - 1, size = 1; axis >= 0; axis--) { inputStrides[axis] = size; size *= dims[axis]!; }
  const shape = axes.map((axis) => dims[axis]!); const strides = axes.map((axis) => inputStrides[axis]!);
  const output = new (data.constructor as new (length: number) => T)(data.length);
  if (!data.length) return output;
  const innerSize = shape[rank - 1]!; const innerStride = strides[rank - 1]!;
  const counters = new Array<number>(rank).fill(0);
  for (let written = 0, offset = 0; ;) {
    for (let index = 0, source = offset; index < innerSize; index++, source += innerStride) output[written++] = data[source]!;
    let axis = rank - 2;
    for (; axis >= 0; axis--) {
      offset += strides[axis]!;
      if (++counters[axis]! < shape[axis]!) break;
      offset -= strides[axis]! * shape[axis]!; counters[axis] = 0;
    }
    if (axis < 0) return output;
  }
}

type PatchProcessor = Qwen2VLImageProcessor & { config: { temporal_patch_size: number; merge_size: number; patch_size: number } };
/**
 * Transformers.js 4.2.0's `Qwen2VLImageProcessor` flattens vision patches with a generic 9-D `Tensor.permute` that divides every
 * element index by each axis (about 1 s per 4000×3000 photo on CPU). This replaces `_call` on the given instance only, and only for
 * that exact class, with the same computation using `permuteData`; pixel values, grids and sizes are bit-identical.
 */
export function useFastPatchFlattening(processor: unknown): void {
  if (!(processor instanceof Qwen2VLImageProcessor) || Object.getPrototypeOf(processor) !== Qwen2VLImageProcessor.prototype) return;
  const instance = processor as PatchProcessor;
  const resize = ImageProcessor.prototype._call;
  Object.defineProperty(instance, '_call', { configurable: true, writable: true, value: async (images: RawImage[], ...args: unknown[]) => {
    const { pixel_values, original_sizes, reshaped_input_sizes } = await resize.call(instance, images, ...args);
    const { temporal_patch_size, merge_size, patch_size } = instance.config;
    let patches = pixel_values as Tensor;
    if (patches.dims[0] === 1) patches = cat(Array.from({ length: temporal_patch_size }, () => patches), 0);
    const gridT = patches.dims[0]! / temporal_patch_size; const channel = patches.dims[1]!;
    const gridH = Math.floor(patches.dims[2]! / patch_size); const gridW = Math.floor(patches.dims[3]! / patch_size);
    // view() validates the element count exactly as upstream does before its permute.
    const view = patches.view(gridT, temporal_patch_size, channel, Math.floor(gridH / merge_size), merge_size, patch_size, Math.floor(gridW / merge_size), merge_size, patch_size);
    const flattened = permuteData(view.data as Float32Array, view.dims, [0, 3, 6, 4, 7, 2, 1, 5, 8]);
    return {
      pixel_values: new Tensor(view.type, flattened, [gridT * gridH * gridW, channel * temporal_patch_size * patch_size * patch_size]),
      image_grid_thw: new Tensor('int64', [gridT, gridH, gridW], [1, 3]),
      original_sizes, reshaped_input_sizes,
    };
  } });
}
