import { LogitsProcessor } from '@huggingface/transformers';

export interface GenerationOptions {
  sampling?: boolean;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  noRepeatNgramSize?: number;
  stop?: string[];
  stopTokenIds?: number[];
}
export function generationSettings(input: GenerationOptions = {}, vocabularySize: number) {
  if (typeof input !== 'object' || input === null || Array.isArray(input)) throw new TypeError('generation must be an object');
  const allowed: Record<keyof GenerationOptions, true> = { sampling: true, temperature: true, topK: true, topP: true, repetitionPenalty: true, noRepeatNgramSize: true, stop: true, stopTokenIds: true };
  for (const key of Object.keys(input)) if (!Object.hasOwn(allowed, key)) throw new TypeError(`Unsupported generation setting: ${key}`);
  if (input.sampling !== undefined && typeof input.sampling !== 'boolean') throw new TypeError('sampling must be boolean');
  const range = (name: string, value: number, min: number, max: number, integer = false) => {
    if (!Number.isFinite(value) || value < min || value > max || integer && !Number.isSafeInteger(value)) throw new RangeError(`${name} must be between ${min} and ${max}`);
    return value;
  };
  const temperature = range('temperature', input.temperature ?? 1, 0.01, 5);
  const topK = range('topK', input.topK ?? 50, 0, vocabularySize, true);
  const topP = range('topP', input.topP ?? 1, 0.000001, 1);
  const repetitionPenalty = range('repetitionPenalty', input.repetitionPenalty ?? 1, 0.1, 10);
  const noRepeatNgramSize = range('noRepeatNgramSize', input.noRepeatNgramSize ?? 0, 0, 32, true);
  const stop = input.stop ?? [];
  if (!Array.isArray(stop) || stop.length > 16 || stop.some((text) => typeof text !== 'string' || text.length < 1 || text.length > 256)) throw new TypeError('stop must contain at most 16 nonempty strings of at most 256 characters');
  const stopTokenIds = input.stopTokenIds ?? [];
  if (!Array.isArray(stopTokenIds) || stopTokenIds.length > 64 || stopTokenIds.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= vocabularySize)) throw new TypeError('stopTokenIds must contain at most 64 vocabulary token IDs');
  if (!input.sampling && (input.temperature !== undefined || input.topK !== undefined || input.topP !== undefined)) throw new TypeError('temperature, topK, and topP require sampling:true');
  return { sampling: input.sampling ?? false, temperature, topK, topP, repetitionPenalty, noRepeatNgramSize, stop, stopTokenIds };
}

interface GenerationLogits { dims: number[]; data: ArrayLike<number | bigint>; }
interface NucleusFilter {
  <T extends GenerationLogits>(ids: bigint[][], logits: T): T;
  _call<T extends GenerationLogits>(ids: bigint[][], logits: T): T;
}
/** Transformers 4.2 has no nucleus warper. Filter using the same temperature/top-K distribution its sampler uses. */
class NucleusProcessorImplementation extends LogitsProcessor {
  private order?: Uint32Array;
  constructor(private readonly p: number, private readonly temperature: number, private readonly topK: number) { super(); }
  override _call<T extends GenerationLogits>(_ids: bigint[][], logits: T): T {
    const width = logits.dims.at(-1)!;
    this.order ??= new Uint32Array(width);
    if (this.order.length !== width) throw new Error('Generation vocabulary changed');
    const values = logits.data as Float32Array;
    for (let offset = 0; offset < values.length; offset += width) {
      const order = this.order;
      for (let i = 0; i < width; i++) order[i] = i;
      order.sort((a, b) => values[offset + b]! - values[offset + a]!);
      const count = this.topK === 0 ? width : Math.min(this.topK, width);
      const max = values[offset + order[0]!]!;
      let total = 0;
      for (let i = 0; i < count; i++) total += Math.exp((values[offset + order[i]!]! - max) / this.temperature);
      let cumulative = 0;
      let kept = 0;
      do { cumulative += Math.exp((values[offset + order[kept]!]! - max) / this.temperature); kept++; } while (kept < count && cumulative < this.p * total);
      for (let i = kept; i < width; i++) values[offset + order[i]!] = -Infinity;
    }
    return logits;
  }
}
export const NucleusProcessor: new (p: number, temperature: number, topK: number) => NucleusFilter = NucleusProcessorImplementation;

/** Retain only the suffix that could still become a stop, never leaking stop text across chunk boundaries. */
export class StopBuffer {
  private pending = '';
  stopped = false;
  constructor(private readonly stops: readonly string[], private readonly deliver: (text: string) => void) {}
  push(text: string): void {
    if (this.stopped) return;
    this.pending += text;
    let index = Infinity;
    for (const stop of this.stops) { const found = this.pending.indexOf(stop); if (found >= 0) index = Math.min(index, found); }
    if (index !== Infinity) { this.emit(this.pending.slice(0, index)); this.pending = ''; this.stopped = true; return; }
    let retained = 0;
    for (const stop of this.stops) for (let length = 1; length < stop.length; length++) if (this.pending.endsWith(stop.slice(0, length))) retained = Math.max(retained, length);
    const safe = this.pending.length - retained;
    this.emit(this.pending.slice(0, safe));
    this.pending = this.pending.slice(safe);
  }
  end(): void { if (!this.stopped) this.emit(this.pending); this.pending = ''; }
  private emit(text: string): void { if (text) this.deliver(text); }
}
