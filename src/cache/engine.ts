export interface CachedEngine { readonly memory: number | null; dispose(): Promise<void>; }
export interface EngineCacheStatus { loaded: boolean; sessions: number; memory: number | null; hits: number; loads: number; }

/** Owns one engine and serializes requests so session history is never shared concurrently. */
export class EngineCache<T extends CachedEngine> {
  private engine: T | undefined;
  private queue: Promise<unknown> = Promise.resolve();
  private timer?: Parameters<typeof globalThis.clearTimeout>[0];
  private hits = 0;
  private loads = 0;
  constructor(private readonly load: () => Promise<T>, private readonly enabled = true, private readonly ttlMs = 30 * 60_000) {
    if (!Number.isSafeInteger(ttlMs) || ttlMs < 0 || ttlMs > 2_147_483_647) throw new RangeError('engineTtlMs must be an integer between 0 and 2147483647');
  }
  status(): EngineCacheStatus { return { loaded: !!this.engine, sessions: this.engine ? 1 : 0, memory: this.engine ? this.engine.memory : 0, hits: this.hits, loads: this.loads }; }
  async use<R>(operation: (engine: T) => Promise<R>): Promise<R> {
    const work = this.queue.catch(() => undefined).then(async () => {
      globalThis.clearTimeout(this.timer);
      this.timer = undefined;
      if (this.engine) this.hits++;
      else { this.engine = await this.load(); this.loads++; }
      try { return await operation(this.engine); }
      finally {
        if (!this.enabled || this.ttlMs === 0) { const engine = this.engine; this.engine = undefined; await engine.dispose(); }
        else {
          this.timer = globalThis.setTimeout(() => {
            void this.release().catch(() => { console.warn('Engine cache scheduled release failed'); });
          }, this.ttlMs);
        }
      }
    });
    this.queue = work;
    return work;
  }
  async release(afterRelease?: () => Promise<void>): Promise<void> {
    const work = this.queue.catch(() => undefined).then(async () => {
      globalThis.clearTimeout(this.timer);
      this.timer = undefined;
      const engine = this.engine;
      this.engine = undefined;
      await engine?.dispose();
      await afterRelease?.();
    });
    this.queue = work;
    await work;
  }
}
