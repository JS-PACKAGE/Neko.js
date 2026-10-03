import { StreamingSha256 } from './sha256.js';
import type { Hash } from 'node:crypto';

export type DownloadResetReason = 'source-changed' | 'resume-disabled' | 'validator-unavailable' | 'invalid-partial' | 'range-rejected' | 'validator-changed' | 'integrity';
export interface DownloadProgress {
  loaded: number;
  total: number;
  phase: 'download' | 'verify' | 'resume';
  resumedFrom?: number;
  resetReason?: DownloadResetReason;
}
export interface DownloadIdentity {
  version: 1;
  source: string;
  destination: string;
  size: number;
  sha256: string;
  validator: string | null;
}
export interface DownloadWriter { write(bytes: Uint8Array): Promise<void>; close(): Promise<void>; }
export interface DownloadStage {
  inspect(): Promise<{ identity: DownloadIdentity | null; size: number }>;
  begin(identity: DownloadIdentity, offset: number): Promise<DownloadWriter>;
  read(): Promise<ReadableStream<Uint8Array>>;
  reset(): Promise<void>;
  promote(): Promise<void>;
}
export interface ResumableDownloadOptions {
  source: string;
  size: number;
  sha256: string;
  stage: DownloadStage;
  fetch(headers: Headers, signal?: AbortSignal): Promise<{ response: Response; destination: string }>;
  signal?: AbortSignal;
  resume?: boolean;
  onProgress?: (event: DownloadProgress) => void;
  integrityError?: (detail: string) => Error;
}

function validator(response: Response): string | null {
  const etag = response.headers.get('etag');
  if (etag && /^"[^"\r\n]*"$/.test(etag)) return etag;
  const modified = response.headers.get('last-modified');
  return modified && Number.isFinite(Date.parse(modified)) ? modified : null;
}
function validIdentity(identity: DownloadIdentity, options: ResumableDownloadOptions): boolean {
  return !!identity && identity.version === 1 && identity.source === options.source && identity.size === options.size
    && identity.sha256 === options.sha256 && typeof identity.destination === 'string'
    && (identity.validator === null || typeof identity.validator === 'string');
}

/** A single network attempt. Only a complete, freshly hashed staged object can be promoted. */
export async function downloadResumable(options: ResumableDownloadOptions): Promise<void> {
  const { stage, signal } = options;
  const failure = (detail: string) => options.integrityError?.(detail) ?? new Error(`Download integrity mismatch: ${detail}`);
  const emit = (event: DownloadProgress) => options.onProgress?.(event);
  signal?.throwIfAborted();
  let partial = await stage.inspect();
  const reset = async (reason: DownloadResetReason) => {
    const resumedFrom = partial.size;
    await stage.reset();
    partial = { identity: null, size: 0 };
    emit({ phase: 'resume', loaded: 0, total: options.size, resumedFrom, resetReason: reason });
  };
  if (!partial.identity || !validIdentity(partial.identity, options)) {
    if (partial.size || partial.identity) await reset(partial.identity?.source !== options.source ? 'source-changed' : 'invalid-partial');
  } else if (!Number.isSafeInteger(partial.size) || partial.size < 0 || partial.size > options.size) await reset('invalid-partial');
  else if (partial.size > 0 && (options.resume === false || partial.size < options.size && !partial.identity.validator)) await reset(options.resume === false ? 'resume-disabled' : 'validator-unavailable');

  let loaded = partial.size;
  if (loaded < options.size) {
    const headers = new Headers();
    // A byte range refers to the identity representation, never a compressed transfer offset.
    headers.set('accept-encoding', 'identity');
    if (loaded) {
      headers.set('range', `bytes=${loaded}-`);
      headers.set('if-range', partial.identity!.validator!);
    }
    const { response, destination } = await options.fetch(headers, signal);
    const cancel = async () => { await response.body?.cancel().catch(() => undefined); };
    if (!response.ok) {
      await cancel();
      if (loaded && response.status === 416) await reset('range-rejected');
      throw new Error(`Model download failed: HTTP ${response.status}`);
    }
    const currentValidator = validator(response);
    if (loaded && response.status === 200) {
      await reset(destination !== partial.identity?.destination ? 'source-changed' : 'validator-changed');
      loaded = 0;
    } else if (loaded) {
      const expectedRange = `bytes ${loaded}-${options.size - 1}/${options.size}`;
      if (response.status !== 206 || response.headers.get('content-range') !== expectedRange
        || destination !== partial.identity!.destination || currentValidator !== partial.identity!.validator) {
        await cancel();
        await reset(destination !== partial.identity?.destination ? 'source-changed' : currentValidator !== partial.identity?.validator ? 'validator-changed' : 'range-rejected');
        throw failure('invalid resume response');
      }
      const encoding = response.headers.get('content-encoding');
      if (encoding && encoding.toLowerCase() !== 'identity') {
        await cancel(); await reset('range-rejected'); throw failure('encoded resume response');
      }
      emit({ phase: 'resume', loaded, total: options.size, resumedFrom: loaded });
    } else if (response.status !== 200 || response.headers.has('content-range')) {
      await cancel(); await reset('range-rejected'); throw failure('unexpected partial response');
    }
    const length = response.headers.get('content-length');
    if (length !== null && (!/^\d+$/.test(length) || Number(length) !== options.size - loaded)) {
      await cancel(); await reset('range-rejected'); throw failure('declared size');
    }
    if (!response.body) { await cancel(); throw failure('missing body'); }
    const identity: DownloadIdentity = { version: 1, source: options.source, destination, size: options.size, sha256: options.sha256, validator: currentValidator };
    let writer: DownloadWriter;
    try { writer = await stage.begin(identity, loaded); }
    catch (error) { await cancel(); throw error; }
    const reader = response.body.getReader();
    const abort = () => { void reader.cancel(signal?.reason).catch(() => undefined); };
    signal?.addEventListener('abort', abort, { once: true });
    let invalid = false;
    let interrupted = false;
    try {
      while (true) {
        signal?.throwIfAborted();
        const { done, value } = await reader.read();
        if (done) break;
        if (value.byteLength > options.size - loaded) { invalid = true; throw failure('exceeds pinned size'); }
        await writer.write(value);
        loaded += value.byteLength;
        emit({ phase: 'download', loaded, total: options.size });
      }
      signal?.throwIfAborted();
      // A truncated connection remains resumable; it is never considered installed.
      if (loaded !== options.size) throw failure('incomplete body');
    } catch (error) {
      interrupted = true;
      await reader.cancel(error).catch(() => undefined);
      throw error;
    } finally {
      signal?.removeEventListener('abort', abort);
      reader.releaseLock();
      await writer.close();
      if (invalid || options.resume === false && interrupted) await stage.reset();
    }
  }

  signal?.throwIfAborted();
  let hash: StreamingSha256 | Hash;
  if (typeof process !== 'undefined' && process.release?.name === 'node') {
    // Native incremental hashing is Node-only and avoids hashing large model files in JavaScript.
    const protocol = 'node:';
    const { createHash } = await import(`${protocol}crypto`);
    hash = createHash('sha256');
  } else hash = new StreamingSha256();
  const body = await stage.read();
  const reader = body.getReader();
  const abort = () => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener('abort', abort, { once: true });
  let verified = 0;
  let invalid = false;
  try {
    while (true) {
      signal?.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) break;
      if (value.byteLength > options.size - verified) { invalid = true; throw failure('staged size'); }
      hash.update(value);
      verified += value.byteLength;
      emit({ phase: 'verify', loaded: verified, total: options.size });
    }
    signal?.throwIfAborted();
    const digest = hash instanceof StreamingSha256 ? hash.digest() : hash.digest('hex');
    if (verified !== options.size || digest !== options.sha256) { invalid = true; throw failure('size or SHA-256'); }
  } catch (error) {
    await reader.cancel(error).catch(() => undefined);
    throw error;
  } finally {
    signal?.removeEventListener('abort', abort);
    reader.releaseLock();
    if (invalid) await reset('integrity');
  }
  signal?.throwIfAborted();
  await stage.promote();
}
