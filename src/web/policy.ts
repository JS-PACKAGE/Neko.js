import { MODEL_FILES } from '../cache/manifest.js';
import { isRegisteredModelUrl, isRegisteredResolveCacheUrl } from '../cache/registry.js';
import { awaitUser, NekoError } from '../errors.js';
import { workerBootstrapUrl } from '../runtime/context.js';

export type ResourceKind = 'model' | 'runtime' | 'page' | 'image' | 'worker';
export interface ResourcePolicy {
  /** Normal return approves; throwing (including undefined) denies. False also denies. */
  network?: (url: URL, kind: ResourceKind) => void | boolean | Promise<void | boolean>;
  /** Called with a canonical Node path before reading local input bytes. Default is deny. */
  localFiles?: (path: string) => void | boolean | Promise<void | boolean>;
}
const runtimeFiles: Record<string, true> = {
  'ort-wasm-simd-threaded.wasm': true,
  'ort-wasm-simd-threaded.mjs': true,
  'ort-wasm-simd-threaded.asyncify.wasm': true,
  'ort-wasm-simd-threaded.asyncify.mjs': true,
};
interface ModelTransfer { modelFiles?: Readonly<Record<string, unknown>>; modelSource?: URL; modelMirror?: URL | undefined; }
const modelCdnHosts: Record<string, true> = { 'cdn-lfs.huggingface.co': true, 'cdn-lfs-us-1.hf.co': true, 'cdn-lfs-eu-1.hf.co': true, 'cas-bridge.xethub.hf.co': true, 'us.aws.cdn.hf.co': true };
function pinnedModelUrl(url: URL, files: Readonly<Record<string, unknown>>): boolean {
  return isRegisteredModelUrl(url, files) || isRegisteredResolveCacheUrl(url, files);
}
function trustedResource(url: URL, kind: ResourceKind, transfer?: ModelTransfer): boolean {
  if (kind === 'model') {
    const files = transfer?.modelFiles ?? MODEL_FILES;
    return pinnedModelUrl(url, files) || (!!transfer?.modelSource && pinnedModelUrl(transfer.modelSource, files) && (url.href === transfer.modelMirror?.href || url.protocol === 'https:' && !url.username && !url.password && Object.hasOwn(modelCdnHosts, url.hostname)));
  }
  if (kind === 'worker') return url.href === workerBootstrapUrl().href;
  if (kind !== 'runtime') return false;
  const name = url.pathname.slice(url.pathname.lastIndexOf('/') + 1);
  return Object.hasOwn(runtimeFiles, name) && url.href === new URL(`./assets/${name}`, import.meta.url).href;
}
export async function authorizeNetwork(policy: ResourcePolicy | undefined, url: URL, kind: ResourceKind, offline = false, transfer?: ModelTransfer, signal?: AbortSignal): Promise<void> {
  const trusted = trustedResource(url, kind, transfer);
  if (!['http:', 'https:', 'file:', 'blob:'].includes(url.protocol) || url.username || url.password) throw new NekoError('Resource protocol is denied', 'preprocess', 'POLICY_DENIED');
  if (offline && (!(kind === 'runtime' || kind === 'worker') || !trusted)) throw new NekoError('Offline policy permits only bundled runtime bootstrap resources', 'preprocess', 'POLICY_DENIED');
  if (policy?.network) {
    const approved = await awaitUser(() => policy.network!(new URL(url), kind), signal, 'preprocess');
    if (approved === false) throw new NekoError('Network policy denied resource', 'preprocess', 'POLICY_DENIED');
    if (approved !== undefined && approved !== true) throw new TypeError('Network policy must approve normally or deny by throwing/false');
  } else if (!trusted) throw new NekoError(`Network access requires explicit approval (${kind})`, 'preprocess', 'POLICY_DENIED');
}
export function policyDestination(policy: ResourcePolicy | undefined, kind: ResourceKind, offline = false, extra?: (url: URL) => void | Promise<void>, signal?: AbortSignal): (url: URL) => Promise<void> {
  return async (url) => { await authorizeNetwork(policy, url, kind, offline, undefined, signal); if (extra) await awaitUser(() => extra(new URL(url)), signal, 'preprocess'); };
}
export async function authorizeLocalFile(policy: ResourcePolicy | undefined, path: string, signal?: AbortSignal): Promise<void> {
  if (!policy?.localFiles) throw new NekoError('Local input files require explicit approval', 'image', 'POLICY_DENIED');
  const approved = await awaitUser(() => policy.localFiles!(path), signal, 'image');
  if (approved === false) throw new NekoError('Local file policy denied resource', 'image', 'POLICY_DENIED');
  if (approved !== undefined && approved !== true) throw new TypeError('Local file policy must approve normally or deny by throwing/false');
}
