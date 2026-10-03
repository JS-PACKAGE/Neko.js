import { NekoError } from '../errors.js';
import type { ModelProfileId } from '../cache/manifest.js';
import { getRegisteredModelProfile } from '../cache/registry.js';

export type BackendDevice = 'webgpu' | 'cpu';
export interface AdapterInfo {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  isFallbackAdapter?: boolean;
}
export interface BackendInfo {
  runtime: 'browser' | 'node';
  device: BackendDevice;
  adapter?: AdapterInfo;
  executionProviders: string[];
  gpuMemoryBytes: null;
  adapterEvidence: 'availability-probe' | null;
  /** Pinned model/runtime compatibility, not installed provider or driver availability. */
  supported: boolean;
  capabilityEvidence: 'model-runtime-compatibility';
  limitation?: string;
}

/** Describes configured providers, not whether every model operator executes on GPU. */
export async function inspectBackend(device: BackendDevice, profile: ModelProfileId = 'default'): Promise<BackendInfo> {
  if (device !== 'cpu' && device !== 'webgpu') throw new TypeError('Unknown backend device');
  getRegisteredModelProfile(profile);
  const candidate = globalThis as typeof globalThis & { process?: { release?: { name?: string } } };
  const runtime = candidate.process?.release?.name === 'node' ? 'node' : 'browser';
  const result: BackendInfo = {
    runtime, device, executionProviders: [device === 'cpu' && runtime === 'browser' ? 'wasm' : device], gpuMemoryBytes: null, adapterEvidence: null,
    supported: !(runtime === 'browser' && device === 'cpu'),
    capabilityEvidence: 'model-runtime-compatibility',
    ...(runtime === 'browser' && device === 'cpu' ? { limitation: 'The pinned model requires GatherBlockQuantized(1), unavailable in this WASM runtime' } : {}),
  };
  // Node uses the native ONNX Runtime provider and does not require navigator.gpu.
  if (runtime === 'node' || device === 'cpu') return result;
  const navigator = globalThis.navigator as unknown as { gpu?: {
    requestAdapter(): Promise<{ info?: AdapterInfo; features: { has(feature: string): boolean } } | null>;
  } } | undefined;
  if (!navigator?.gpu) throw new NekoError('WebGPU API is not present. CPU/WASM does not support this pinned model; use a WebGPU-capable secure browser or supported Node runtime.', 'backend', 'UNSUPPORTED_BACKEND');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new NekoError('WebGPU adapter is unavailable. CPU/WASM does not support this pinned model.', 'backend', 'UNSUPPORTED_BACKEND');
  if (!adapter.features.has('shader-f16')) throw new NekoError('WebGPU adapter does not support shader-f16 required conservatively for the pinned model graphs', 'backend', 'UNSUPPORTED_BACKEND');
  result.adapterEvidence = 'availability-probe';
  if (adapter.info) {
    const { vendor, architecture, device, description, isFallbackAdapter } = adapter.info;
    result.adapter = {
      ...(vendor === undefined ? {} : { vendor }), ...(architecture === undefined ? {} : { architecture }),
      ...(device === undefined ? {} : { device }), ...(description === undefined ? {} : { description }),
      ...(isFallbackAdapter === undefined ? {} : { isFallbackAdapter }),
    };
  }
  return result;
}
