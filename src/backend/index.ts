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
}

/** Describes configured providers, not whether every model operator executes on GPU. */
export async function inspectBackend(device: BackendDevice): Promise<BackendInfo> {
  if (device !== 'cpu' && device !== 'webgpu') throw new TypeError('Unknown backend device');
  const candidate = globalThis as typeof globalThis & { process?: { release?: { name?: string } } };
  const runtime = candidate.process?.release?.name === 'node' ? 'node' : 'browser';
  const result: BackendInfo = {
    runtime, device, executionProviders: [device === 'cpu' && runtime === 'browser' ? 'wasm' : device], gpuMemoryBytes: null, adapterEvidence: null,
  };
  // Node uses the native ONNX Runtime provider and does not require navigator.gpu.
  if (runtime === 'node' || device === 'cpu') return result;
  const navigator = globalThis.navigator as unknown as { gpu?: {
    requestAdapter(): Promise<{ info?: AdapterInfo; features: { has(feature: string): boolean } } | null>;
  } } | undefined;
  if (!navigator?.gpu) throw new Error('WebGPU API is not present; select CPU explicitly to use WASM');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter is unavailable; select CPU explicitly to use WASM');
  if (!adapter.features.has('shader-f16')) throw new Error('WebGPU adapter does not support shader-f16 required by the fp16 vision encoder');
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
