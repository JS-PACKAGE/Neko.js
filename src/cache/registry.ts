import { ALL_MODEL_FILES, MODEL_ID, MODEL_REVISION, MODEL_DTYPE, type ModelFileName, type ModelProfileId, type ModelDtype } from './manifest.js';

export type ModelId = typeof MODEL_ID | 'onnx-community/Qwen3.5-2B-ONNX-OPT';
export type ModelAssetManifest = Readonly<Record<ModelFileName, { readonly size: number; readonly sha256: string }>>;
export interface RegisteredModelProfile {
  readonly id: ModelId;
  readonly revision: string;
  readonly profile: ModelProfileId;
  readonly dtype: ModelDtype;
  readonly files: Readonly<Partial<ModelAssetManifest>>;
  readonly allFiles: ModelAssetManifest;
  readonly baseUrl: string;
  readonly license: 'apache-2.0';
  readonly architecture: 'Qwen3_5ForConditionalGeneration';
}
// LFS hashes: immutable Hub API revision. Non-LFS metadata: SHA-256 of raw pinned bytes.
const alternativeFiles: ModelAssetManifest = {
  ...ALL_MODEL_FILES,
  'config.json': { size: 2993, sha256: 'b028de63b0ed8b37107acaaf1475d40d6d4feb5721153674e7d1d0bdbfd0f258' },
  'onnx/embed_tokens_q4.onnx': { size: 857, sha256: '0255dd844858758f452d9678f1e2c91db178f21b7da254d29b12e7fe3d23e305' },
  'onnx/embed_tokens_q4.onnx_data': { size: 325795840, sha256: '9a6404d9b1c79ffc038d5c28deec04d419efb4594125453c991be38b4522ecb6' },
  'onnx/decoder_model_merged_q4.onnx': { size: 586260, sha256: '8207fcf67d692e251eec7f24868be5b753f402fb84a259eb4578253692701405' },
  'onnx/decoder_model_merged_q4.onnx_data': { size: 1207357440, sha256: '0c0f71175574f0060ed441acb1896825234ef27fcf008822c490dbfa6b0c90fe' },
  'onnx/vision_encoder_fp16.onnx': { size: 345836, sha256: 'b4e8366cfa40715e12021c682e7d629a9be1d39187d08939c91bb4037366440f' },
  'onnx/vision_encoder_fp16.onnx_data': { size: 667748352, sha256: 'ce9c4352f07aedb7daebdfdb9a723fa2ca3017b56e609a512f65318ceb052563' },
  'onnx/vision_encoder_q4.onnx': { size: 339182, sha256: '986f2792878369c6e0e28e27de0c14a41227a3fd24b9da8eae66d93bf223ae66' },
  'onnx/vision_encoder_q4.onnx_data': { size: 217952256, sha256: '0ea0ab9559904e1e5150a0ca194136922c6b8f1dfaaa44cc5e174ca59b231bb3' },
};
function freezeManifest(files: ModelAssetManifest): ModelAssetManifest {
  for (const file of Object.values(files)) Object.freeze(file);
  return Object.freeze(files);
}
const defaultFiles = freezeManifest({ ...ALL_MODEL_FILES });
freezeManifest(alternativeFiles);
export const MODEL_REGISTRY = Object.freeze({
  [MODEL_ID]: Object.freeze({ id: MODEL_ID, revision: MODEL_REVISION, files: defaultFiles, license: 'apache-2.0', architecture: 'Qwen3_5ForConditionalGeneration', upstream: 'Qwen/Qwen3.5-0.8B' }),
  'onnx-community/Qwen3.5-2B-ONNX-OPT': Object.freeze({ id: 'onnx-community/Qwen3.5-2B-ONNX-OPT', revision: '2ea7886f48b926aca97de8b0e041ffca7e3ebaa9', files: alternativeFiles, license: 'apache-2.0', architecture: 'Qwen3_5ForConditionalGeneration', upstream: 'Qwen/Qwen3.5-2B' }),
} as const);
const profiles: Record<string, RegisteredModelProfile> = Object.create(null);
for (const model of Object.values(MODEL_REGISTRY)) {
  for (const profile of ['default', 'all-q4'] as const) {
    const files = { ...model.files } as Partial<ModelAssetManifest>;
    const excluded = profile === 'default' ? 'q4' : 'fp16';
    delete files[`onnx/vision_encoder_${excluded}.onnx`];
    delete files[`onnx/vision_encoder_${excluded}.onnx_data`];
    profiles[`${model.id}:${profile}`] = Object.freeze({ id: model.id, revision: model.revision, profile,
      dtype: Object.freeze({ ...MODEL_DTYPE, vision_encoder: profile === 'default' ? 'fp16' : 'q4' }),
      files: Object.freeze(files), allFiles: model.files, baseUrl: `https://huggingface.co/${model.id}/resolve/${model.revision}/`, license: model.license, architecture: model.architecture });
  }
}
Object.freeze(profiles);
export function getRegisteredModelProfile(profile: ModelProfileId = 'default', model: ModelId = MODEL_ID): RegisteredModelProfile {
  if (typeof model !== 'string' || (profile !== 'default' && profile !== 'all-q4')) throw new TypeError('Unknown registered model or profile');
  const selected = profiles[`${model}:${profile}`];
  if (!selected) throw new TypeError('Unknown registered model or profile');
  return selected;
}
export function isRegisteredModelUrl(url: URL, files: Readonly<Record<string, unknown>>): boolean {
  for (const model of Object.values(MODEL_REGISTRY)) {
    const base = `https://huggingface.co/${model.id}/resolve/${model.revision}/`;
    if (url.href.startsWith(base)) {
      const name = url.href.slice(base.length);
      if (!Object.hasOwn(model.files, name) || !Object.hasOwn(files, name)) return false;
      const expected = model.files[name as ModelFileName];
      const supplied = files[name];
      return !!supplied && typeof supplied === 'object' && 'size' in supplied && 'sha256' in supplied && supplied.size === expected.size && supplied.sha256 === expected.sha256;
    }
  }
  return false;
}
