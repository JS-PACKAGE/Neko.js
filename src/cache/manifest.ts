export const MODEL_ID = 'onnx-community/Qwen3.5-0.8B-ONNX-OPT';
export const MODEL_REVISION = 'fafab72d87a9e6be3925b38caf48286d2838f2d0';
export const MODEL_DTYPE = { embed_tokens: 'q4', decoder_model_merged: 'q4', vision_encoder: 'fp16' } as const;

// ONNX/LFS digests come from the pinned Hub revision; metadata digests are SHA-256 of its raw files.
export const MODEL_FILES = {
  'onnx/embed_tokens_q4.onnx': { size: 857, sha256: '8773dcf4858f855bfce13def4356ca17e0bc516a477154496b6ffb6dd6d084dc' },
  'onnx/embed_tokens_q4.onnx_data': { size: 162897920, sha256: '9210b1d26eb14136d3522584d17da14cc5f4b82b6f5607e6d80d59424a8cbe99' },
  'onnx/decoder_model_merged_q4.onnx': { size: 576621, sha256: '7390858c80a67275d8cb81fb70f66a24d3b86fe333057c1312de09c184a2d41b' },
  'onnx/decoder_model_merged_q4.onnx_data': { size: 483655680, sha256: '1a9165072dd51a9b6a917b6357a3686372b7429313022802cd4c9a730e8c9749' },
  'onnx/vision_encoder_fp16.onnx': { size: 187476, sha256: '7343242b844eed7288a7f6ef3f0dc57a7ceb5118c3f50547bb0458de5dfea645' },
  'onnx/vision_encoder_fp16.onnx_data': { size: 204798464, sha256: '83a2b4babd9a146dd28cb925609413422f0ba0f47474c3e4704620f1190fbd25' },
  'tokenizer.json': { size: 19226111, sha256: '89da80cc6689bef4d90cc1028249436975ffb0814618f1d93c65310e05801a9b' },
  'tokenizer_config.json': { size: 9161, sha256: 'fccbff64ebe09343aa2171028657f5b038db96fb4f657609bc76743eddfa3b9d' },
  'config.json': { size: 2849, sha256: '36fed6a902ccd06ef19a452bd5a0750bd88fe347d06ab75ef515615bac5b296d' },
  'processor_config.json': { size: 1300, sha256: '14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b' },
  'preprocessor_config.json': { size: 336, sha256: '6a970fd06f30e6943b3e2c14d5d3b42d49b06cf99b99103d56689bef462d90f8' },
  'generation_config.json': { size: 248, sha256: 'dc0cbe66543f310896469b7b1448af792f403293a1080baaf04d586c57b23e48' },
  'chat_template.jinja': { size: 7755, sha256: '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80' },
} as const;

export const ALL_MODEL_FILES = {
  ...MODEL_FILES,
  'onnx/vision_encoder_q4.onnx': { size: 185278, sha256: '9b62022e77de4b22ca0bbc4453083c0cdeb8f967ccf13931d4c24c6e7c776177' },
  'onnx/vision_encoder_q4.onnx_data': { size: 68267008, sha256: '98aebedf02fc5414fd1c7f06a6580b42e272600ace9ecd33bfbc479a0c541c64' },
} as const;
export type ModelProfileId = 'default' | 'all-q4';
export type ModelDtype = { embed_tokens: 'q4'; decoder_model_merged: 'q4'; vision_encoder: 'fp16' | 'q4' };

export type ModelFileName = keyof typeof ALL_MODEL_FILES;
export const MODEL_BASE_URL = `https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/`;
export function modelFileUrl(file: ModelFileName): string { return `${MODEL_BASE_URL}${file}`; }
