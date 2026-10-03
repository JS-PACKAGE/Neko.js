import { createNeko } from '../dist/node/index.js';

const arguments_ = process.argv.slice(2);
if (arguments_.length > 1 || (arguments_[0] && !arguments_[0].startsWith('--model-profile='))) {
  throw new TypeError('Usage: NEKO_MODEL_CACHE=<cache root> node scripts/prefetch-model.mjs [--model-profile=default|all-q4]');
}
const modelProfile = arguments_[0]?.slice('--model-profile='.length) ?? 'default';
if (!['default', 'all-q4'].includes(modelProfile)) throw new TypeError('--model-profile must be default or all-q4');
if (!process.env.NEKO_MODEL_CACHE) throw new TypeError('NEKO_MODEL_CACHE must name an SDK model cache root');
const neko = await createNeko({ device: 'cpu', modelProfile, cacheDir: process.env.NEKO_MODEL_CACHE });
try {
  await neko.cache.model.prefetch();
  console.log(JSON.stringify({ modelProfile, pinnedAssetsVerified: true, inferencePerformed: false }));
} finally {
  await neko.dispose();
}
