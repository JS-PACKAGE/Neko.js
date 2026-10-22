import { copyFile, cp, mkdir, readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const dist = join(root, 'dist');
const browserEntries = [
  ['src/index.ts', 'neko.js'],
  ['src/runtime/worker.ts', 'worker.js'],
  ['src/web/index.ts', 'web.js'],
  ['src/report/index.ts', 'report.js'],
  ['src/backend/index.ts', 'backend.js'],
  ['src/types.ts', 'types.js'],
  ['src/documents/index.ts', 'documents.js'],
];
const nodeEntries = {
  index: 'src/index.ts',
  worker: 'src/runtime/worker.ts',
  types: 'src/types.ts',
  web: 'src/web/index.ts',
  report: 'src/report/index.ts',
  backend: 'src/backend/index.ts',
  documents: 'src/documents/index.ts',
};

// Transformers.js 4.2.0's greedy sampler takes the argmax with its shared `max()`, whose call sites see arrays of many element types, so
// V8 runs it as a slow polymorphic loop: about 0.8 ms per token over the 248,320-entry vocabulary, against about 0.1 ms for a loop that
// only ever sees Float32Array. Same comparisons and first-index tie-breaking; the build fails if the upstream line ever changes.
const greedyArgmax = {
  name: 'neko-greedy-argmax',
  setup(build) {
    build.onLoad({ filter: /[\\/]@huggingface[\\/]transformers[\\/]dist[\\/]transformers\.(node\.mjs|web\.js)$/ }, async ({ path }) => {
      const source = await readFile(path, 'utf8');
      const target = 'const argmax = max(logits.data)[1];';
      if (source.split(target).length !== 2) throw new Error(`Greedy sampler argmax not found exactly once in ${path}`);
      return { loader: 'js', contents: `${source.replace(target, 'const argmax = nekoGreedyArgmax(logits.data);')}
function nekoGreedyArgmax(data) {
  if (data.length === 0) throw Error("Array must not be empty");
  let best = data[0];
  let index = 0;
  for (let i = 1; i < data.length; ++i) {
    const value = data[i];
    if (value > best) { best = value; index = i; }
  }
  return index;
}
` };
    });
  },
};

await mkdir(join(dist, 'browser'), { recursive: true });
for (const [entryPoint, outfile] of browserEntries) {
  await build({
    absWorkingDir: root,
    entryPoints: [entryPoint],
    outfile: join(dist, 'browser', outfile),
    bundle: true,
    plugins: [greedyArgmax],
    format: 'esm',
    platform: 'browser',
    target: ['es2022'],
    conditions: ['browser', 'import', 'default'],
    mainFields: ['browser', 'module', 'main'],
    // Blob module workers retain the approved package asset base, not their opaque blob URL.
    ...(entryPoint === 'src/runtime/worker.ts' ? {
      define: { 'import.meta.url': '__nekoWorkerModuleUrl', 'self.location': '__nekoWorkerLocation', location: '__nekoWorkerLocation' },
      banner: { js: 'const __nekoWorkerModuleUrl = self.name; const __nekoWorkerLocation = new URL(__nekoWorkerModuleUrl);' },
    } : {}),
    sourcemap: true,
    // Browsers download these bundles; the maps keep stack traces readable. minifySyntax is omitted on purpose: it folds the computed
    // `${'node:'}crypto` imports into literals, which makes consumers' bundlers try to resolve Node built-ins.
    minifyWhitespace: true,
    minifyIdentifiers: true,
    legalComments: 'external',
  });
}

await build({
  absWorkingDir: root,
  entryPoints: nodeEntries,
  outdir: join(dist, 'node'),
  bundle: true,
  plugins: [greedyArgmax],
  splitting: true,
  format: 'esm',
  platform: 'node',
  target: ['node22'],
  external: ['onnxruntime-node', 'sharp', 'parse5', 'pdfjs-dist', '@napi-rs/canvas'],
  sourcemap: true,
  legalComments: 'external',
});

const wasmSource = join(root, 'node_modules/onnxruntime-web/dist');
const wasmDestination = join(dist, 'browser/assets');
await mkdir(wasmDestination, { recursive: true });
for (const file of [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]) {
  const destination = join(wasmDestination, file);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(join(wasmSource, file), destination);
}

// PDF.js is lazy-imported only for extraction. Keep its worker, fonts and image decoders beside it.
const pdfSource = join(root, 'node_modules/pdfjs-dist');
const pdfDestination = join(dist, 'browser/assets/pdf');
await mkdir(pdfDestination, { recursive: true });
for (const file of ['pdf.mjs', 'pdf.worker.mjs']) {
  await copyFile(join(pdfSource, 'legacy/build', file), join(pdfDestination, file));
}
for (const directory of ['cmaps', 'standard_fonts', 'wasm', 'iccs']) {
  await cp(join(pdfSource, directory), join(pdfDestination, directory), { recursive: true });
}
await copyFile(join(pdfSource, 'LICENSE'), join(pdfDestination, 'LICENSE'));
