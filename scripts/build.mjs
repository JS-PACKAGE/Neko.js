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

const replaceOnce = (source, target, replacement, path) => {
  if (source.split(target).length !== 2) throw new Error(`Transformers.js patch target not found exactly once in ${path}: ${target.split('\n')[0]}`);
  return source.replace(target, () => replacement);
};
// Build-time patches to the pinned inference dependencies; a changed upstream target fails the build.
// - Its greedy sampler takes the argmax with its shared `max()`, whose call sites see arrays of many element types, so V8 runs it as a
//   slow polymorphic loop: about 0.8 ms per token over the 248,320-entry vocabulary, against about 0.1 ms for a loop that only ever sees
//   Float32Array. Eight ordered comparisons per iteration reduce loop overhead while preserving NaN handling and first-index ties.
// - Its BPE model accepts a prebuilt vocabulary Map and merge-rank Map, so Node can parse tokenizer.json on a worker thread and build
//   the same Maps from compact strings (`src/core/text-resources.ts`). Plain JSON configurations take the original path.
// - Its BPE memo cache moved every hit to the most-recent end of its 10,000-entry Map (delete + set). That churn made V8 rehash the
//   Map over and over and took about half of report planning's time on long pages; hits are now plain lookups, so the oldest
//   inserted entry is evicted first. The cache only memoizes a pure function, so encodings are unchanged.
// - Text-only RoPE deltas subtract the padded prompt length: adding it instead shifts every decode position by twice that length.
// - Metadata probes must recognize native cache path hits too. Ignoring them probes an unrelated local path, which the verified
//   fetch correctly rejects. A path hit has unknown metadata size; the verified cache still checks bytes before returning the path.
// - The browser native bridge reuses immutable group-zero layouts per pipeline. Weak keys do not retain pipelines, and no buffer
//   bindings or model data are cached. Patch the embedded ORT bundle, not the separately copied runtime asset or WebGPU prototypes.
// - Empty native extension chains skip callback tables and captured state. Keep that state in a nonempty-branch-only function:
//   capturing a converter variable allocates a context even when dispatch is skipped. Refresh heap views before probing; nonempty
//   chains retain upstream dispatch, including its rejection of unsupported extensions.
const transformersPatches = {
  name: 'neko-transformers-patches',
  setup(build) {
    build.onLoad({ filter: /[\\/]@huggingface[\\/]transformers[\\/]dist[\\/]transformers\.(node\.mjs|web\.js)$/ }, async ({ path }) => {
      let source = await readFile(path, 'utf8');
      source = replaceOnce(source, 'const argmax = max(logits.data)[1];', 'const argmax = nekoGreedyArgmax(logits.data);', path);
      source = replaceOnce(source, '    super(config);\n    this.tokens_to_ids = object_to_map(config.vocab);\n',
        '    super(config);\n    this.tokens_to_ids = config.vocab instanceof Map ? config.vocab : object_to_map(config.vocab);\n', path);
      source = replaceOnce(source, '    this.bpe_ranks = new Map(this.merges.map((x, i) => [JSON.stringify(x), i]));\n',
        '    this.bpe_ranks = config.bpe_ranks instanceof Map ? config.bpe_ranks : new Map(this.merges.map((x, i) => [JSON.stringify(x), i]));\n', path);
      source = replaceOnce(source, '    const cached = this.cache.get(token);\n', '    const cached = this.cache.cache.get(token);\n', path);
      source = replaceOnce(source, 'max(data.subarray(dims[1] * i, dims[1] * (i + 1)))[0] + 1n + BigInt(dims[1])',
        'max(data.subarray(dims[1] * i, dims[1] * (i + 1)))[0] + 1n - BigInt(dims[1])', path);
      source = replaceOnce(source, '  if (cachedResponse !== void 0 && typeof cachedResponse !== "string") {',
        '  if (typeof cachedResponse === "string") return { exists: true, fromCache: true };\n  if (cachedResponse !== void 0) {', path);
      return { loader: 'js', contents: `${source}
function nekoGreedyArgmax(data) {
  const length = data.length;
  if (length === 0) throw Error("Array must not be empty");
  let best = data[0];
  let index = 0;
  let i = 1;
  const end = length - 7;
  for (; i < end; i += 8) {
    let value = data[i];
    if (value > best) { best = value; index = i; }
    value = data[i + 1];
    if (value > best) { best = value; index = i + 1; }
    value = data[i + 2];
    if (value > best) { best = value; index = i + 2; }
    value = data[i + 3];
    if (value > best) { best = value; index = i + 3; }
    value = data[i + 4];
    if (value > best) { best = value; index = i + 4; }
    value = data[i + 5];
    if (value > best) { best = value; index = i + 5; }
    value = data[i + 6];
    if (value > best) { best = value; index = i + 6; }
    value = data[i + 7];
    if (value > best) { best = value; index = i + 7; }
  }
  for (; i < length; ++i) {
    const value = data[i];
    if (value > best) { best = value; index = i; }
  }
  return index;
}
` };
    });
    if (build.initialOptions.platform === 'browser') {
      let runtimePatched = false;
      if (build.initialOptions.entryPoints.some((entry) => entry === 'src/index.ts' || entry === 'src/runtime/worker.ts')) {
        build.onEnd((result) => {
          if (!result.errors.length && !runtimePatched) throw new Error('Pinned WebGPU runtime layout patch was not applied');
        });
      }
      build.onLoad({ filter: /[\\/]onnxruntime-web[\\/]dist[\\/]ort\.webgpu\.bundle\.min\.mjs$/ }, async ({ path }) => {
        let source = replaceOnce(await readFile(path, 'utf8'), 't=n.getBindGroupLayout(t)', 't=nekoGetGroupLayout(n,t)', path);
        source = replaceOnce(source, 'fu(u,{327681:I=>{E=(p(),A)[I+8>>>2>>>0]}})',
          'E=(p(),A)[u>>>2>>>0]?((entry)=>{let external=0;fu(entry,{327681:I=>{external=(p(),A)[I+8>>>2>>>0]}});return external})(u):0', path);
        // Keep the stateless converter in its runtime factory, without a new closure per bind group.
        source = replaceOnce(source, 'function qu(e,t){function n(u){', 'function nekoNativeBindGroupEntry(u){', path);
        source = replaceOnce(source, 'resource:c}}e>>>=0,t={label:Ne(', 'resource:c}}function qu(e,t){e>>>=0,t={label:Ne(', path);
        source = replaceOnce(source, 'h.push(n(c+40*b))', 'h.push(nekoNativeBindGroupEntry(c+40*b))', path);
        // Pin the decoder's malformed-UTF-8 behavior; only batch its ASCII string construction.
        const nativeUtf8Decoder = 'xn=(e,t=0,n,o)=>{if(16<(n=In(e,t>>>=0,n,o))-t&&e.buffer&&An)return An.decode(e.buffer instanceof ArrayBuffer?e.subarray(t,n):e.slice(t,n));for(o="";t<n;){var u=e[t++];if(128&u){var c=63&e[t++];if((224&u)==192)o+=String.fromCharCode((31&u)<<6|c);else{var h=63&e[t++];65536>(u=(240&u)==224?(15&u)<<12|c<<6|h:(7&u)<<18|c<<12|h<<6|63&e[t++])?o+=String.fromCharCode(u):(u-=65536,o+=String.fromCharCode(55296|u>>10,56320|1023&u))}}else o+=String.fromCharCode(u)}return o}';
        source = replaceOnce(source, nativeUtf8Decoder, replaceOnce(nativeUtf8Decoder,
          'else o+=String.fromCharCode(u)',
          'else if(t+1<n&&!((c=e[t])&128)&&!((h=e[t+1])&128)){var b;if(t+2<n&&!((b=e[t+2])&128)){o+=String.fromCharCode(u,c,h,b);t+=3}else{o+=String.fromCharCode(u,c,h);t+=2}}else o+=String.fromCharCode(u)', path), path);
        // Pointer keys need no array length. Preserve unsigned handles and explicit resource deletion.
        for (const [before, after] of [
          ['de=[]', 'de=new Map()'],
          ['uu=(e,t)=>{de[e>>>0]=t}', 'uu=(e,t)=>{de.set(e>>>0,t)}'],
          ['L=e=>{if(e)return de[e>>>0]}', 'L=e=>{if(e)return de.get(e>>>0)}'],
          ['function Eu(e){delete de[e>>>0]}', 'function Eu(e){de.delete(e>>>0)}'],
          ['de[u>>>0]=B.queue,de[o>>>0]=B,G', 'de.set(u>>>0,B.queue),de.set(o>>>0,B),G'],
          ['de[n>>>0]=u,o&&(Re[n]=[])', 'de.set(n>>>0,u),o&&(Re[n]=[])'],
          ['de[o>>>0]=u,Pr(', 'de.set(o>>>0,u),Pr('],
          ['de[n>>>0]=e}', 'de.set(n>>>0,e)}'],
          ['de[o>>>0]=c,Et(', 'de.set(o>>>0,c),Et('],
          ['n=t.beginComputePass(n),de[e>>>0]=n,e', 'n=t.beginComputePass(n),de.set(e>>>0,n),e'],
          ['t=t.finish(),de[e>>>0]=t,e', 't=t.finish(),de.set(e>>>0,t),e'],
          ['t=nekoGetGroupLayout(n,t),de[e>>>0]=t,e', 't=nekoGetGroupLayout(n,t),de.set(e>>>0,t),e'],
          ['n=t.createCommandEncoder(n),de[e>>>0]=n,e', 'n=t.createCommandEncoder(n),de.set(e>>>0,n),e'],
          ['t=n.createQuerySet(t),de[e>>>0]=t,e', 't=n.createQuerySet(t),de.set(e>>>0,t),e'],
          ['de[te>>>0]=F.queue,de[j>>>0]=F,j', 'de.set(te>>>0,F.queue),de.set(j>>>0,F),j'],
          ['j=Eo(j),de[j>>>0]=F,j', 'j=Eo(j),de.set(j>>>0,F),j'],
        ]) source = replaceOnce(source, before, after, path);
        if (source.includes('de[')) throw new Error('Pinned WebGPU runtime registry migration is incomplete');
        runtimePatched = true;
        return { loader: 'js', contents: `${source}
const nekoBindGroupLayouts = new WeakMap();
function nekoGetGroupLayout(pipeline, index) {
  if (index !== 0) return pipeline.getBindGroupLayout(index);
  let layout = nekoBindGroupLayouts.get(pipeline);
  if (layout === undefined) { layout = pipeline.getBindGroupLayout(index); nekoBindGroupLayouts.set(pipeline, layout); }
  return layout;
}
` };
      });
    }
  },
};

await mkdir(join(dist, 'browser'), { recursive: true });
for (const [entryPoint, outfile] of browserEntries) {
  await build({
    absWorkingDir: root,
    entryPoints: [entryPoint],
    outfile: join(dist, 'browser', outfile),
    bundle: true,
    plugins: [transformersPatches],
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
  plugins: [transformersPatches],
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
