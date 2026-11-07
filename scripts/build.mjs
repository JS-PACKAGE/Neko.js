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
// - Generation keeps mutable last-position scores separate from raw decoder outputs, but reuses its own Float32 working tensor
//   after each completed sampling step. Other dtypes/layouts retain upstream slicing and conversion; there is no cross-generation pool.
// - Its BPE model accepts a prebuilt vocabulary Map and merge-rank Map, so Node can parse tokenizer.json on a worker thread and build
//   the same Maps from compact strings (`src/core/text-resources.ts`). Plain JSON configurations take the original path.
// - Its BPE memo cache moved every hit to the most-recent end of its 10,000-entry Map (delete + set). That churn made V8 rehash the
//   Map over and over and took about half of report planning's time on long pages; hits are now plain lookups, so the oldest
//   inserted entry is evicted first. The cache only memoizes a pure function, so encodings are unchanged.
// - Text-only RoPE deltas subtract the padded prompt length: adding it instead shifts every decode position by twice that length.
// - Metadata probes must recognize native cache path hits too. Ignoring them probes an unrelated local path, which the verified
//   fetch correctly rejects. A path hit has unknown metadata size; the verified cache still checks bytes before returning the path.
// - The browser native bridge reuses immutable group-zero layouts per pipeline (weak keys do not retain pipelines) and bind groups
//   for repeated (layout, buffer-range) bindings: ORT's WebGPU provider created about 1.08 million per text report, one per kernel
//   dispatch, mostly identical. No model data is cached. Patch the embedded ORT bundle, not the separately copied runtime asset or
//   WebGPU prototypes.
// - Per-dispatch native queries churned the JS heap: uniform uploads sliced a view, handle release deleted Map entries (tombstones,
//   rehashing) and buffer size/usage queries boxed BigInts. Each is replaced by an allocation-free equivalent.
// - Empty native extension chains skip callback tables and captured state. Keep that state in a nonempty-branch-only function:
//   capturing a converter variable allocates a context even when dispatch is skipped. Refresh heap views before probing; nonempty
//   chains retain upstream dispatch, including its rejection of unsupported extensions.
const transformersPatches = {
  name: 'neko-transformers-patches',
  setup(build) {
    build.onLoad({ filter: /[\\/]@huggingface[\\/]transformers[\\/]dist[\\/]transformers\.(node\.mjs|web\.js)$/ }, async ({ path }) => {
      let source = await readFile(path, 'utf8');
      source = replaceOnce(source, 'const argmax = max(logits.data)[1];', 'const argmax = nekoGreedyArgmax(logits.data);', path);
      source = replaceOnce(source, '    let return_dict_items = {};', '    let return_dict_items = {};\n    let nekoWorkingScores;', path);
      source = replaceOnce(source, '      const logits = outputs.logits.slice(null, -1, null).to("float32");',
        '      const logits = nekoWorkingScores = nekoCopyLastLogits(outputs.logits, nekoWorkingScores);', path);
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
function nekoCopyLastLogits(source, previous) {
  const dims = source.dims;
  if (source.type !== "float32" || dims.length !== 3 || dims[1] !== 1) return source.slice(null, -1, null).to("float32");
  const data = source.data;
  if (!(data instanceof Float32Array) || data.length === 0 || data.length !== dims[0] * dims[2]) return source.slice(null, -1, null).to("float32");
  const previousData = previous?.data;
  if (!(previousData instanceof Float32Array) || previous.dims.length !== 2 || previous.dims[0] !== dims[0] || previous.dims[1] !== dims[2]
    || previousData.length !== data.length || previousData.buffer === data.buffer) {
    previous = new Tensor2("float32", new Float32Array(data.length), [dims[0], dims[2]]);
  }
  previous.data.set(data);
  return previous;
}
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
        // Cache only the existing view's buffer identity; every check still reads live Memory.buffer.
        // Update with the private view before publishing HEAP8, preserving constructor failure and setter reentry.
        source = replaceOnce(source, 'function p(){Fe.buffer!=X.buffer&&se()}',
          'var nekoHeapViewBuffer;function p(){Fe.buffer!=(nekoHeapViewBuffer??X.buffer)&&se()}', path);
        source = replaceOnce(source, 'r.HEAP8=X=new Int8Array(e),',
          'r.HEAP8=(X=new Int8Array(e),nekoHeapViewBuffer=e,X),', path);
        source = replaceOnce(source, 'fu(u,{327681:I=>{E=(p(),A)[I+8>>>2>>>0]}})',
          'E=(p(),A)[u>>>2>>>0]?((entry)=>{let external=0;fu(entry,{327681:I=>{external=(p(),A)[I+8>>>2>>>0]}});return external})(u):0', path);
        // Keep the stateless converters in their runtime factory, without new closures per bind group.
        source = replaceOnce(source, 'function qu(e,t){function n(u){', 'function nekoNativeBindGroupEntry(u){', path);
        source = replaceOnce(source, 'resource:c}}e>>>=0,t={label:Ne(', 'resource:c}}function qu(e,t){e>>>=0,t={label:Ne(', path);
        source = replaceOnce(source, 'h.push(n(c+40*b))', 'h.push(nekoNativeBindGroupEntry(c+40*b))', path);
        source = replaceOnce(source, 'entries:function(u,c){for(var h=[],b=0;b<u;++b)h.push(nekoNativeBindGroupEntry(c+40*b));return h}',
          'entries:nekoNativeBindGroupEntries', path);
        // Size fresh entry arrays from the heap count; undefined counts still produce an empty list.
        source = replaceOnce(source, 'function qu(e,t){',
          'function nekoNativeBindGroupEntries(u,c){for(var h=new Array(u>>>0),b=0;b<u;++b)h[b]=nekoNativeBindGroupEntry(c+40*b);return h}function qu(e,t){', path);
        // Reuse bind groups for repeated (layout, buffer-range) bindings. Entries are hashed into a numeric key and verified exactly
        // against the stored descriptor, so a collision is only a miss. Identities come from per-object ids, so a released handle
        // reassigned to a new buffer or layout never aliases a cached group; chains, non-buffer resources and oversized entry lists
        // take the upstream path. Two bounded generations avoid per-hit Map churn; destroyed buffers simply age out with their groups.
        source = replaceOnce(source,
          'function qu(e,t){e>>>=0,t={label:Ne(4+(t>>>=0)),layout:L((p(),A)[t+12>>>2>>>0]),entries:nekoNativeBindGroupEntries((p(),A)[t+16>>>2>>>0],(p(),A)[t+20>>>2>>>0])},e=L(e);var o=yo(0);return uu(o,e.createBindGroup(t)),o}',
          'function nekoDescribeBindGroup(t){return{label:Ne(4+t),layout:L((p(),A)[t+12>>>2>>>0]),entries:nekoNativeBindGroupEntries((p(),A)[t+16>>>2>>>0],(p(),A)[t+20>>>2>>>0])}}'
          + 'var nekoBindGroupFresh=new Map,nekoBindGroupStale=new Map,nekoResourceIds=new WeakMap,nekoNextResourceId=1,nekoBindGroupScratch=new Int32Array(97);'
          + 'function nekoResourceId(o){var i=nekoResourceIds.get(o);if(i===void 0){i=nekoNextResourceId++;nekoResourceIds.set(o,i)}return i}'
          + 'function nekoSameDescriptor(d,s,n){if(d.length!==n)return!1;for(var i=0;i<n;++i)if(d[i]!==s[i])return!1;return!0}'
          + 'function nekoCachedBindGroup(e,t){var h=(p(),A),layout=L(h[t+12>>>2>>>0]),count=h[t+16>>>2>>>0]>>>0,c=h[t+20>>>2>>>0],s=nekoBindGroupScratch;if(!layout||count>16)return e.createBindGroup(nekoDescribeBindGroup(t));'
          + 'var k1=nekoResourceId(layout),k2=k1^count,n=1;s[0]=k1;'
          + 'for(var b=0;b<count;++b){var u=c+40*b,buffer=h[u>>>2>>>0]?void 0:L(h[u+8>>>2>>>0]);if(!buffer)return e.createBindGroup(nekoDescribeBindGroup(t));'
          + 'var binding=h[u+4>>>2>>>0],id=nekoResourceId(buffer),ol=h[u+16>>>2>>>0]|0,oh=h[u+20>>>2>>>0]|0,sl=h[u+24>>>2>>>0]|0,sh=h[u+28>>>2>>>0]|0;s[n++]=binding;s[n++]=id;s[n++]=ol;s[n++]=oh;s[n++]=sl;s[n++]=sh;'
          + 'k1=Math.imul(k1^binding,2654435761)+id|0;k1=Math.imul(k1^ol,2246822519)^sl;k2=Math.imul(k2+id,16777619)^oh;k2=Math.imul(k2^sh,3266489917)+binding|0}'
          + 'var key=(k1^Math.imul(k2,2654435761))&1073741823,cached=nekoBindGroupFresh.get(key);'
          + 'if(cached===void 0){cached=nekoBindGroupStale.get(key);if(cached!==void 0){nekoBindGroupStale.delete(key);nekoBindGroupFresh.set(key,cached)}}'
          + 'if(cached!==void 0&&nekoSameDescriptor(cached.descriptor,s,n))return cached.group;'
          + 'var group=e.createBindGroup(nekoDescribeBindGroup(t));if(nekoBindGroupFresh.size>=16384){nekoBindGroupStale=nekoBindGroupFresh;nekoBindGroupFresh=new Map}nekoBindGroupFresh.set(key,{group:group,descriptor:s.slice(0,n)});return group}'
          + 'function qu(e,t){e>>>=0,t>>>=0;var o=yo(0);return uu(o,nekoCachedBindGroup(L(e),t)),o}', path);
        // Pin the decoder's malformed-UTF-8 behavior; only batch its ASCII string construction.
        const nativeUtf8Decoder = 'xn=(e,t=0,n,o)=>{if(16<(n=In(e,t>>>=0,n,o))-t&&e.buffer&&An)return An.decode(e.buffer instanceof ArrayBuffer?e.subarray(t,n):e.slice(t,n));for(o="";t<n;){var u=e[t++];if(128&u){var c=63&e[t++];if((224&u)==192)o+=String.fromCharCode((31&u)<<6|c);else{var h=63&e[t++];65536>(u=(240&u)==224?(15&u)<<12|c<<6|h:(7&u)<<18|c<<12|h<<6|63&e[t++])?o+=String.fromCharCode(u):(u-=65536,o+=String.fromCharCode(55296|u>>10,56320|1023&u))}}else o+=String.fromCharCode(u)}return o}';
        source = replaceOnce(source, nativeUtf8Decoder, replaceOnce(nativeUtf8Decoder,
          'else o+=String.fromCharCode(u)',
          'else if(t+1<n&&!((c=e[t])&128)&&!((h=e[t+1])&128)){var b;if(t+2<n&&!((b=e[t+2])&128)){o+=String.fromCharCode(u,c,h,b);t+=3}else{o+=String.fromCharCode(u,c,h);t+=2}}else o+=String.fromCharCode(u)', path), path);
        // Upload uniform data straight from the heap view with an element offset instead of a per-call subarray.
        source = replaceOnce(source, 'o=(p(),J).subarray(o>>>0,o+u>>>0),e.writeBuffer(t,n,o,0,u)}', 'e.writeBuffer(t,n,(p(),J),o,u)}', path);
        // Buffer size/usage are immutable, but every native query boxed a fresh BigInt. Remember both per buffer object.
        source = replaceOnce(source, 'function Pu(e){return BigInt(L(e>>>0).size)}function _u(e){return BigInt(L(e>>>0).usage)}',
          'var nekoBufferFacts=new WeakMap;function nekoBufferFact(e){var b=L(e>>>0),f=nekoBufferFacts.get(b);if(f===void 0){f={size:BigInt(b.size),usage:BigInt(b.usage)};nekoBufferFacts.set(b,f)}return f}'
          + 'function Pu(e){return nekoBufferFact(e).size}function _u(e){return nekoBufferFact(e).usage}', path);
        // Pointer keys need no array length. Preserve unsigned handles. Releasing clears the slot instead of deleting it: handles are
        // reused malloc addresses, so the registry stays bounded by peak liveness without per-release tombstones and rehashing.
        for (const [before, after] of [
          ['de=[]', 'de=new Map()'],
          ['uu=(e,t)=>{de[e>>>0]=t}', 'uu=(e,t)=>{de.set(e>>>0,t)}'],
          ['L=e=>{if(e)return de[e>>>0]}', 'L=e=>{if(e)return de.get(e>>>0)}'],
          ['function Eu(e){delete de[e>>>0]}', 'function Eu(e){de.set(e>>>0,void 0)}'],
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
