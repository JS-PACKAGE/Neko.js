import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { crc32, createDeflateRaw } from 'node:zlib';
import { createNeko } from '../dist/node/index.js';
import { getRegisteredModelProfile } from '../dist/src/cache/registry.js';

const arguments_ = process.argv.slice(2);
if (arguments_.length > 1 || (arguments_[0] && !arguments_[0].startsWith('--cache-dir='))) {
  throw new TypeError('Usage: node scripts/build-extension.mjs [--cache-dir=<SDK model cache root>]');
}
const cacheDir = arguments_[0]?.slice('--cache-dir='.length);
const root = new URL('../', import.meta.url);
const source = new URL('examples/chrome-extension/', root);
const output = new URL('dist/extension/', root);
const files = ['manifest.json', 'background.js', 'content.js', 'offscreen.html', 'offscreen.js', 'icons/icon-16.png', 'icons/icon-32.png', 'icons/icon-48.png', 'icons/icon-128.png'];
// Inline execution is used, so the SDK worker and PDF assets are not needed.
const runtime = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.asyncify.mjs'];

await rm(output, { recursive: true, force: true });
await mkdir(new URL('neko/assets/', output), { recursive: true });
await mkdir(new URL('icons/', output), { recursive: true });
for (const name of files) await cp(new URL(name, source), new URL(name, output));
await cp(new URL('dist/browser/neko.js', root), new URL('neko/neko.js', output));
for (const name of runtime) await cp(new URL(`dist/browser/assets/${name}`, root), new URL(`neko/assets/${name}`, output));

// The pinned default-profile files go into the package under model/. Node can follow Hugging Face's redirects, so a missing
// cache is downloaded here with the SDK's size/SHA-256 checks; each copy is hashed again against the pinned manifest.
const neko = await createNeko({ device: 'cpu', modelProfile: 'default', ...(cacheDir ? { cacheDir } : {}) });
let model;
try {
  await neko.cache.model.prefetch();
  const diagnostics = await neko.cache.model.diagnostics();
  model = getRegisteredModelProfile(diagnostics.profile, diagnostics.model);
  const installed = join(diagnostics.path, ...model.id.split('/'), model.revision);
  for (const [name, spec] of Object.entries(model.files)) {
    const destination = new URL(`model/${name}`, output);
    await mkdir(new URL('./', destination), { recursive: true });
    const hash = createHash('sha256');
    await pipeline(createReadStream(join(installed, ...name.split('/'))), async function* (chunks) {
      for await (const chunk of chunks) { hash.update(chunk); yield chunk; }
    }, createWriteStream(destination));
    const { size } = await stat(destination);
    if (size !== spec.size || hash.digest('hex') !== spec.sha256) throw new Error(`Pinned model file failed size/SHA-256 verification while packaging: ${name}`);
  }
} finally {
  await neko.dispose();
}

// Redistributed code and weights carry their licenses. Bundled packages are read from the SDK bundle's source map.
const apache = await readFile(new URL('node_modules/@huggingface/transformers/LICENSE', root), 'utf8');
const mit = (holder) => `MIT License\n\nCopyright (c) ${holder}\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n`;
// Packages published without a license file; holders from ONNX Runtime's bundled legal comment and the package author field.
const mitHolders = { 'onnxruntime-web': 'Microsoft Corporation', 'onnxruntime-common': 'Microsoft Corporation', '@cfworker/json-schema': 'Jeremy Danyow' };
const { sources } = JSON.parse(await readFile(new URL('dist/browser/neko.js.map', root), 'utf8'));
const packages = new Set();
for (const path of sources) {
  const match = path.match(/node_modules\/((?:@[^/]+\/)?[^/]+)\//);
  // ONNX Runtime Web's own source map names its embedded onnxruntime-common sources "common".
  if (match) packages.add(match[1] === 'common' ? 'onnxruntime-common' : match[1]);
}
const notices = [
  'Third-party notices for the Neko.js page reader extension.',
  '',
  `Model: ${model.id} (revision ${model.revision}), https://huggingface.co/${model.id}`,
  `License: ${model.license}. The files under model/ are unmodified copies of the pinned files; the full license text follows the package list.`,
];
for (const name of [...packages].sort()) {
  const directory = new URL(`node_modules/${name}/`, root);
  const { version, license } = JSON.parse(await readFile(new URL('package.json', directory), 'utf8'));
  const licenseFile = (await readdir(directory)).find((file) => /^licen[cs]e(\.(md|txt))?$/i.test(file));
  let text;
  if (licenseFile) text = await readFile(new URL(licenseFile, directory), 'utf8');
  else if (license === 'MIT' && Object.hasOwn(mitHolders, name)) text = mit(mitHolders[name]);
  else throw new Error(`Bundled package ${name} has no license file; review it and add its notice`);
  notices.push('', '-'.repeat(78), `${name} ${version} (${license})`, '', text.trim());
}
notices.push('', '-'.repeat(78), `${model.id} model weights (${model.license})`, '', apache.trim(), '');
await writeFile(new URL('THIRD_PARTY_NOTICES.txt', output), notices.join('\n'));

// Store upload: manifest.json at the zip root. Entries are streamed with data descriptors and a fixed timestamp.
const { version } = JSON.parse(await readFile(new URL('manifest.json', output), 'utf8'));
const archive = fileURLToPath(new URL(`dist/neko-extension-${version}.zip`, root));
const directory = fileURLToPath(output);
const entries = (await readdir(directory, { recursive: true, withFileTypes: true })).filter((entry) => entry.isFile())
  .map((entry) => relative(directory, join(entry.parentPath, entry.name)).split(sep).join('/')).sort();
const zip = createWriteStream(archive);
let offset = 0;
const write = (bytes) => new Promise((resolve, reject) => { offset += bytes.length; zip.write(bytes, (error) => (error ? reject(error) : resolve())); });
const central = [];
for (const entry of entries) {
  const name = Buffer.from(entry, 'utf8');
  const start = offset;
  const local = Buffer.alloc(30);
  local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x0808, 6); local.writeUInt16LE(8, 8);
  local.writeUInt16LE(0, 10); local.writeUInt16LE(0x21, 12); local.writeUInt16LE(name.length, 26);
  await write(Buffer.concat([local, name]));
  let checksum = 0; let size = 0; let compressed = 0;
  await pipeline(createReadStream(join(directory, entry)), async function* (chunks) {
    for await (const chunk of chunks) { checksum = crc32(chunk, checksum); size += chunk.length; yield chunk; }
  }, createDeflateRaw(), async (chunks) => { for await (const chunk of chunks) { compressed += chunk.length; await write(chunk); } });
  const descriptor = Buffer.alloc(16);
  descriptor.writeUInt32LE(0x08074b50, 0); descriptor.writeUInt32LE(checksum, 4); descriptor.writeUInt32LE(compressed, 8); descriptor.writeUInt32LE(size, 12);
  await write(descriptor);
  const header = Buffer.alloc(46);
  header.writeUInt32LE(0x02014b50, 0); header.writeUInt16LE(20, 4); header.writeUInt16LE(20, 6); header.writeUInt16LE(0x0808, 8); header.writeUInt16LE(8, 10);
  header.writeUInt16LE(0, 12); header.writeUInt16LE(0x21, 14); header.writeUInt32LE(checksum, 16); header.writeUInt32LE(compressed, 20); header.writeUInt32LE(size, 24);
  header.writeUInt16LE(name.length, 28); header.writeUInt32LE(start, 42);
  central.push(header, name);
}
const centralStart = offset;
for (const bytes of central) await write(bytes);
// Without ZIP64 every offset and size must fit in 32 bits; the store rejects packages over 2 GB anyway.
if (offset > 0xffffffff) throw new Error('Extension package exceeds the 4 GiB ZIP limit');
const end = Buffer.alloc(22);
end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
end.writeUInt32LE(offset - centralStart, 12); end.writeUInt32LE(centralStart, 16);
await write(end);
await new Promise((resolve, reject) => { zip.end((error) => (error ? reject(error) : resolve())); });

console.log(`Built ${directory} (load it unpacked from chrome://extensions) and ${archive} (${(offset / 2 ** 20).toFixed(0)} MiB, Chrome Web Store upload).`);
