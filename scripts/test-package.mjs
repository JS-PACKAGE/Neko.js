import { execFile } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

const run = promisify(execFile);
const root = resolve(new URL('../', import.meta.url).pathname);
const temporary = await mkdtemp(join(tmpdir(), 'neko-package-consumer-'));
try {
  const packed = await run('npm', ['pack', '--json', '--pack-destination', temporary], { cwd: root, maxBuffer: 4 * 1024 * 1024 });
  const [{ filename, files }] = JSON.parse(packed.stdout);
  const listed = new Set(files.map(({ path }) => path));

  const consumer = join(temporary, 'consumer');
  await mkdir(consumer);
  await writeFile(join(consumer, 'package.json'), JSON.stringify({
    name: 'neko-package-consumer',
    private: true,
    type: 'module',
    allowScripts: { 'onnxruntime-node@1.30.0': true, 'sharp@0.35.4': true },
  }, null, 2));
  await run('npm', ['install', '--no-audit', '--no-fund', join(temporary, filename)], {
    cwd: consumer,
    maxBuffer: 16 * 1024 * 1024,
  });

  const packageRoot = join(consumer, 'node_modules', 'neko.js');
  const pkg = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8'));
  for (const target of Object.values(pkg.exports).flatMap((entry) => Object.values(entry))) {
    if (typeof target === 'string' && target.startsWith('./') && !listed.has(target.slice(2))) {
      throw new Error(`Packed exports map points to missing artifact ${target}`);
    }
  }
  for (const file of ['dist/node/index.js', 'dist/node/web.js', 'dist/node/report.js', 'dist/node/backend.js', 'dist/browser/neko.js', 'dist/browser/types.js']) {
    if (!listed.has(file)) throw new Error(`Packed artifact is missing ${file}`);
  }
  if (pkg.dependencies?.['onnxruntime-node'] !== '1.30.0') throw new Error('Packed consumer contract does not pin native ONNX Runtime 1.30.0');
  const typeFixture = join(consumer, 'consumer.ts');
  await writeFile(typeFixture, `
    import { createNeko } from 'neko.js';
    type Instance = Awaited<ReturnType<typeof createNeko>>;
    const input: Parameters<Instance['infer']>[0] = { prompt: 'Describe the image.', image: new URL('file:///tmp/example.png') };
    async function usePackage(instance: Instance) {
      await instance.infer(input);
      await instance.describe('<html><title>Exhibit</title><p>Colored shapes.</p></html>', { language: 'en', format: 'json' });
      instance.cache.model.status();
      instance.cache.engine.status();
    }
    void usePackage;
  `);
  await run(join(root, 'node_modules/.bin/tsc'), [
    '--noEmit', '--strict', '--skipLibCheck', 'false', '--module', 'NodeNext',
    '--moduleResolution', 'NodeNext', '--target', 'ES2022', '--typeRoots', join(root, 'node_modules/@types'),
    typeFixture,
  ], { cwd: consumer, maxBuffer: 16 * 1024 * 1024 });
  const checks = `
    import * as neko from 'neko.js';
    import * as types from 'neko.js/types';
    import * as web from 'neko.js/web';
    import * as report from 'neko.js/report';
    import * as backend from 'neko.js/backend';
    import * as ort from 'onnxruntime-node';
    if (typeof neko.createNeko !== 'function' || typeof neko.Neko !== 'function' || typeof neko.NekoError !== 'function') throw new Error('Main SDK exports are unavailable');
    if (!types || !web || !report || !backend || !ort.InferenceSession) throw new Error('A packed public entry could not be imported');
    const versions = ort.env.versions;
    if (versions?.common !== '1.30.0') throw new Error('Consumer resolved an unexpected native ONNX Runtime: ' + JSON.stringify(versions));
    console.log(JSON.stringify({ exports: ['neko.js', 'neko.js/types', 'neko.js/web', 'neko.js/report', 'neko.js/backend'], onnxruntime: versions.common }));
  `;
  await run(process.execPath, ['--input-type=module', '-e', checks], { cwd: consumer });
  if (!process.argv.includes('--artifact-only')) {
    const modelSmoke = `
      import assert from 'node:assert/strict';
      import sharp from 'sharp';
      import path from 'node:path';
      import { createNeko } from 'neko.js';
      const svg = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="#0000ff"/><rect x="56" y="56" width="112" height="112" fill="#ff0000"/></svg>');
      const imageBuffer = await sharp(svg).png().toBuffer();
      const imagePath = path.join(process.cwd(), 'quality-fixture.png');
      await sharp(imageBuffer).toFile(imagePath);
      const neko = await createNeko({ device: 'cpu', ...(process.env.NEKO_MODEL_CACHE ? { cacheDir: process.env.NEKO_MODEL_CACHE } : {}) });
      try {
        let textStream = '';
        const answer = await neko.infer({
          prompt: 'What is 2 + 2? Answer using a single digit.',
          maxNewTokens: 16,
          onToken: (chunk) => { textStream += chunk; },
        });
        assert.match(answer.text, /\\b4\\b/, 'the packed consumer must answer a basic text-only question');
        assert.equal(textStream, answer.text, 'the packed consumer streams the decoded answer');
        assert.ok(answer.usage.inputTokens > 0 && answer.usage.outputTokens > 0, 'the packed consumer reports generated token usage');
        assert.equal(answer.finishReason, 'stop', 'the packed consumer completes the single-token answer without truncation');

        const visual = await neko.infer({
          image: imagePath,
          prompt: 'Name the main geometric shape in this image in one word.',
          maxNewTokens: 16,
        });
        assert.match(visual.text.toLowerCase(), /square/, 'the packed consumer identifies a visible square');
        assert.equal(visual.finishReason, 'stop', 'the packed consumer completes image inference without truncation');

        const dataUrl = 'data:image/png;base64,' + imageBuffer.toString('base64');
        const report = await neko.describe(
          '<html><head><title>Color exhibit</title></head><body><h1>Color exhibit</h1><p>The exhibit shows a red square against a blue background.</p><img src="' + dataUrl + '" alt="red square on blue background"></body></html>',
          { language: 'en', format: 'json', maxNewTokens: 256, imageFailurePolicy: 'error' },
        );
        assert.equal(report.page.title, 'Color exhibit');
        assert.equal(report.images.length, 1, 'the report preserves the embedded image');
        assert.equal(report.images[0].status, 'described', 'the report actually describes the embedded image');
        assert.equal(report.images[0].source.imageId, report.images[0].imageId, 'the report preserves image provenance');
        assert.ok(report.sections.length > 0 && report.sections.some((section) => /red/i.test(section.keyPoints.join(' ')) && /blue/i.test(section.keyPoints.join(' '))), 'the report captures the page facts');
        assert.match(report.images[0].description.toLowerCase(), /square/);
        const { runtime, device, executionProviders, sessions } = visual.backend;
        console.log(JSON.stringify({ text: answer.text, image: visual.text, report: { title: report.page.title, imageDescription: report.images[0].description, facts: report.sections.map((section) => section.keyPoints) }, backend: { runtime, device, executionProviders, sessions }, usage: answer.usage }));
      } finally {
        await neko.dispose();
      }
    `;
    const { stdout } = await run(process.execPath, ['--input-type=module', '-e', modelSmoke], {
      cwd: consumer,
      env: { ...process.env },
      maxBuffer: 16 * 1024 * 1024,
    });
    console.log(stdout.trim());
  }
  console.log(`Packed consumer ${process.argv.includes('--artifact-only') ? 'artifact check' : 'model inference'} passed (${filename}).`);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
