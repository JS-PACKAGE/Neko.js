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
  for (const file of ['dist/node/index.js', 'dist/node/worker.js', 'dist/node/web.js', 'dist/node/report.js', 'dist/node/backend.js', 'dist/browser/neko.js', 'dist/browser/worker.js', 'dist/browser/types.js']) {
    if (!listed.has(file)) throw new Error(`Packed artifact is missing ${file}`);
  }
  if (typeof pkg.dependencies?.['onnxruntime-node'] !== 'string') throw new Error('Packed consumer is missing its native ONNX Runtime dependency');
  const typeFixture = join(consumer, 'consumer.ts');
  await writeFile(typeFixture, `
    import { createNeko, type InferOptions, type InferencePlanOptions, type ModelSource, type Neko, type NekoOptions, type Page, type ReportCheckpoint, type StructuredInferOptions } from 'neko.js';
    import * as web from 'neko.js/web';
    import { parseStructuredReport, serializeStructuredReport, parseReportCheckpoint, serializeReportCheckpoint } from 'neko.js/report';
    const policy: NonNullable<NekoOptions['policy']> = {
      network(url, kind) { return kind === 'model' || kind === 'runtime' || kind === 'worker' || (kind === 'image' && url.protocol === 'https:'); },
      localFiles(path) { return path.endsWith('/approved.png'); },
    };
    const modelSource: ModelSource = { baseUrl: 'http://127.0.0.1:8788/models/' };
    const options: NekoOptions = { device: 'cpu', modelProfile: 'all-q4', modelSource, execution: 'worker', policy, queue: { maxPending: 8 } };
    async function usePackage(instance: Neko, page: Page) {
      const history: InferOptions = {
        messages: [
          { role: 'system', content: 'Answer briefly.' },
          { role: 'user', content: [{ type: 'text', text: 'Describe these images.' }, { type: 'image', image: new URL('file:///approved.png') }, { type: 'image', image: new URL('file:///second.png') }] },
        ],
        maxNewTokens: 32,
      };
      const images: InferOptions = { images: [new URL('file:///approved.png'), new URL('file:///second.png')], prompt: 'Compare these images.', maxNewTokens: 32 };
      await instance.infer(history);
      await instance.infer(images);
      const planning: InferencePlanOptions = { ...images, schema: { type: 'object' } };
      const plan = await instance.planInference(planning);
      plan.inputTokens; plan.availableOutputTokens; plan.fits;
      const structured: StructuredInferOptions = {
        prompt: 'Return an object with an integer answer.',
        schema: { type: 'object', properties: { answer: { type: 'integer' } }, required: ['answer'], additionalProperties: false },
        maxNewTokens: 32,
      };
      const result = await instance.inferStructured(structured);
      result.structured.mode;
      result.value;
      let checkpoint: ReportCheckpoint | undefined;
      const report = await instance.describe(page, {
        format: 'json',
        sources: { paragraphIds: [page.paragraphs[0].id], paragraph: (source) => source.text.length > 0, image: (source) => source.url.length > 0 },
        budget: { maxTotalTokens: 1000, maxDurationMs: 10000 },
        onEvent(event) { event.type; },
        onCheckpoint(value) { checkpoint = value; },
      });
      report.metadata.model.profile;
      report.schemaVersion; report.sourceFacts; report.metadata.coverage;
      await parseStructuredReport(await serializeStructuredReport(report), page);
      if (checkpoint) await instance.describe(page, { format: 'json', resume: await parseReportCheckpoint(await serializeReportCheckpoint(checkpoint)) });
      await instance.describe('<p>Text input.</p>', { format: 'markdown' });
      await instance.load();
      await instance.warmup();
      await instance.runtimeStatus();
      await instance.queueStatus();
      await instance.backend.current();
      await instance.cache.model.status();
      await instance.cache.engine.status();
    }
    async function construct(page: Page) {
      const instance = await createNeko(options);
      await usePackage(instance, page);
    }
    void construct;
    void web.extractPage;
  `);
  await run(join(root, 'node_modules/.bin/tsc'), [
    '--noEmit', '--strict', '--skipLibCheck', 'false', '--module', 'NodeNext',
    '--moduleResolution', 'NodeNext', '--target', 'ES2022', '--lib', 'DOM,ESNext',
    typeFixture,
  ], { cwd: consumer, maxBuffer: 16 * 1024 * 1024 });
  const checks = `
    import * as neko from 'neko.js';
    import * as types from 'neko.js/types';
    import * as web from 'neko.js/web';
    import * as report from 'neko.js/report';
    import * as backend from 'neko.js/backend';
    import * as ort from 'onnxruntime-node';
    if (typeof neko.createNeko !== 'function' || typeof neko.Neko?.create !== 'function' || typeof neko.NekoError !== 'function') throw new Error('Main SDK exports are unavailable');
    if (!types || !web || !report || !backend || typeof ort.InferenceSession?.create !== 'function') throw new Error('A packed public entry could not be imported');
    const versions = ort.env.versions;
    if (typeof versions?.common !== 'string') throw new Error('Consumer ONNX Runtime metadata is unavailable: ' + JSON.stringify(versions));
    const instance = await neko.createNeko({ device: 'cpu', localFilesOnly: true });
    try {
      for (const operation of [
        () => instance.infer({ prompt: '   ' }),
        () => instance.infer({ prompt: 'Hello', maxNewTokens: 0 }),
        () => instance.planInference({ prompt: 'Hello', contextWindowTokens: 0 }),
        () => instance.describe('<p>Hello</p>', { language: 'not_a_language' }),
      ]) {
        let error;
        try { await operation(); } catch (cause) { error = cause; }
        if (!(error instanceof neko.NekoError) || error.stage !== 'preprocess') throw new Error('Invalid requests must reject before model loading: ' + String(error));
      }
      if ((await instance.cache.engine.status()).loaded) throw new Error('Preflight validation loaded the model');
    } finally { await instance.dispose(); }
    console.log(JSON.stringify({ exports: ['neko.js', 'neko.js/types', 'neko.js/web', 'neko.js/report', 'neko.js/backend'], onnxruntime: versions.common }));
  `;
  await run(process.execPath, ['--input-type=module', '-e', checks], { cwd: consumer });
  if (!process.argv.includes('--artifact-only')) {
    const modelSmoke = `
      import assert from 'node:assert/strict';
      import sharp from 'sharp';
      import path from 'node:path';
      import { realpath } from 'node:fs/promises';
      import { createNeko } from 'neko.js';
      import * as web from 'neko.js/web';
      import { parseStructuredReport, serializeStructuredReport, parseReportCheckpoint, serializeReportCheckpoint } from 'neko.js/report';
      const svg = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="#0000ff"/><rect x="56" y="56" width="112" height="112" fill="#ff0000"/></svg>');
      const imageBuffer = await sharp(svg).png().toBuffer();
      const imagePath = path.join(process.cwd(), 'quality-fixture.png');
      await sharp(imageBuffer).toFile(imagePath);
      const canonicalImagePath = await realpath(imagePath);
      const dataUrl = 'data:image/png;base64,' + imageBuffer.toString('base64');
      const policy = {
        network(_url, kind) { return kind === 'model' || kind === 'worker'; },
        localFiles(filePath) { return filePath === canonicalImagePath; },
      };
      const neko = await createNeko({ device: 'cpu', modelProfile: 'default', execution: 'worker', localFilesOnly: true, policy, ...(process.env.NEKO_MODEL_CACHE ? { cacheDir: process.env.NEKO_MODEL_CACHE } : {}) });
      try {
        const textPlan = await neko.planInference({ prompt: 'What is 2 + 2? Answer using a single digit.', maxNewTokens: 16 });
        assert.equal(textPlan.fits, true);
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
        assert.equal(textPlan.inputTokens, answer.usage.inputTokens, 'planned tokens must match actual chat preprocessing');
        assert.equal(answer.model.profile, 'default');
        assert.equal(answer.execution?.mode, 'worker', 'the packed consumer runs inference in a worker');

        const visualPlan = await neko.planInference({ image: imagePath, prompt: 'Name the main geometric shape in this image in one word.', maxNewTokens: 16 });
        const visual = await neko.infer({
          image: imagePath,
          prompt: 'Name the main geometric shape in this image in one word.',
          maxNewTokens: 16,
        });
        assert.match(visual.text.toLowerCase(), /square/, 'the packed consumer identifies a visible square');
        assert.equal(visual.model.profile, 'default');
        assert.equal(visual.finishReason, 'stop', 'the packed consumer completes image inference without truncation');
        assert.equal(visualPlan.inputTokens, visual.usage.inputTokens, 'image planning must include expanded visual tokens');
        assert.deepEqual(visualPlan.images, visual.images, 'planning preserves the same decoded image identity');

        const pair = await neko.infer({ images: [imagePath, dataUrl], prompt: 'Describe both images briefly.', maxNewTokens: 32 });
        assert.equal(pair.images?.length, 2, 'the packed consumer preserves both image observations');

        const overflow = await neko.planInference({ prompt: 'What is 2 + 2?', contextWindowTokens: 32, maxNewTokens: 32 });
        assert.equal(overflow.fits, false, 'planning reports context overflow without generating');
        const structured = await neko.inferStructured({
          prompt: 'Return only JSON: the answer to 2 + 2 is the integer 4.',
          schema: { type: 'object', properties: { answer: { type: 'integer' } }, required: ['answer'], additionalProperties: false },
          maxNewTokens: 32,
        });
        assert.deepEqual(structured.value, { answer: 4 });

        const page = await web.extractPage(
          '<html><head><title>Color exhibit</title></head><body><h1>Color exhibit</h1><p>The exhibit shows a red square against a blue background.</p><img src="' + dataUrl + '" alt="red square on blue background"></body></html>',
        );
        let checkpoint;
        const report = await neko.describe(page, {
          language: 'en',
          format: 'json',
          maxNewTokens: 256,
          imageFailurePolicy: 'error',
          onCheckpoint(value) { checkpoint = value; },
        });
        assert.equal(report.page.title, 'Color exhibit');
        assert.equal(report.images.length, 1, 'the report preserves the embedded image');
        assert.equal(report.images[0].status, 'described', 'the report actually describes the embedded image');
        assert.equal(report.images[0].source.imageId, report.images[0].imageId, 'the report preserves image provenance');
        assert.ok(report.sections.length > 0 && report.sections.some((section) => /red/i.test(section.keyPoints.join(' ')) && /blue/i.test(section.keyPoints.join(' '))), 'the report captures the page facts');
        assert.match(report.images[0].description.toLowerCase(), /square/);
        assert.equal(report.metadata.model.profile, 'default');
        assert.equal(report.metadata.execution.mode, 'worker');
        const restored = await parseStructuredReport(await serializeStructuredReport(report), page);
        assert.deepEqual(restored, report, 'the packed report survives validated JSON persistence');
        assert.equal(restored.sourceFacts.map((fact) => fact.citation.quote).join(''), page.paragraphs.map((paragraph) => paragraph.text).join(''), 'the report retains the exact selected source text');
        assert.equal(restored.metadata.coverage.semanticRetention, 'not-measured');
        const corrupted = JSON.parse(await serializeStructuredReport(report));
        corrupted.sourceFacts[0].citation.quote += ' invented';
        await assert.rejects(() => parseStructuredReport(JSON.stringify(corrupted)), 'tampered retained evidence must fail validation');
        assert.ok(checkpoint, 'the packed consumer receives report checkpoints');
        const restoredCheckpoint = await parseReportCheckpoint(await serializeReportCheckpoint(checkpoint));
        const resumed = await neko.describe(page, { language: 'en', format: 'json', resume: restoredCheckpoint });
        assert.ok(resumed.metadata.resumedStages > 0, 'the packed consumer reuses checkpointed report stages');
        const [modelCache, engineCache, backend] = await Promise.all([
          neko.cache.model.status(),
          neko.cache.engine.status(),
          neko.backend.current(),
        ]);
        console.log(JSON.stringify({ text: answer.text, image: visual.text, report: { title: report.page.title, imageDescription: report.images[0].description, facts: report.sections.map((section) => section.keyPoints) }, backend: { ...backend, execution: answer.execution }, cache: { model: modelCache, engine: engineCache }, usage: answer.usage }));
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
