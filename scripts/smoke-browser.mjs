import { createReadStream } from 'node:fs';
import { mkdtemp, readFile, readdir, rm, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { chromium } from '@playwright/test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';


const device = process.argv[2] ?? 'webgpu';
const maxNewTokens = Number(process.argv[3] ?? '48');
const launchArgs = process.argv.slice(4);
if (!['cpu', 'webgpu'].includes(device)) throw new TypeError('Usage: node scripts/smoke-browser.mjs [cpu|webgpu] [maxNewTokens] [--enable-unsafe-webgpu] [--preflight-only] [--headed] [--offline-reload]');
if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 2048) throw new RangeError('maxNewTokens must be from 1 through 2048');
if (launchArgs.some((argument) => !['--enable-unsafe-webgpu', '--preflight-only', '--headed', '--offline-reload'].includes(argument))) throw new TypeError('Only --enable-unsafe-webgpu, --preflight-only, --headed, and --offline-reload are supported as optional arguments');
if (device === 'cpu') console.warn('CPU/WASM is unsupported by this pinned model; creation rejects with UNSUPPORTED_BACKEND, without fallback.');
const modelCacheDir = process.env.NEKO_MODEL_CACHE;
const modelFiles = new Map();
async function indexModelCache(directory, prefix = '') {
  for (const entry of await readdir(join(directory, prefix), { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) await indexModelCache(directory, name);
    else if (entry.isFile()) modelFiles.set(name, join(directory, name));
  }
}
if (modelCacheDir) await indexModelCache(modelCacheDir);
const cachedModelRequests = [];
const root = new URL('../', import.meta.url);
const files = new Map([
  ['/examples/browser-prototype.html', { url: new URL('examples/browser-prototype.html', root), type: 'text/html' }],
  ['/dist/browser/neko.js', { url: new URL('dist/browser/neko.js', root), type: 'text/javascript' }],
]);

const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    const asset = pathname.match(/^\/dist\/browser\/assets\/(ort-wasm-simd-threaded(?:\.jsep|\.asyncify)?\.(?:mjs|wasm))$/);
    const resource = files.get(pathname) ?? (asset ? {
      url: new URL(`dist/browser/assets/${asset[1]}`, root),
      type: asset[1].endsWith('.wasm') ? 'application/wasm' : 'text/javascript',
    } : undefined);
    const modelName = pathname.startsWith('/model-cache/') ? decodeURIComponent(pathname.slice('/model-cache/'.length)) : undefined;
    const modelFile = modelName ? modelFiles.get(modelName) : undefined;
    if (modelFile) {
      const size = (await stat(modelFile)).size;
      const range = request.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
      const start = range ? Number(range[1]) : 0;
      const end = range && range[2] ? Number(range[2]) : size - 1;
      if (range && (start > end || end >= size)) {
        response.writeHead(416, { 'content-range': `bytes */${size}` }).end();
        return;
      }
      response.writeHead(range ? 206 : 200, {
        'accept-ranges': 'bytes',
        'access-control-allow-origin': '*',
        'content-length': String(end - start + 1),
        ...(range ? { 'content-range': `bytes ${start}-${end}/${size}` } : {}),
      });
      if (request.method === 'HEAD') {
        response.end();
        return;
      }
      cachedModelRequests.push(modelName);
      createReadStream(modelFile, range ? { start, end } : undefined).pipe(response);
      return;
    }
    if (!resource) {
      response.writeHead(404).end('Not found');
      return;
    }
    response.writeHead(200, { 'content-type': resource.type }).end(await readFile(resource.url));
  } catch (error) {
    response.writeHead(500).end(error instanceof Error ? error.message : String(error));
  }
});

server.listen(0, '127.0.0.1');
await once(server, 'listening');
const address = server.address();
let profileDir;
let context;
let browser;
let offlineMode = false;
const offlineNetworkRequests = [];
try {
  profileDir = await mkdtemp(join(tmpdir(), 'neko-browser-smoke-'));
  const chromiumArgs = launchArgs.filter((argument) => argument === '--enable-unsafe-webgpu');
  context = await chromium.launchPersistentContext(profileDir, { args: chromiumArgs, headless: !launchArgs.includes('--headed') });
  browser = context.browser();
  const page = await context.newPage();
  await page.addInitScript(({ port, prefix, useSharedCache }) => {
    globalThis.__cachePutErrors = [];
    globalThis.__webgpuComputeEvidence = { dispatchWorkgroups: 0, dispatchWorkgroupsIndirect: 0, queueSubmit: 0 };
    globalThis.__offlineModelRequests = [];
    const offlineModel = sessionStorage.getItem('neko-browser-smoke-offline') === 'true';
    const nativeCachePut = Cache.prototype.put;
    Cache.prototype.put = function (request, response) {
      return nativeCachePut.call(this, request, response).catch((error) => {
        globalThis.__cachePutErrors.push({ request: String(request), message: error.message, stack: error.stack });
        throw error;
      });
    };
    const computePass = globalThis.GPUComputePassEncoder?.prototype;
    if (computePass?.dispatchWorkgroups) {
      const original = computePass.dispatchWorkgroups;
      computePass.dispatchWorkgroups = function (...args) {
        globalThis.__webgpuComputeEvidence.dispatchWorkgroups++;
        return original.apply(this, args);
      };
    }
    if (computePass?.dispatchWorkgroupsIndirect) {
      const original = computePass.dispatchWorkgroupsIndirect;
      computePass.dispatchWorkgroupsIndirect = function (...args) {
        globalThis.__webgpuComputeEvidence.dispatchWorkgroupsIndirect++;
        return original.apply(this, args);
      };
    }
    const queue = globalThis.GPUQueue?.prototype;
    if (queue?.submit) {
      const original = queue.submit;
      queue.submit = function (...args) {
        globalThis.__webgpuComputeEvidence.queueSubmit++;
        return original.apply(this, args);
      };
    }
    const nativeFetch = globalThis.fetch.bind(globalThis);
    globalThis.fetch = (input, init) => {
      const url = input instanceof Request ? input.url : input instanceof URL ? input.href : input;
      if (typeof url !== 'string' || !url.startsWith(prefix)) return nativeFetch(input, init);
      if (offlineModel) {
        globalThis.__offlineModelRequests.push(url);
        return Promise.reject(new TypeError('Offline smoke blocks model download requests'));
      }
      if (!useSharedCache) return nativeFetch(input, init);
      const name = decodeURIComponent(url.slice(prefix.length));
      const localName = name.split('/').map(encodeURIComponent).join('/');
      const localUrl = `http://127.0.0.1:${port}/model-cache/${localName}`;
      return nativeFetch(input instanceof Request ? new Request(localUrl, input) : localUrl, init);
    };
  }, {
    port: address.port,
    useSharedCache: !!modelCacheDir,
    prefix: 'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT/resolve/fafab72d87a9e6be3925b38caf48286d2838f2d0/',
  });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  if (launchArgs.includes('--offline-reload')) {
    await page.route('https://huggingface.co/**', async (route) => {
      if (offlineMode) {
        offlineNetworkRequests.push(route.request().url());
        await route.abort();
      } else {
        await route.continue();
      }
    });
  }
  await page.goto(`http://127.0.0.1:${address.port}/examples/browser-prototype.html`);

  const adapter = await page.evaluate(async () => {
    const gpu = navigator.gpu;
    const result = gpu ? await gpu.requestAdapter() : null;
    return {
      apiAvailable: !!gpu,
      adapter: result ? {
        info: result.info ?? null,
        shaderF16: result.features.has('shader-f16'),
        fallback: result.info?.isFallbackAdapter ?? null,
      } : null,
      storage: navigator.storage?.estimate ? await navigator.storage.estimate() : null,
    };
  });
  console.log(JSON.stringify({ chromiumVersion: browser?.version() ?? 'unknown', chromiumHeadless: !launchArgs.includes('--headed'), chromiumLaunchArgs: chromiumArgs, sharedModelCache: !!modelCacheDir, requestedDevice: device, adapterPreflight: adapter }));
  if (launchArgs.includes('--preflight-only')) {
    console.log('WebGPU capability probe only; this run performs no model inference.');
    process.exitCode = 2;
  } else {
    const imageBase64 = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const context = canvas.getContext('2d');
      context.fillStyle = '#0000ff';
      context.fillRect(0, 0, 224, 224);
      context.fillStyle = '#ff0000';
      context.fillRect(72, 72, 80, 80);
      return canvas.toDataURL('image/png').split(',')[1];
    });
    const imageBuffer = Buffer.from(imageBase64, 'base64');
    const dataUrl = `data:image/png;base64,${imageBase64}`;
    const cases = [
      { mode: 'image', prompt: 'Name the main geometric shape in this image in one word.', maxNewTokens },
      { mode: 'text', prompt: 'What is 2 + 2? Reply with a single digit.', maxNewTokens: 16 },
      { mode: 'report', prompt: '', maxNewTokens: 256 },
    ];
    const reportHtml = `<html><head><title>Color exhibit</title></head><body><h1>Color exhibit</h1><p>The exhibit has a red square against a blue background.</p><img src="${dataUrl}" alt="red square on blue background"></body></html>`;
    const runInference = async (phase) => {
      for (const testCase of cases) {
        await page.locator('#mode').selectOption(testCase.mode);
        await page.locator('#device').selectOption(device);
        await page.locator('#max-new-tokens').fill(String(testCase.maxNewTokens));
        await page.locator('#prompt').fill(testCase.prompt);
        if (testCase.mode === 'report') {
          await page.locator('#language').selectOption('en');
          await page.locator('#html').fill(reportHtml);
        } else if (testCase.mode === 'image') {
          await page.locator('#image').setInputFiles({ name: 'red-square.png', mimeType: 'image/png', buffer: imageBuffer });
        }
        await page.locator('#run').click();
        await page.waitForFunction(
          () => ['complete', 'error'].includes(document.querySelector('#status')?.dataset.state ?? ''),
          null,
          { timeout: 3_500_000 },
        );
        const observed = await page.evaluate(async () => ({
          mode: document.querySelector('#mode').value,
          language: document.querySelector('#language').value,
          prompt: document.querySelector('#prompt').value,
          html: document.querySelector('#html').value,
          state: document.querySelector('#status').dataset.state,
          status: document.querySelector('#status').textContent,
          output: document.querySelector('#output').textContent,
          details: document.querySelector('#details').textContent,
          cachePutErrors: globalThis.__cachePutErrors,
          offlineModelRequests: globalThis.__offlineModelRequests,
          webgpuComputeEvidence: globalThis.__webgpuComputeEvidence,
          storageAfter: navigator.storage?.estimate ? await navigator.storage.estimate() : null,
        }));
        console.log(JSON.stringify({ fixture: 'synthetic blue canvas with red square', phase, requestedDevice: device, expectedMode: testCase.mode, maxNewTokens: testCase.maxNewTokens, cachedModelFilesServed: [...cachedModelRequests], offlineNetworkRequests, ...observed, pageErrors }, null, 2));
        if (observed.mode !== testCase.mode) throw new Error(`Browser UI did not select ${testCase.mode} mode`);
        if (observed.state === 'error') {
          if (device === 'webgpu' && !/WebGPU API is not present|WebGPU adapter is unavailable|shader-f16/i.test(observed.status)) {
            throw new Error(`Browser UI failed outside the expected WebGPU availability checks: ${observed.status}`);
          }
          console.log(`${phase} ${testCase.mode} inference failed before producing model output: ${observed.status}`);
          process.exitCode = 2;
          return false;
        }
        if (observed.cachePutErrors.length) throw new Error(`Browser model cache writes failed: ${JSON.stringify(observed.cachePutErrors)}`);
        if (device === 'webgpu' && !Object.values(observed.webgpuComputeEvidence).some((count) => count > 0)) {
          throw new Error('Browser inference produced no observable WebGPU compute submission');
        }
        if (testCase.mode === 'report') {
          if (observed.html !== reportHtml) throw new Error('Browser UI changed the report HTML fixture');
          const report = JSON.parse(observed.output);
          if (report.language !== 'en' || report.page.title !== 'Color exhibit') throw new Error('Browser report lost its requested language or source title');
          if (report.images.length !== 1 || report.images[0].status !== 'described' || report.images[0].source.imageId !== report.images[0].imageId) {
            throw new Error('Browser report did not describe the source image with matching provenance');
          }
          const facts = report.sections.flatMap((section) => section.keyPoints).join(' ');
          if (!/red/i.test(facts) || !/blue/i.test(facts) || !/square/i.test(report.images[0].description)) {
            throw new Error('Browser report did not retain the source page facts and describe the visible square');
          }
          const details = JSON.parse(observed.details);
          if (details.backend.device !== device) throw new Error('Browser report used an unexpected selected device');
          continue;
        }
        if (observed.prompt !== testCase.prompt) throw new Error('Browser UI changed the exact caller-provided prompt');
        const result = JSON.parse(observed.details);
        if (result.model.id !== 'onnx-community/Qwen3.5-0.8B-ONNX-OPT' || result.model.revision !== 'fafab72d87a9e6be3925b38caf48286d2838f2d0') {
          throw new Error('Browser UI returned a result for an unexpected model revision');
        }
        if (result.backend.device !== device || result.backend.providerEvidence !== 'loaded-session-configuration') {
          throw new Error(`Browser UI result does not show the explicitly selected ${device} session configuration`);
        }
        if (testCase.mode === 'image') {
          if (!/square/i.test(observed.output)) throw new Error(`Browser image inference did not identify the visible square: ${observed.output}`);
          if (result.finishReason !== 'stop') throw new Error(`Browser image inference was truncated (${result.finishReason}): ${observed.output}`);
        }
        if (testCase.mode === 'text' && !/\b4\b/.test(observed.output)) {
          throw new Error('Browser text-only inference did not answer 2 + 2 correctly');
        }
      }
      return true;
    };
    const firstRunSucceeded = await runInference('initial');
    if (firstRunSucceeded && launchArgs.includes('--offline-reload')) {
      await page.evaluate(() => sessionStorage.setItem('neko-browser-smoke-offline', 'true'));
      offlineMode = true;
      await page.reload();
      await page.locator('#local-only').check();
      await runInference('offline-reload');
      const offlineModelRequests = await page.evaluate(() => globalThis.__offlineModelRequests);
      console.log(JSON.stringify({ offlineModelRequests, blockedHuggingFaceRequests: offlineNetworkRequests }));
      if (offlineModelRequests.length || offlineNetworkRequests.length) {
        throw new Error('Offline reload attempted a Hugging Face model request');
      }
    }
  }
} finally {
  await context?.close();
  if (profileDir) await rm(profileDir, { recursive: true, force: true });
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}
