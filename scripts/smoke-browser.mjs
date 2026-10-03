import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { chromium } from '@playwright/test';
import { env as transformersEnv } from '@huggingface/transformers';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { MODEL_BASE_URL, MODEL_ID, MODEL_REVISION, modelFileUrl } from '../dist/src/cache/manifest.js';
import { getRegisteredModelProfile } from '../dist/src/cache/registry.js';
import { createModelMirror, MODEL_MIRROR_PATH } from './serve-model-mirror.mjs';


const device = process.argv[2] ?? 'webgpu';
const maxNewTokens = Number(process.argv[3] ?? '48');
const allOptions = process.argv.slice(4);
const profileOptions = allOptions.filter((argument) => argument.startsWith('--model-profile='));
if (profileOptions.length > 1) throw new TypeError('Provide --model-profile only once');
const modelProfile = profileOptions[0]?.slice('--model-profile='.length) ?? 'default';
const launchArgs = allOptions.filter((argument) => !argument.startsWith('--model-profile='));
if (device !== 'webgpu') throw new TypeError('Usage: node scripts/smoke-browser.mjs [webgpu] [maxNewTokens] [--model-profile=default|all-q4] [--require-hardware] [--enable-unsafe-webgpu] [--preflight-only] [--headed] [--offline-reload]');
if (!['default', 'all-q4'].includes(modelProfile)) throw new TypeError('--model-profile must be default or all-q4');
if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 2048) throw new RangeError('maxNewTokens must be from 1 through 2048');
if (launchArgs.some((argument) => !['--require-hardware', '--enable-unsafe-webgpu', '--preflight-only', '--headed', '--offline-reload'].includes(argument))) throw new TypeError('Only --model-profile=default|all-q4, --require-hardware, --enable-unsafe-webgpu, --preflight-only, --headed, and --offline-reload are supported as optional arguments');
const requireHardware = launchArgs.includes('--require-hardware');
const modelCacheDir = process.env.NEKO_MODEL_CACHE;
const modelCacheRoot = modelCacheDir ? resolve(modelCacheDir, '../../..') : undefined;
if (modelCacheDir && resolve(modelCacheRoot, MODEL_ID, MODEL_REVISION) !== resolve(modelCacheDir)) {
  throw new TypeError('NEKO_MODEL_CACHE must be the pinned revision directory containing tokenizer.json and onnx/');
}
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
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
let modelMirror;
let profileDir;
let context;
let browser;
const offlineNetworkRequests = [];
const untrustedExternalRequests = [];
const modelHosts = new Set(['huggingface.co', 'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co', 'cas-bridge.xethub.hf.co']);
let offlineMode = false;
try {
  profileDir = await mkdtemp(join(tmpdir(), 'neko-browser-smoke-'));
  const chromiumArgs = launchArgs.filter((argument) => argument === '--enable-unsafe-webgpu');
  context = await chromium.launchPersistentContext(profileDir, { channel: 'chromium', args: chromiumArgs, headless: !launchArgs.includes('--headed') });
  browser = context.browser();
  await context.route('**/*', async (route) => {
    const requested = new URL(route.request().url());
    const local = requested.hostname === '127.0.0.1' || requested.hostname === 'localhost';
    if (!modelHosts.has(requested.hostname)) {
      if (local) return route.continue();
      untrustedExternalRequests.push(requested.href);
      return route.abort();
    }
    if (offlineMode) {
      offlineNetworkRequests.push(requested.href);
      return route.abort();
    }
    return route.continue();
  });
  const page = await context.newPage();
  await page.addInitScript((settings) => {
    if (settings.requireHardware && navigator.gpu) {
      const nativeRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
      navigator.gpu.requestAdapter = async (options) => {
        const adapter = await nativeRequestAdapter(options);
        if (!adapter || adapter.info?.isFallbackAdapter !== false) throw new Error('Hardware WebGPU verification requires a non-fallback adapter');
        return adapter;
      };
    }
    globalThis.__cachePutErrors = [];
    globalThis.__webgpuComputeEvidence = { dispatchWorkgroups: 0, dispatchWorkgroupsIndirect: 0, queueSubmit: 0 };
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
  }, { requireHardware });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto(`http://127.0.0.1:${address.port}/examples/browser-prototype.html`);

  const adapter = await page.evaluate(async () => {
    const gpu = navigator.gpu;
    const result = gpu ? await gpu.requestAdapter() : null;
    return {
      apiAvailable: !!gpu,
      adapter: result ? {
        info: {
          vendor: result.info?.vendor ?? '',
          architecture: result.info?.architecture ?? '',
          device: result.info?.device ?? '',
          description: result.info?.description ?? '',
        },
        shaderF16: result.features.has('shader-f16'),
        fallback: result.info?.isFallbackAdapter ?? null,
      } : null,
      storage: navigator.storage?.estimate ? await navigator.storage.estimate() : null,
    };
  });
  console.log(JSON.stringify({ chromiumVersion: browser?.version() ?? 'unknown', chromiumHeadless: !launchArgs.includes('--headed'), chromiumLaunchArgs: chromiumArgs, sharedModelCache: !!modelCacheDir, requestedDevice: device, requestedProfile: modelProfile, adapterPreflight: adapter }));
  if (!launchArgs.includes('--preflight-only') && (!adapter.adapter || !adapter.adapter.shaderF16)) {
    throw new Error('Actual WebGPU inference requires an available shader-f16 adapter; no CPU/WASM fallback is supported');
  }
  const requestsBeforeConsentChecks = untrustedExternalRequests.length;
  await page.locator('#mode').selectOption('report');
  await page.locator('#html').fill('https://example.invalid/private-page');
  await page.locator('#run').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.startsWith('Review the trust warning'));
  await page.locator('#approve-remote-page').check();
  await page.locator('#local-only').check();
  await page.locator('#run').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Local-only mode cannot fetch a remote page URL.');
  await page.locator('#local-only').uncheck();
  if (untrustedExternalRequests.length !== requestsBeforeConsentChecks) throw new Error('A remote page URL triggered network access without approval.');
  console.log('Remote report URL was blocked without explicit approval and remained blocked in local-only mode.');
  if (launchArgs.includes('--preflight-only')) {
    console.log('WebGPU capability probe only; this run performs no model inference.');
    process.exitCode = 2;
  } else {
    let cacheSeed;
    if (modelCacheRoot) {
      modelMirror = await createModelMirror({ cacheDir: modelCacheRoot, profile: modelProfile, allowedOrigin: `http://127.0.0.1:${address.port}` });
      modelMirror.listen(0, '127.0.0.1');
      await once(modelMirror, 'listening');
      const modelMirrorOrigin = `http://127.0.0.1:${modelMirror.address().port}`;
      const mirrorBaseUrl = new URL(MODEL_MIRROR_PATH, `${modelMirrorOrigin}/`).href;
      const modelFiles = Object.entries(getRegisteredModelProfile(modelProfile).files).map(([name, spec]) => ({
        name,
        size: spec.size,
        sha256: spec.sha256,
        sourceUrl: new URL(name, mirrorBaseUrl).href,
        cacheUrl: modelFileUrl(name),
      }));
      const runtimePath = '/dist/browser/assets/ort-wasm-simd-threaded.asyncify.wasm';
      const runtimeBytes = await readFile(new URL(runtimePath.slice(1), root));
      const seedPayload = {
        cacheName: transformersEnv.cacheKey,
        profile: modelProfile,
        origin: `http://127.0.0.1:${address.port}`,
        mirrorOrigin: modelMirrorOrigin,
        modelBaseUrl: MODEL_BASE_URL,
        files: modelFiles,
        runtimeAsset: {
          url: new URL(runtimePath, `http://127.0.0.1:${address.port}`).href,
          size: runtimeBytes.byteLength,
          sha256: sha256(runtimeBytes),
        },
      };
      cacheSeed = await page.evaluate(async (payload) => {
        await import('/dist/browser/neko.js');
        if (typeof globalThis.caches?.open !== 'function') throw new Error('CacheStorage is unavailable at the browser smoke origin');
        const cache = await globalThis.caches.open(payload.cacheName);
        const digestHex = async (bytes) => {
          const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
          return Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
        };
        const verifiedFiles = [];
        for (const file of payload.files) {
          if (new URL(file.sourceUrl).origin !== payload.mirrorOrigin || file.cacheUrl !== `${payload.modelBaseUrl}${file.name}`) {
            throw new Error(`Unexpected model cache seed URL for ${file.name}`);
          }
          const response = await fetch(file.sourceUrl, { cache: 'no-store' });
          if (!response.ok || response.headers.get('content-length') !== String(file.size)) {
            throw new Error(`Local model mirror response mismatch for ${file.name}`);
          }
          const bytes = await response.arrayBuffer();
          const digest = await digestHex(bytes);
          if (bytes.byteLength !== file.size || digest !== file.sha256) throw new Error(`Local model mirror integrity mismatch for ${file.name}`);
          await cache.put(file.cacheUrl, new Response(bytes));
          verifiedFiles.push({ name: file.name, size: bytes.byteLength, sha256: digest });
        }
        const runtimeAsset = payload.runtimeAsset;
        if (new URL(runtimeAsset.url).origin !== payload.origin) throw new Error('Runtime cache seed must use the browser smoke origin');
        const runtimeResponse = await fetch(runtimeAsset.url, { cache: 'no-store' });
        if (!runtimeResponse.ok) throw new Error(`Browser smoke server returned HTTP ${runtimeResponse.status} for the runtime binary`);
        const runtimeBytes = await runtimeResponse.arrayBuffer();
        const runtimeSha256 = await digestHex(runtimeBytes);
        if (runtimeBytes.byteLength !== runtimeAsset.size || runtimeSha256 !== runtimeAsset.sha256) throw new Error('Runtime binary cache seed integrity mismatch');
        await cache.put(runtimeAsset.url, new Response(runtimeBytes));
        return {
          source: 'verified-node-cache-local-mirror',
          cacheName: payload.cacheName,
          profile: payload.profile,
          files: verifiedFiles,
          totalModelBytes: verifiedFiles.reduce((total, file) => total + file.size, 0),
          runtimeAsset: { url: runtimeAsset.url, size: runtimeBytes.byteLength, sha256: runtimeSha256 },
        };
      }, seedPayload);
      console.log(JSON.stringify({ modelCacheSeed: cacheSeed }, null, 2));
    }
    if (cacheSeed) offlineMode = true;
    await page.locator('#local-only').check();
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
        await page.locator('#model-profile').selectOption(modelProfile);
        await page.locator('#max-new-tokens').fill(String(testCase.maxNewTokens));
        await page.locator('#prompt').fill(testCase.prompt);
        if (testCase.mode === 'report') {
          await page.locator('#language').selectOption('en');
          await page.locator('#html').fill(reportHtml);
        } else if (testCase.mode === 'image') {
          await page.locator('#image').setInputFiles({ name: 'red-square.png', mimeType: 'image/png', buffer: imageBuffer });
        }
        const computeBefore = await page.evaluate(() => ({ ...globalThis.__webgpuComputeEvidence }));
        await page.locator('#run').click();
        await page.waitForFunction(
          () => ['complete', 'error'].includes(document.querySelector('#status')?.dataset.state ?? ''),
          null,
          { timeout: 3_500_000 },
        );
        const observed = await page.evaluate(async () => ({
          mode: document.querySelector('#mode').value,
          modelProfile: document.querySelector('#model-profile').value,
          language: document.querySelector('#language').value,
          prompt: document.querySelector('#prompt').value,
          html: document.querySelector('#html').value,
          state: document.querySelector('#status').dataset.state,
          status: document.querySelector('#status').textContent,
          output: document.querySelector('#output').textContent,
          details: document.querySelector('#details').textContent,
          cachePutErrors: globalThis.__cachePutErrors,
          webgpuComputeEvidence: globalThis.__webgpuComputeEvidence,
          storageAfter: navigator.storage?.estimate ? await navigator.storage.estimate() : null,
        }));
        console.log(JSON.stringify({ fixture: testCase.mode, phase, requestedDevice: device, requestedProfile: modelProfile, maxNewTokens: testCase.maxNewTokens, state: observed.state, webgpuComputeEvidence: observed.webgpuComputeEvidence, offlineNetworkRequests: offlineNetworkRequests.length, cachePutErrors: observed.cachePutErrors.length, pageErrors }, null, 2));
        if (observed.mode !== testCase.mode || observed.modelProfile !== modelProfile) throw new Error(`Browser UI did not select ${testCase.mode} mode and ${modelProfile} profile`);
        if (observed.state === 'error') {
          if (device === 'webgpu' && !/WebGPU API is not present|WebGPU adapter is unavailable|shader-f16/i.test(observed.status)) {
            throw new Error(`Browser UI failed outside the expected WebGPU availability checks: ${observed.status}`);
          }
          console.log(`${phase} ${testCase.mode} inference failed before producing model output: ${observed.status}`);
          process.exitCode = 2;
          return false;
        }
        if (observed.cachePutErrors.length) throw new Error(`Browser model cache writes failed: ${JSON.stringify(observed.cachePutErrors)}`);
        const dispatches = observed.webgpuComputeEvidence.dispatchWorkgroups + observed.webgpuComputeEvidence.dispatchWorkgroupsIndirect;
        const priorDispatches = computeBefore.dispatchWorkgroups + computeBefore.dispatchWorkgroupsIndirect;
        if (dispatches <= priorDispatches || observed.webgpuComputeEvidence.queueSubmit <= computeBefore.queueSubmit) {
          throw new Error('This fixture produced no observable WebGPU compute dispatch and queue submission');
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
          if (report.metadata.model.profile !== modelProfile || report.metadata.backend.device !== device || details.backend.device !== device) {
            throw new Error('Browser report did not preserve the selected profile and device');
          }
          continue;
        }
        if (observed.prompt !== testCase.prompt) throw new Error('Browser UI changed the exact caller-provided prompt');
        const result = JSON.parse(observed.details);
        if (result.model.profile !== modelProfile) throw new Error(`Browser UI returned profile ${result.model.profile}, expected ${modelProfile}`);
        if (result.backend.device !== device) throw new Error(`Browser UI result does not show the explicitly selected ${device} device`);
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
      offlineMode = true;
      await page.reload();
      await page.locator('#local-only').check();
      await runInference('offline-reload');
      console.log(JSON.stringify({ blockedModelHostRequests: offlineNetworkRequests, blockedUntrustedExternalRequests: untrustedExternalRequests }));
      if (offlineNetworkRequests.length || untrustedExternalRequests.length) throw new Error('Offline reload attempted a model-host or untrusted external request');
    }
  }
} finally {
  await context?.close();
  if (modelMirror) await new Promise((resolve, reject) => modelMirror.close((error) => error ? reject(error) : resolve()));
  if (profileDir) await rm(profileDir, { recursive: true, force: true });
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}
