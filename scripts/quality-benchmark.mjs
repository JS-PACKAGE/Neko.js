import { loadQualityContract, ARTIFACT_VERSION, FIXTURE_VERSION } from './quality/contract.mjs';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import sharp from 'sharp';
import { env as transformersEnv } from '@huggingface/transformers';
import { MODEL_BASE_URL, modelFileUrl } from '../dist/src/cache/manifest.js';
import { getRegisteredModelProfile } from '../dist/src/cache/registry.js';
import { createModelMirror, MODEL_MIRROR_PATH } from './serve-model-mirror.mjs';
import { evaluateClaims, EVALUATOR_VERSION } from './quality/evaluate.mjs';
import { evaluateHierarchy } from './quality/hierarchy.mjs';
import { POLICY_VERSION } from './quality/policy.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const maxProgressSummaries = 64;

function usage() {
  return `Neko.js quality benchmark (offline unless --allow-network is explicit)\n\nUsage: npm run quality:benchmark -- [options]\n\n  --case all|text|image|boundaries|hierarchy (default: all)\n  --runtime node|browser (default: node)\n  --device cpu|webgpu (Node default: cpu; browser: webgpu)\n  --model-profile default|all-q4\n  --context-window-tokens N\n  --max-new-tokens N\n  --profile-prefix PATH (Node only)\n  --cache-dir PATH\n  --browser-profile PATH (dedicated Chromium directory)\n  --browser-port N (default: 4173)\n  --allow-network (explicit model download permission)\n  --headed\n  --output PATH\n  --help\n\nCompletion is NOT quality acceptance. Run quality:gate against the raw artifact.`;
}

const { values } = parseArgs({ options: {
  case: { type: 'string', default: 'all' },
  runtime: { type: 'string', default: 'node' },
  device: { type: 'string' },
  'model-profile': { type: 'string', default: 'default' },
  'context-window-tokens': { type: 'string' },
  'max-new-tokens': { type: 'string' },
  'profile-prefix': { type: 'string' },
  'cache-dir': { type: 'string' },
  'browser-profile': { type: 'string' },
  'browser-port': { type: 'string', default: '4173' },
  'allow-network': { type: 'boolean', default: false },
  headed: { type: 'boolean', default: false },
  output: { type: 'string' },
  help: { type: 'boolean', default: false },
} });
if (values.help) {
  console.log(usage());
  process.exit(0);
}

const selectedCase = values.case;
const runtime = values.runtime;
const device = values.device ?? (runtime === 'browser' ? 'webgpu' : 'cpu');
const modelProfile = values['model-profile'];
const browserPort = Number(values['browser-port']);
const contextOverride = values['context-window-tokens'] === undefined ? undefined : Number(values['context-window-tokens']);
const tokenOverride = values['max-new-tokens'] === undefined ? undefined : Number(values['max-new-tokens']);
if (!['all', 'text', 'image', 'boundaries', 'hierarchy'].includes(selectedCase)) throw new TypeError('--case must be all, text, image, boundaries, or hierarchy');
if (!['node', 'browser'].includes(runtime)) throw new TypeError('--runtime must be node or browser');
if (!['cpu', 'webgpu'].includes(device)) throw new TypeError('--device must be cpu or webgpu');
if (runtime === 'browser' && device !== 'webgpu') throw new TypeError('Browser runtime requires --device webgpu; CPU/WASM is unsupported for this model');
if (!['default', 'all-q4'].includes(modelProfile)) throw new TypeError('--model-profile must be default or all-q4');
if (!Number.isSafeInteger(browserPort) || browserPort < 1 || browserPort > 65535) throw new RangeError('--browser-port must be an integer from 1 through 65535');
if (contextOverride !== undefined && (!Number.isSafeInteger(contextOverride) || contextOverride < 32)) throw new RangeError('--context-window-tokens must be an integer of at least 32');
if (tokenOverride !== undefined && (!Number.isSafeInteger(tokenOverride) || tokenOverride < 1 || tokenOverride > 2048)) throw new RangeError('--max-new-tokens must be from 1 through 2048');
if (runtime === 'browser' && values['profile-prefix']) throw new TypeError('--profile-prefix currently requires --runtime node');
if (runtime !== 'browser' && values['browser-profile']) throw new TypeError('--browser-profile requires --runtime browser');
if (runtime === 'browser' && !values['allow-network'] && !values['cache-dir']) throw new TypeError('Offline browser runs require --cache-dir to seed the verified Node cache');

const contract = await loadQualityContract({ selectedCase, maxNewTokens: tokenOverride, contextWindowTokens: contextOverride });
const { manifest, hierarchyRecords, hierarchyHtml, imageRaster, imageDataUrl, textPrompt, cases, oracleSha256, stimulusKey } = contract;
const fixtureEntries = contract.fixtures;
const rasterMetadata = await sharp(imageRaster).metadata();
const fixtureMismatch = contract.fixtureMismatch
  || rasterMetadata.format !== manifest.image.raster.format
  || rasterMetadata.width !== manifest.image.raster.width
  || rasterMetadata.height !== manifest.image.raster.height;

function errorInfo(error) {
  return { name: error instanceof Error ? error.name : 'Error', message: error instanceof Error ? error.message : String(error), ...(typeof error?.code === 'string' ? { code: error.code } : {}), ...(typeof error?.stage === 'string' ? { stage: error.stage } : {}) };
}
function addProgressSummary(summaries, event) {
  if (!event || typeof event !== 'object') return;
  const phase = typeof event.phase === 'string' ? event.phase : 'unknown';
  const file = typeof event.file === 'string' ? event.file : 'unknown';
  const key = JSON.stringify([phase, file]);
  const previous = summaries.get(key);
  if (!previous && summaries.size >= maxProgressSummaries) return;
  const summary = { phase, file, eventCount: (previous?.eventCount ?? 0) + 1 };
  const loaded = typeof event.loaded === 'number' && Number.isFinite(event.loaded) ? event.loaded : undefined;
  const total = typeof event.total === 'number' && Number.isFinite(event.total) ? event.total : undefined;
  if (loaded !== undefined) summary.loaded = Math.max(previous?.loaded ?? 0, loaded);
  else if (previous?.loaded !== undefined) summary.loaded = previous.loaded;
  if (total !== undefined) summary.total = total;
  else if (previous?.total !== undefined) summary.total = previous.total;
  summaries.set(key, summary);
}
function runtimeStatusEvidence(model) {
  if (typeof model?.runtimeStatus !== 'function') return {};
  return model.runtimeStatus().then(
    (runtimeStatus) => ({ runtimeStatus }),
    (error) => ({ runtimeStatusError: errorInfo(error) }),
  );
}
function cacheEvidence(status) {
  return status ? { downloaded: status.downloaded, verified: status.verified, bytes: status.bytes, totalBytes: status.totalBytes, fileCount: status.files?.length ?? null, presentFiles: status.files?.filter((file) => file.present).length ?? null } : null;
}
function makeResult(testCase, status, fields = {}) {
  return { id: testCase.id, fixtureId: testCase.fixtureId, kind: testCase.kind, status, attempted: false, ...fields };
}
function asText(result) {
  return typeof result?.text === 'string' ? result.text : undefined;
}
function runProfile(result, caseId) {
  return caseId === 'hierarchy' ? result?.metadata?.model?.profile ?? null : result?.model?.profile ?? null;
}
function backendValue(result, caseId) {
  return caseId === 'hierarchy' ? result?.metadata?.backend ?? null : result?.backend ?? null;
}

function scoreHierarchy(report) {
  return evaluateHierarchy(report, hierarchyRecords);
}

function score(testCase, result) {
  if (testCase.id !== 'hierarchy') {
    const output = asText(result);
    return output === undefined ? null : evaluateClaims(output, manifest[testCase.id]);
  }
  return scoreHierarchy(result);
}

function sourceIdsFromPage(page) {
  return Array.isArray(page?.paragraphs) ? page.paragraphs.map(({ id }) => id) : [];
}

async function runNode() {
  const { createNeko, MODEL_ID, MODEL_REVISION, extractPage } = await import('../dist/node/index.js');
  const progress = new Map();
  const options = {
    device,
    localFilesOnly: !values['allow-network'],
    progressCallback: (event) => addProgressSummary(progress, event),
    ...(values['cache-dir'] ? { cacheDir: resolve(values['cache-dir']) } : {}),
    ...(values['profile-prefix'] ? { profilePrefix: resolve(values['profile-prefix']) } : {}),
    modelProfile,
  };
  const sourcePage = await extractPage(hierarchyHtml, { includeImages: false });
  const expectedSourceIds = sourceIdsFromPage(sourcePage);
  const model = await createNeko(options);
  const results = [];
  let cache;
  let backend;
  let preflightError;
  try {
    try {
      cache = await model.cache.model.status();
      backend = await model.backend.current();
    } catch (error) { preflightError = errorInfo(error); }
    const blocked = fixtureMismatch || preflightError || (!values['allow-network'] && !cache?.verified);
    if (blocked) {
      const error = fixtureMismatch
        ? { name: 'FixtureIntegrityError', code: 'FIXTURE_MISMATCH', message: 'Fixture bytes or dimensions do not match the committed oracle manifest.' }
        : preflightError ?? { name: 'ModelCacheError', code: 'MODEL_CACHE_UNAVAILABLE', message: 'Verified model assets are not present; no inference or download was attempted. Pass --allow-network only to explicitly permit downloads.' };
      for (const testCase of cases) results.push(makeResult(testCase, 'blocked', { error }));
    } else {
      let profileMismatch = false;
      for (const testCase of cases) {
        if (profileMismatch) {
          results.push(makeResult(testCase, 'blocked', { error: { name: 'ProfileMismatch', code: 'MODEL_PROFILE_MISMATCH', message: `Requested ${modelProfile}, but the first completed inference did not report that selected profile.` } }));
          continue;
        }
        try {
          let raw;
          if (testCase.kind === 'infer-text') raw = await model.infer({ prompt: testCase.prompt, maxNewTokens: testCase.maxNewTokens, ...(contextOverride === undefined ? {} : { contextWindowTokens: contextOverride }) });
          else if (testCase.kind === 'infer-image') raw = await model.infer({ image: new Blob([imageRaster], { type: 'image/png' }), prompt: testCase.prompt, maxNewTokens: testCase.maxNewTokens, ...(contextOverride === undefined ? {} : { contextWindowTokens: contextOverride }) });
          else raw = await model.describe(testCase.html, { format: 'json', language: 'en', maxNewTokens: testCase.maxNewTokens, contextWindowTokens: testCase.contextWindowTokens });
          const observedProfile = runProfile(raw, testCase.id);
          const profileKnown = observedProfile === modelProfile;
          const evaluation = testCase.id === 'hierarchy'
            ? scoreHierarchy(raw, expectedSourceIds)
            : score(testCase, raw, expectedSourceIds);
          results.push({ id: testCase.id, fixtureId: testCase.fixtureId, kind: testCase.kind, status: profileKnown ? 'completed' : 'profile-unverified', attempted: true, observedProfile, expectedProfile: modelProfile, backend: backendValue(raw, testCase.id), ...(evaluation ? { evaluation } : { evaluationError: 'Inference result did not contain the expected text output.' }), output: raw });
          if (!profileKnown) profileMismatch = true;
        } catch (error) {
          results.push(makeResult(testCase, 'failed', { attempted: true, error: errorInfo(error), ...await runtimeStatusEvidence(model) }));
        }
      }
    }
  } finally {
    await model.dispose();
  }
  return { results, model: { id: MODEL_ID, revision: MODEL_REVISION }, requestedBackend: backend, cache: cacheEvidence(cache), sourceIds: expectedSourceIds, progress: [...progress.values()], profiles: values['profile-prefix'] ? await readProfileTraces(resolve(values['profile-prefix'])) : [] };
}

async function readProfileTraces(prefix) {
  const directory = dirname(prefix);
  let filenames;
  try { filenames = (await readdir(directory)).filter((name) => name.startsWith(basename(prefix)) && name.endsWith('.json')); }
  catch { return []; }
  const traces = [];
  for (const name of filenames) {
    const path = resolve(directory, name);
    try {
      const events = JSON.parse(await readFile(path, 'utf8'));
      const providerEvents = {};
      for (const event of Array.isArray(events) ? events : []) {
        const provider = event?.args?.provider;
        if (typeof provider === 'string') providerEvents[provider] = (providerEvents[provider] ?? 0) + 1;
      }
      traces.push({ path, eventCount: Array.isArray(events) ? events.length : null, providerEvents });
    } catch (error) { traces.push({ path, error: errorInfo(error) }); }
  }
  return traces;
}

async function runBrowser() {
  const { chromium } = await import('@playwright/test');
  const browserAssets = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];
  const server = createServer(async (request, response) => {
    const path = new URL(request.url ?? '/', 'http://localhost').pathname;
    let file;
    let type;
    if (path === '/' || path === '/index.html') { response.writeHead(200, { 'content-type': 'text/html' }).end('<!doctype html><meta charset="utf-8"><title>Neko quality run</title>'); return; }
    if (path === '/dist/browser/neko.js') { file = resolve(root, 'dist/browser/neko.js'); type = 'text/javascript'; }
    else {
      const asset = path.split('/').at(-1);
      if (path.startsWith('/dist/browser/assets/') && browserAssets.includes(asset)) {
        file = resolve(root, 'dist/browser/assets', asset);
        type = asset.endsWith('.wasm') ? 'application/wasm' : 'text/javascript';
      }
    }
    if (!file) { response.writeHead(404).end('Not found'); return; }
    try { response.writeHead(200, { 'content-type': type }).end(await readFile(file)); }
    catch { response.writeHead(404).end('Missing browser build asset'); }
  });
  server.listen(browserPort, '127.0.0.1');
  await once(server, 'listening');
  const origin = `http://127.0.0.1:${browserPort}`;
  let browser;
  let context;
  let modelMirror;
  let modelMirrorOrigin;
  let mirrorSeedComplete = false;
  let persistent = false;
  const seedMirrorUrls = new Set();
  const externalRequests = [];
  try {
    let seedPayload;
    if (!values['allow-network']) {
      modelMirror = await createModelMirror({ cacheDir: resolve(values['cache-dir']), profile: modelProfile, allowedOrigin: origin });
      modelMirror.listen(0, '127.0.0.1');
      await once(modelMirror, 'listening');
      modelMirrorOrigin = `http://127.0.0.1:${modelMirror.address().port}`;
      const modelFiles = getRegisteredModelProfile(modelProfile).files;
      const mirrorBaseUrl = new URL(MODEL_MIRROR_PATH, `${modelMirrorOrigin}/`).href;
      const files = Object.entries(modelFiles).map(([name, spec]) => ({
        name,
        size: spec.size,
        sha256: spec.sha256,
        sourceUrl: new URL(name, mirrorBaseUrl).href,
        cacheUrl: modelFileUrl(name),
      }));
      for (const file of files) seedMirrorUrls.add(file.sourceUrl);
      const runtimePath = '/dist/browser/assets/ort-wasm-simd-threaded.asyncify.wasm';
      const runtimeBytes = await readFile(resolve(root, runtimePath.slice(1)));
      const runtimeAsset = { url: new URL(runtimePath, origin).href, size: runtimeBytes.byteLength, sha256: sha256(runtimeBytes) };
      seedPayload = { cacheName: transformersEnv.cacheKey, profile: modelProfile, origin, mirrorOrigin: modelMirrorOrigin, mirrorBaseUrl, modelBaseUrl: MODEL_BASE_URL, files, runtimeAsset };
    }
    if (values['browser-profile']) {
      persistent = true;
      context = await chromium.launchPersistentContext(resolve(values['browser-profile']), { headless: !values.headed, serviceWorkers: 'block' });
    } else {
      browser = await chromium.launch({ headless: !values.headed });
      context = await browser.newContext({ serviceWorkers: 'block' });
    }
    await context.route('**/*', (route) => {
      const request = route.request();
      const requestUrl = new URL(request.url());
      if (requestUrl.origin === origin) return route.continue();
      const isSeedRequest = !mirrorSeedComplete && request.method() === 'GET' && seedMirrorUrls.has(request.url());
      if (isSeedRequest) return route.continue();
      const blocked = !values['allow-network'];
      externalRequests.push({ origin: requestUrl.origin, hostname: requestUrl.hostname, path: requestUrl.pathname, method: request.method(), resourceType: request.resourceType(), blocked });
      if (!blocked) return route.continue();
      return route.abort('blockedbyclient');
    });
    const page = await context.newPage();
    await page.goto(origin, { waitUntil: 'load' });
    let cacheSeed;
    if (seedPayload) {
      cacheSeed = await page.evaluate(async (payload) => {
        await import('/dist/browser/neko.js');
        if (typeof globalThis.caches?.open !== 'function') throw new Error('CacheStorage is unavailable at the benchmark origin');
        const cache = await globalThis.caches.open(payload.cacheName);
        const digestHex = async (bytes) => {
          const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
          return Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
        };
        const verifiedFiles = [];
        for (const file of payload.files) {
          if (new URL(file.sourceUrl).origin !== payload.mirrorOrigin || file.cacheUrl !== `${payload.modelBaseUrl}${file.name}`) throw new Error(`Unexpected model cache seed URL for ${file.name}`);
          const response = await fetch(file.sourceUrl, { cache: 'no-store' });
          if (!response.ok) throw new Error(`Local model mirror returned HTTP ${response.status} for ${file.name}`);
          if (response.headers.get('content-length') !== String(file.size)) throw new Error(`Local model mirror length mismatch for ${file.name}`);
          const bytes = await response.arrayBuffer();
          const digest = await digestHex(bytes);
          if (bytes.byteLength !== file.size || digest !== file.sha256) throw new Error(`Local model mirror integrity mismatch for ${file.name}`);
          await cache.put(file.cacheUrl, new Response(bytes));
          verifiedFiles.push({ name: file.name, size: bytes.byteLength, sha256: digest });
        }
        if (new URL(payload.runtimeAsset.url).origin !== payload.origin) throw new Error('Runtime cache seed must use the benchmark origin');
        const runtimeResponse = await fetch(payload.runtimeAsset.url, { cache: 'no-store' });
        if (!runtimeResponse.ok) throw new Error(`Benchmark server returned HTTP ${runtimeResponse.status} for the runtime binary`);
        const runtimeBytes = await runtimeResponse.arrayBuffer();
        const runtimeSha256 = await digestHex(runtimeBytes);
        if (runtimeBytes.byteLength !== payload.runtimeAsset.size || runtimeSha256 !== payload.runtimeAsset.sha256) throw new Error('Runtime binary cache seed integrity mismatch');
        await cache.put(payload.runtimeAsset.url, new Response(runtimeBytes));
        return {
          source: 'verified-node-cache-local-mirror',
          cacheName: payload.cacheName,
          profile: payload.profile,
          files: verifiedFiles,
          totalModelBytes: verifiedFiles.reduce((total, file) => total + file.size, 0),
          runtimeAsset: { url: payload.runtimeAsset.url, size: runtimeBytes.byteLength, sha256: runtimeSha256 },
        };
      }, seedPayload);
      mirrorSeedComplete = true;
      await context.setOffline(true);
    }
    const input = {
      device,
      modelProfile,
      allowNetwork: values['allow-network'],
      imageDataUrl,
      textPrompt,
      hierarchyHtml,
      contextOverride,
      cases: cases.map(({ id, fixtureId, kind, prompt, html, maxNewTokens, contextWindowTokens }) => ({ id, fixtureId, kind, prompt, html, maxNewTokens, contextWindowTokens })),
      expectedSourceIds: [],
    };
    const result = await page.evaluate(async (payload) => {
      const api = await import('/dist/browser/neko.js');
      const sourcePage = await api.extractPage(payload.hierarchyHtml, { includeImages: false });
      payload.expectedSourceIds = sourcePage.paragraphs.map(({ id }) => id);
      const progress = new Map();
      const addProgress = (event) => {
        if (!event || typeof event !== 'object') return;
        const phase = typeof event.phase === 'string' ? event.phase : 'unknown';
        const file = typeof event.file === 'string' ? event.file : 'unknown';
        const key = JSON.stringify([phase, file]);
        const previous = progress.get(key);
        if (!previous && progress.size >= 64) return;
        const summary = { phase, file, eventCount: (previous?.eventCount ?? 0) + 1 };
        const loaded = typeof event.loaded === 'number' && Number.isFinite(event.loaded) ? event.loaded : undefined;
        const total = typeof event.total === 'number' && Number.isFinite(event.total) ? event.total : undefined;
        if (loaded !== undefined) summary.loaded = Math.max(previous?.loaded ?? 0, loaded);
        else if (previous?.loaded !== undefined) summary.loaded = previous.loaded;
        if (total !== undefined) summary.total = total;
        else if (previous?.total !== undefined) summary.total = previous.total;
        progress.set(key, summary);
      };
      let model;
      let cache;
      let backend;
      let preflightError;
      const results = [];
      let profileMismatch = false;
      try {
        model = await api.createNeko({ device: payload.device, localFilesOnly: !payload.allowNetwork, modelProfile: payload.modelProfile, progressCallback: addProgress });
        try { cache = await model.cache.model.status(); backend = await model.backend.current(); }
        catch (error) { preflightError = { name: error?.name ?? 'Error', message: error?.message ?? String(error), code: error?.code, stage: error?.stage }; }
        if (preflightError || (!payload.allowNetwork && !cache?.verified)) {
          const error = preflightError ?? { name: 'ModelCacheError', code: 'MODEL_CACHE_UNAVAILABLE', message: 'Verified browser cache is missing; no inference or download was attempted.' };
          for (const testCase of payload.cases) results.push({ id: testCase.id, fixtureId: testCase.fixtureId, kind: testCase.kind, status: 'blocked', attempted: false, error });
        } else {
          for (const testCase of payload.cases) {
            if (profileMismatch) {
              results.push({ id: testCase.id, fixtureId: testCase.fixtureId, kind: testCase.kind, status: 'blocked', attempted: false, error: { name: 'ProfileMismatch', code: 'MODEL_PROFILE_MISMATCH', message: `Requested ${payload.modelProfile}, but the preceding inference did not report that selected profile.` } });
              continue;
            }
            try {
              let output;
              if (testCase.kind === 'infer-text') output = await model.infer({ prompt: testCase.prompt, maxNewTokens: testCase.maxNewTokens, ...(payload.contextOverride === undefined ? {} : { contextWindowTokens: payload.contextOverride }) });
              else if (testCase.kind === 'infer-image') output = await model.infer({ image: payload.imageDataUrl, prompt: testCase.prompt, maxNewTokens: testCase.maxNewTokens, ...(payload.contextOverride === undefined ? {} : { contextWindowTokens: payload.contextOverride }) });
              else output = await model.describe(testCase.html, { format: 'json', language: 'en', maxNewTokens: testCase.maxNewTokens, contextWindowTokens: testCase.contextWindowTokens });
              const observedProfile = testCase.id === 'hierarchy' ? output?.metadata?.model?.profile ?? null : output?.model?.profile ?? null;
              const profileKnown = observedProfile === payload.modelProfile;
              results.push({ id: testCase.id, fixtureId: testCase.fixtureId, kind: testCase.kind, status: profileKnown ? 'completed' : 'profile-unverified', attempted: true, observedProfile, expectedProfile: payload.modelProfile, output });
              if (!profileKnown) profileMismatch = true;
            } catch (error) {
              let statusEvidence = {};
              if (typeof model?.runtimeStatus === 'function') {
                try { statusEvidence.runtimeStatus = await model.runtimeStatus(); }
                catch (statusError) { statusEvidence.runtimeStatusError = { name: statusError?.name ?? 'Error', message: statusError?.message ?? String(statusError), code: statusError?.code, stage: statusError?.stage }; }
              }
              results.push({ id: testCase.id, fixtureId: testCase.fixtureId, kind: testCase.kind, status: 'failed', attempted: true, error: { name: error?.name ?? 'Error', message: error?.message ?? String(error), code: error?.code, stage: error?.stage }, ...statusEvidence });
            }
          }
        }
      } catch (error) {
        const detail = { name: error?.name ?? 'Error', message: error?.message ?? String(error), code: error?.code, stage: error?.stage };
        for (const testCase of payload.cases) results.push({ id: testCase.id, fixtureId: testCase.fixtureId, kind: testCase.kind, status: 'failed', attempted: false, error: detail });
      } finally { await model?.dispose(); }
      return { model: { id: api.MODEL_ID, revision: api.MODEL_REVISION }, cache: cache ? { downloaded: cache.downloaded, verified: cache.verified, bytes: cache.bytes, totalBytes: cache.totalBytes, fileCount: cache.files?.length ?? null, presentFiles: cache.files?.filter(({ present }) => present).length ?? null } : null, backend, sourceIds: payload.expectedSourceIds, progress: [...progress.values()], results };
    }, input);
    for (const resultCase of result.results) {
      const original = cases.find(({ id }) => id === resultCase.id);
      if (resultCase.status === 'completed' || resultCase.status === 'profile-unverified') {
        const evaluation = score(original, resultCase.output, result.sourceIds);
        if (evaluation) resultCase.evaluation = evaluation;
        else resultCase.evaluationError = 'Inference result did not contain the expected text output.';
        resultCase.backend = backendValue(resultCase.output, original.id);
      }
    }
    const hfRequests = externalRequests.filter(({ hostname }) => hostname === 'huggingface.co' || hostname.endsWith('.huggingface.co') || hostname.endsWith('.hf.co'));
    const networkPolicyViolation = !values['allow-network'] && externalRequests.length > 0;
    const networkEvidence = {
      allowNetwork: values['allow-network'],
      enforcement: values['allow-network'] ? 'explicit-opt-in' : 'local-cache-seed-then-browser-offline-and-external-route-block',
      externalRequests,
      blockedExternalRequestCount: externalRequests.filter(({ blocked }) => blocked).length,
      hfModelRequestCount: hfRequests.length,
      noHfModelRequests: hfRequests.length === 0,
    };
    return {
      ...result,
      ...(cacheSeed ? { cacheSeed } : {}),
      networkPolicyViolation,
      networkEvidence,
      browser: { engine: 'chromium', origin, port: browserPort, version: browser?.version() ?? context.browser()?.version() ?? null, headed: values.headed, persistentProfile: persistent },
    };
  } finally {
    await context?.close().catch(() => undefined);
    await browser?.close().catch(() => undefined);
    if (modelMirror?.listening) await new Promise((resolveClose, rejectClose) => modelMirror.close((error) => error ? rejectClose(error) : resolveClose()));
    await new Promise((resolveClose, rejectClose) => server.close((error) => error ? rejectClose(error) : resolveClose()));
  }
}

function caseStatus(results) {
  if (fixtureMismatch) return 'fixture-invalid';
  if (results.every(({ status }) => status === 'completed')) return 'completed';
  if (results.some(({ status }) => status === 'failed' || status === 'profile-unverified')) return 'incomplete';
  return 'blocked';
}

const config = {
  runtime,
  device,
  modelProfile,
  selectedCase,
  allowNetwork: values['allow-network'],
  cacheDir: values['cache-dir'] ? resolve(values['cache-dir']) : null,
  profilePrefix: values['profile-prefix'] ? resolve(values['profile-prefix']) : null,
  browserProfile: values['browser-profile'] ? resolve(values['browser-profile']) : null,
  headed: values.headed,
  browserPort: runtime === 'browser' ? browserPort : null,
  contextWindowTokens: contextOverride ?? null,
  maxNewTokens: tokenOverride ?? null,
  automaticRetries: 0,
};
let execution;
let runnerError;
if (fixtureMismatch) {
  runnerError = { name: 'FixtureIntegrityError', code: 'FIXTURE_MISMATCH', message: 'Image source/raster hashes, dimensions, or hierarchy paragraph count differ from the committed manifest.' };
} else {
  try { execution = runtime === 'browser' ? await runBrowser() : await runNode(); }
  catch (error) { runnerError = errorInfo(error); }
}
const results = execution?.results ?? cases.map((testCase) => makeResult(testCase, 'blocked', { error: runnerError ?? { name: 'RuntimeError', message: 'Runtime did not produce a result.' } }));
const model = execution?.model ?? { id: null, revision: null };
const output = {
  schemaVersion: ARTIFACT_VERSION,
  fixtureVersion: FIXTURE_VERSION,
  policyVersion: POLICY_VERSION,
  evaluatorVersion: EVALUATOR_VERSION,
  oracleSha256,
  runId: new Date().toISOString(),
  status: execution?.networkPolicyViolation ? 'incomplete' : caseStatus(results),
  comparable: results.length > 0 && results.every(({ status }) => status === 'completed') && !fixtureMismatch && !execution?.networkPolicyViolation,
  stimulusKey,
  config,
  environment: { node: process.version, platform: process.platform, arch: process.arch, sharp: sharp.versions.sharp, libvips: sharp.versions.vips },
  model,
  modelProfileEvidence: { requested: modelProfile, observed: results.map(({ id, observedProfile }) => ({ id, observed: observedProfile ?? null })) },
  backend: { requestedDevice: device, requestedRuntime: runtime, observed: execution?.requestedBackend ?? execution?.backend ?? null },
  caseInputs: contract.caseInputs,
  cache: execution?.cache ?? null,
  fixtures: fixtureEntries,
  imageRenderer: { sourceSha256: fixtureEntries.find(({ id }) => id === manifest.image.id).sha256, rasterSha256: fixtureEntries.find(({ id }) => id === `${manifest.image.id}-raster`).sha256, expectedRasterSha256: manifest.image.raster.sha256, bytesIdenticalToManifest: !fixtureMismatch },
  sourceIdCount: execution?.sourceIds?.length ?? 0,
  progress: execution?.progress ?? [],
  profiles: execution?.profiles ?? [],
  ...(execution?.browser ? { browser: execution.browser } : {}),
  ...(execution?.cacheSeed ? { browserCacheSeed: execution.cacheSeed } : {}),
  ...(execution?.networkEvidence ? { networkEvidence: execution.networkEvidence } : {}),
  ...(execution?.networkPolicyViolation ? { networkPolicyViolation: true } : {}),
  ...(runnerError ? { runnerError } : {}),
  results,
};

if (values.output) {
  const target = resolve(values.output);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({ status: output.status, comparable: output.comparable, stimulusKey: output.stimulusKey, output: target, resultCount: results.length }));
} else {
  console.log(JSON.stringify(output, null, 2));
}
