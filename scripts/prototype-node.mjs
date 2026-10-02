import { parseArgs } from 'node:util';
import { readFile, stat, readdir, mkdir } from 'node:fs/promises';
import { resolve, dirname, basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const { values } = parseArgs({ options: {
  image: { type: 'string' }, prompt: { type: 'string' }, 'prompt-file': { type: 'string' },
  device: { type: 'string', default: 'webgpu' }, offline: { type: 'boolean', default: false },
  'cache-dir': { type: 'string' }, 'probe-only': { type: 'boolean', default: false },
  'profile-prefix': { type: 'string' },
} });

let attemptedNetworkRequests = 0;
let restoreNetwork;
let networkRequests = 0;

async function reportProfiles(prefix, previous) {
  const directory = dirname(prefix);
  const files = (await readdir(directory)).filter(name => name.startsWith(basename(prefix)) && name.endsWith('.json') && !previous.has(name));
  const traces = [];
  for (const name of files) {
    const path = join(directory, name);
    const events = JSON.parse(await readFile(path, 'utf8'));
    if (!Array.isArray(events)) throw new Error(`Unexpected ONNX profiling trace format: ${path}`);
    const providerEvents = {};
    for (const event of events) {
      const provider = event?.args?.provider;
      if (typeof provider === 'string') providerEvents[provider] = (providerEvents[provider] ?? 0) + 1;
    }
    traces.push({ path, providerEvents });
  }
  console.log(JSON.stringify({ phase: 'profiling', evidence: 'recorded-provider-events', traces }));
}

async function probeNativeProvider(device, profilePrefix) {
  const ort = await import('onnxruntime-node');
  // Microsoft ONNX Runtime v1.24.3 test/testdata/mul_1.onnx, git blob
  // 0b6dc510261322f3ff63deaa4114a0a023b35246: X * [1,2,3,4,5,6].
  const fixture = Buffer.from('CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4KDAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==', 'base64');
  const session = await ort.InferenceSession.create(fixture, {
    executionProviders: [device],
    ...(device === 'webgpu' ? { extra: { session: { disable_cpu_ep_fallback: '1' } } } : {}),
    ...(profilePrefix ? { enableProfiling: true, profileFilePrefix: profilePrefix } : {}),
  });
  try {
    const output = await session.run({ X: new ort.Tensor('float32', new Float32Array([2, 2, 2, 2, 2, 2]), [3, 2]) });
    const actual = Array.from(output.Y.data);
    const expected = [2, 4, 6, 8, 10, 12];
    if (actual.some((value, index) => value !== expected[index]) || actual.length !== expected.length) throw new Error('Native provider probe returned incorrect multiplication result');
    return { executionProviders: [device], cpuFallbackDisabled: device === 'webgpu', output: actual };
  } finally {
    try { if (profilePrefix) session.endProfiling(); }
    finally { await session.release(); }
  }
}

async function main() {
  if (Number(process.versions.node.split('.')[0]) < 22) throw new Error('This runner requires Node.js 22 or newer');
  if (!['cpu', 'webgpu'].includes(values.device)) throw new Error('--device must be cpu or webgpu');
  if (values.prompt !== undefined && values['prompt-file'] !== undefined) throw new Error('Provide only one of --prompt and --prompt-file');
  const profilePrefix = values['profile-prefix'] ? resolve(values['profile-prefix']) : undefined;
  let previousProfiles = new Set();
  if (profilePrefix) {
    await mkdir(dirname(profilePrefix), { recursive: true });
    previousProfiles = new Set(await readdir(dirname(profilePrefix)));
  }
  const ort = await import('onnxruntime-node');
  const { env } = await import('@huggingface/transformers');
  if (values.offline) {
    const previousGlobalFetch = globalThis.fetch;
    const previousEnvFetch = env.fetch;
    const blockedFetch = async () => {
      attemptedNetworkRequests++;
      throw new Error('Network access is disabled for this offline execution');
    };
    globalThis.fetch = blockedFetch;
    env.fetch = blockedFetch;
    restoreNetwork = () => { globalThis.fetch = previousGlobalFetch; env.fetch = previousEnvFetch; };
  }
  else {
    const previousEnvFetch = env.fetch;
    env.fetch = async (...args) => { networkRequests++; return previousEnvFetch(...args); };
    restoreNetwork = () => { env.fetch = previousEnvFetch; };
  }
  console.log(JSON.stringify({ phase: 'environment', node: process.version, transformers: env.version, onnxruntime: ort.env.versions, platform: process.platform, arch: process.arch, requestedDevice: values.device, runtime: 'node', gpuMemoryBytes: null, processMemory: process.memoryUsage() }));
  if (values['probe-only']) {
    console.log(JSON.stringify({ phase: 'native-provider-probe', ...(await probeNativeProvider(values.device, profilePrefix)), processMemory: process.memoryUsage() }));
    if (profilePrefix) await reportProfiles(profilePrefix, previousProfiles);
    return;
  }
  if (!values.image) throw new Error('--image requires a real image path or URL');
  let image = values.image;
  if (!/^https?:\/\//i.test(image)) {
    const path = image.startsWith('file:') ? fileURLToPath(image) : resolve(image);
    if (!(await stat(path)).isFile()) throw new Error('--image must identify a file');
    image = new Blob([await readFile(path)]);
  } else if (values.offline) throw new Error('--offline requires a local image file');
  const prompt = values['prompt-file'] !== undefined ? await readFile(values['prompt-file'], 'utf8') : values.prompt;
  if (prompt === undefined || !prompt.trim()) throw new Error('Provide a nonempty --prompt or --prompt-file');
  const { createNeko } = await import('../dist/node/index.js');
  const downloads = new Map();
  const progressCallback = event => {
    if (event?.phase === 'download') downloads.set(event.file, Math.max(downloads.get(event.file) ?? 0, event.loaded));
  };
  const loadStarted = performance.now();
  const model = await createNeko({ device: values.device, localFilesOnly: values.offline, progressCallback, ...(values['cache-dir'] ? { cacheDir: resolve(values['cache-dir']) } : {}), ...(profilePrefix ? { profilePrefix } : {}) });
  const outerLoadMs = performance.now() - loadStarted;
  try {
    console.log(JSON.stringify({ phase: 'created', outerLoadMs, networkRequests, downloadedBytes: [...downloads.values()].reduce((sum, bytes) => sum + bytes, 0), downloadedFiles: Object.fromEntries(downloads), modelCache: await model.cache.model.status(), engine: model.cache.engine.status(), processMemory: process.memoryUsage() }));
    const result = await model.infer({ image, prompt });
    if (values.offline && attemptedNetworkRequests !== 0) throw new Error('Offline inference attempted network access');
    console.log(JSON.stringify({ phase: 'inference', ...result, processMemory: process.memoryUsage() }));
  } finally {
    await model.dispose();
    console.log(JSON.stringify({ phase: 'disposed', processMemory: process.memoryUsage() }));
    if (profilePrefix) await reportProfiles(profilePrefix, previousProfiles);
  }
}

main().catch(error => { console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1; }).finally(() => {
  if (restoreNetwork) {
    restoreNetwork();
    console.log(JSON.stringify(values.offline ? { phase: 'offline-network', attemptedNetworkRequests } : { phase: 'network', networkRequests }));
  }
});
