import { parseArgs } from 'node:util';
import { readFile, stat, readdir, mkdir, realpath } from 'node:fs/promises';
import { resolve, dirname, basename, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const { values } = parseArgs({ options: {
  image: { type: 'string' }, prompt: { type: 'string' }, 'prompt-file': { type: 'string' },
  device: { type: 'string', default: 'webgpu' }, offline: { type: 'boolean', default: false },
  'model-profile': { type: 'string', default: 'default' }, 'allow-remote-image': { type: 'boolean', default: false },
  'cache-dir': { type: 'string' }, 'probe-only': { type: 'boolean', default: false },
  'profile-prefix': { type: 'string' },
} });

let approvedModelTransfers = 0;
let approvedImageTransfers = 0;
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
  if (!['default', 'all-q4'].includes(values['model-profile'])) throw new Error('--model-profile must be default or all-q4');
  if (values.prompt !== undefined && values['prompt-file'] !== undefined) throw new Error('Provide only one of --prompt and --prompt-file');
  const modelProfile = values['model-profile'];
  const profilePrefix = values['profile-prefix'] ? resolve(values['profile-prefix']) : undefined;
  let previousProfiles = new Set();
  if (profilePrefix) {
    await mkdir(dirname(profilePrefix), { recursive: true });
    previousProfiles = new Set(await readdir(dirname(profilePrefix)));
  }
  const ort = await import('onnxruntime-node');
  const { env } = await import('@huggingface/transformers');
  console.log(JSON.stringify({ phase: 'environment', node: process.version, transformers: env.version, onnxruntime: ort.env.versions, platform: process.platform, arch: process.arch, requestedDevice: values.device, modelProfile, runtime: 'node', gpuMemoryBytes: null, processMemory: process.memoryUsage() }));
  if (values['probe-only']) {
    console.log(JSON.stringify({ phase: 'native-provider-probe', ...(await probeNativeProvider(values.device, profilePrefix)), processMemory: process.memoryUsage() }));
    if (profilePrefix) await reportProfiles(profilePrefix, previousProfiles);
    return;
  }
  if (!values.image) throw new Error('--image requires a real image path or URL');
  let image;
  let localImagePath;
  let approvedRemoteImage;
  if (/^https?:\/\//i.test(values.image)) {
    approvedRemoteImage = new URL(values.image);
    if (!values['allow-remote-image']) throw new Error('Remote image inputs require explicit --allow-remote-image approval for this URL');
    if (values.offline) throw new Error('--offline cannot be combined with a remote image URL');
    image = approvedRemoteImage;
  } else {
    const path = values.image.startsWith('file:') ? fileURLToPath(values.image) : resolve(values.image);
    localImagePath = await realpath(path);
    if (!(await stat(localImagePath)).isFile()) throw new Error('--image must identify a file');
    image = pathToFileURL(localImagePath);
  }
  const prompt = values['prompt-file'] !== undefined ? await readFile(values['prompt-file'], 'utf8') : values.prompt;
  if (prompt === undefined || !prompt.trim()) throw new Error('Provide a nonempty --prompt or --prompt-file');
  const { createNeko } = await import('../dist/node/index.js');
  const downloads = new Map();
  const progressCallback = event => {
    if (event?.phase === 'download') downloads.set(event.file, Math.max(downloads.get(event.file) ?? 0, event.loaded));
  };
  const policy = {
    network(url, kind) {
      if (kind === 'model') { approvedModelTransfers++; return true; }
      if (kind === 'image' && approvedRemoteImage?.href === url.href) { approvedImageTransfers++; return true; }
      return false;
    },
    localFiles(path) { return localImagePath !== undefined && path === localImagePath; },
  };
  const loadStarted = performance.now();
  const model = await createNeko({ device: values.device, modelProfile, policy, localFilesOnly: values.offline, progressCallback, ...(values['cache-dir'] ? { cacheDir: resolve(values['cache-dir']) } : {}), ...(profilePrefix ? { profilePrefix } : {}) });
  const outerLoadMs = performance.now() - loadStarted;
  try {
    const [modelCache, engineCache, requestedBackend] = await Promise.all([
      model.cache.model.status(),
      model.cache.engine.status(),
      model.backend.current(),
    ]);
    console.log(JSON.stringify({ phase: 'created', outerLoadMs, requestedDevice: values.device, requestedProfile: modelProfile, requestedBackend, networkPolicyApprovals: { modelTransfers: approvedModelTransfers, imageInputs: approvedImageTransfers }, downloadedBytes: [...downloads.values()].reduce((sum, bytes) => sum + bytes, 0), downloadedFiles: Object.fromEntries(downloads), modelCache, engineCache, processMemory: process.memoryUsage() }));
    const result = await model.infer({ image, prompt });
    if (result.model.profile !== modelProfile) throw new Error(`Requested profile ${modelProfile}, received ${result.model.profile}`);
    if (values.offline && (approvedModelTransfers !== 0 || approvedImageTransfers !== 0)) throw new Error('Offline inference unexpectedly approved a network transfer');
    const [modelCacheAfter, engineCacheAfter] = await Promise.all([
      model.cache.model.status(),
      model.cache.engine.status(),
    ]);
    console.log(JSON.stringify({ phase: 'inference', ...result, networkPolicyApprovals: { modelTransfers: approvedModelTransfers, imageInputs: approvedImageTransfers }, downloadedBytes: [...downloads.values()].reduce((sum, bytes) => sum + bytes, 0), downloadedFiles: Object.fromEntries(downloads), cache: { model: modelCacheAfter, engine: engineCacheAfter }, processMemory: process.memoryUsage() }));
  } finally {
    await model.dispose();
    console.log(JSON.stringify({ phase: 'disposed', processMemory: process.memoryUsage() }));
    if (profilePrefix) await reportProfiles(profilePrefix, previousProfiles);
  }
}

main().catch(error => { console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1; });
