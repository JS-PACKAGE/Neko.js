# Usage

## Current scope

Neko.js currently exposes a real local image-and-prompt inference prototype, not the full website-report workflow. Use `createPrototype()` for one image and one prompt; page extraction and report-rendering helpers are separate utilities and are not yet wired into an end-to-end website summarizer. The model may produce inaccurate descriptions. Do not use its output for security, authorization, or other consequential decisions.

The model is pinned to `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`. Its embedding and decoder are Q4 and its vision encoder is FP16. Transformers.js 4.2.0 and ONNX Runtime run the inference. Model assets are fetched from Hugging Face on first use and checked against pinned sizes and SHA-256 digests before being used or cached.

## Install and build

- Requires Node.js 22 or newer; the package is ESM.
- `npm ci` installs locked dependencies. The Node path overrides Transformers.js's native `onnxruntime-node` to 1.30.0 for the model's `com.microsoft:CausalConvWithState` operator; browser inference continues to use its separate locked ONNX Runtime Web package unchanged. `npm run build` emits JavaScript, declarations, browser bundles, and browser ONNX Runtime WASM assets into `dist/`.
- `npm test`, `npm run typecheck`, and `npm run lint` run the Node test suite, TypeScript check, and ESLint respectively.
- `npm run test:browser` runs the Playwright suite in Chromium, Firefox, and WebKit; install those browsers with Playwright before running it.
- `examples/browser-prototype.html` is a static browser demo; serve it from a secure HTTP(S) origin after building. This project does not provide an application HTTP server.

`npm run smoke:browser` launches the real Chromium demo with a generated red-square image and a whitespace-preserving prompt. It defaults to explicit WebGPU. The explicit `cpu`/WASM helper mode is retained for diagnostics but is unsupported by this pinned model: session creation is known to fail at `GatherBlockQuantized(1)` and the demo does not fall back. To reuse local Node-cached model files without downloading them again, set `NEKO_MODEL_CACHE` to the model-revision directory containing `config.json` and `onnx/`; without it, the smoke uses the pinned Hugging Face URLs normally. Use `--headed` to launch Chromium with a fresh isolated persistent profile, or `--offline-reload` to run then reload the UI with Hugging Face model requests blocked. Either real inference run may download approximately 871 MB of model assets; this long, network-backed smoke is intentionally excluded from `npm run test:browser`. To inspect Chromium's WebGPU adapter without loading model weights, use `npm run smoke:browser -- webgpu 8 --enable-unsafe-webgpu --preflight-only`; this is capability information only, not inference or proof of hardware execution.

## Verified measurements

Measurements used Node.js 22.23.3, Transformers.js 4.2.0, native ONNX Runtime 1.30.0, and the pinned Qwen revision. The hybrid provider profile recorded 128 WebGPU events for embedding, 536 WebGPU + 50 CPU for vision, and 169,472 WebGPU + 12,288 CPU for decoder; this is not an all-GPU execution claim.

- Populated verified cache: model load 6,414.542 ms, first non-empty decoded output 1,487.161 ms, generation 9,353.701 ms, inference total 9,486.618 ms; zero network requests.
- Cold empty cache: 13 files / 871,364,778 bytes downloaded; outer load 37,118.820 ms, inference load 37,114.206 ms, first output 1,228.452 ms, generation 7,728.604 ms, inference total 7,796.944 ms.
- Fresh offline process using the same cache: outer load 4,655.131 ms, inference load 4,652.725 ms, first output 2,447.610 ms, generation 9,874.194 ms, inference total 9,997.882 ms; zero network requests.

`loadMs` includes cache prefetch/verification and processor/model loading; first-output time is not raw-token or kernel latency. Process RSS was about 2.147 GB after load, 1.966 GB during inference, and 0.731 GB after disposal. GPU memory is unknown. These are measured runs, not cross-platform performance guarantees.


## Image inference

Node.js example:

```js
import { createPrototype } from 'neko.js';

const model = await createPrototype({ device: 'webgpu' });
try {
  const result = await model.infer({
    image: './photo.png',
    prompt: 'Describe the image.',
    maxNewTokens: 128,
  });
  console.log(result.text);
  console.log(result.model, result.backend, result.timings, result.memory);
} finally {
  await model.dispose();
}
```

Browser example:

```js
import { createPrototype } from './dist/neko.js';

const model = await createPrototype({ device: 'webgpu' });
try {
  const result = await model.infer({ image: fileInput.files[0], prompt: 'Describe the image.' });
  output.textContent = result.text;
  details.textContent = JSON.stringify({ model: result.model, backend: result.backend, timings: result.timings, memory: result.memory }, null, 2);
} finally {
  await model.dispose();
}
```

`createPrototype(options?)` loads the pinned processor and model before it resolves. It accepts `device: 'cpu' | 'webgpu'`, `localFilesOnly`, a `progressCallback`, and `wasmPaths` for browser WASM files. `cacheDir` customizes the native cache directory in Node.js. Browser model caching uses the Cache API; use a secure context and allow the model host through your network/CSP policy. `localFilesOnly: true` prevents network downloads and fails when required files are missing from the cache. WebGPU availability failures are explicit; the prototype does not silently switch devices. CPU is explicit and is never selected automatically after a WebGPU failure.

An actual demo UI run in persistent headed Chromium 153 selected WebGPU, uploaded a football image, and rendered `A football player in a Paris Saint-Germain uniform is dribbling the ball while being pursued by a defender in a dark blue kit.` The browser reported Apple/Metal 3, `fallback:false`, and `shader-f16`; all loaded session configurations used WebGPU. UI load/preprocess/first non-empty output/generation/total timings were 5,238.6/271.6/3,963.9/4,717.6/4,989.2 ms. Separate headed-browser smoke instrumentation observed 2,656 `dispatchWorkgroups` and 1,104 `queue.submit` calls. Reloading the same persistent profile with `localFilesOnly` enabled produced the same output with no observed Hugging Face requests; Cache Storage used about 895 MB. This proves an actual inference but not that every operator ran on GPU; GPU memory was not measured. The browser CPU/WASM path is confirmed unsupported by this pinned model: session creation fails with `GatherBlockQuantized(1)`, and the demo does not fall back.

`infer({ image, prompt, maxNewTokens? })` accepts a Node image path or supported image input, and in browsers a `Blob`/`File` is suitable for a user-selected image. The prompt must contain non-whitespace text. `maxNewTokens` defaults to 128 and must be an integer from 1 through 2048. Browser image URLs are subject to CORS; a user-selected `File` avoids a cross-origin image request.

The result includes the pinned model identity and dtype, observed runtime/device, configured execution providers, loaded ONNX session configuration, timings, and memory fields. `providerEvidence: 'loaded-session-configuration'` and session device values describe the loaded configuration only. They do not prove that every operator ran on a GPU. `loadMs` includes model-cache prefetch/verification and processor/model loading; `firstTokenMs` is the first non-empty decoded streamer callback, not a raw-token or kernel timing. JavaScript/GPU memory values are unavailable and are reported as `null`; do not substitute estimates. Node and browser performance/results are not asserted to be equivalent.

Call `dispose()` when finished to release model resources and restore the runtime's shared Transformers.js configuration. `status()` returns engine-cache state. Progress callback events are runtime events; treat their shape as opaque unless a specific event contract is documented.

## Page extraction and image preparation

```js
import { extractPage, loadImage } from 'neko.js';

const page = await extractPage('https://example.test/article', {
  maxHtmlBytes: 2 * 1024 * 1024,
  maxImages: 20,
  timeoutMs: 10_000,
});
const prepared = await loadImage(page.images[0]);
// prepared.data is a normalized PNG byte array; prepared.imageId identifies its source.
```

`extractPage(input, options?)` accepts an HTTP(S) URL or HTML string. Provide `baseUrl` to resolve relative URLs in HTML strings. Defaults limit HTML to 2 MiB, images to 20 unique URLs, image downloads to 10 MiB, and each request to 10 seconds. The parser extracts semantic text blocks and image sources without executing page scripts or fetching linked resources. Image sources include `img`, `picture/source[srcset]`, inline styles, `background` attributes, and `og:image`; provenance is retained.

`loadImage(image, options?)` is a browser API requiring `createImageBitmap` and a canvas implementation. It fetches HTTP(S) raster images or decodes raster data URLs, checks MIME type against the image bytes, caps input size and decoded dimensions, applies image orientation, scales to at most 1280×1280 without enlargement, and emits PNG bytes. SVG and mismatched/invalid content are rejected. It preprocesses; it does not generate a visual description. Network image loads are subject to CORS.

## Report utilities

`renderMarkdown(report)` renders a `StructuredReport`, escaping untrusted report text and only creating provenance links for HTTP(S) URLs. `validateStructuredReport(report, page)` checks citations against extracted page/image IDs. These utilities do not call the model or create an end-to-end website report.

## Cache and backend notes

The model manifest fixes the Hugging Face revision and SHA-256/size of required files. On cache hits, integrity is checked before model use; a mismatch is an error, not a silent cache miss. The `prefetch(signal?)` method is exposed on a created prototype for explicitly ensuring pinned model files are present. Browser cache storage is managed by the browser and may be cleared by its user or eviction policy.

`device: 'cpu'` and `device: 'webgpu'` are explicit requests. Browser WebGPU requires an available adapter with `shader-f16` support for the FP16 vision encoder. Browser CPU/WASM cannot load the pinned model because ONNX Runtime Web lacks an implementation for `GatherBlockQuantized(1)`. The Node CPU inference path remains untested. There is no automatic provider fallback or cross-platform parity guarantee.

## Security and privacy

Page HTML and generated model text are untrusted data. Parsing is inert, and the extraction helper does not load page scripts or subresources. Markdown output is escaped but must still be rendered with a safe Markdown renderer. Local model inference does not prevent the model from generating harmful or incorrect content. See [SECURITY.md](../../SECURITY.md) for the threat model and reporting process.
