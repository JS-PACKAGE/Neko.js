# Usage

## Current scope

Neko.js is a local multimodal SDK for Node.js and supported WebGPU browsers. `createNeko()` lazily owns the pinned Qwen model and exposes text/image inference, website-to-structured-report generation, backend status, and explicit model/engine cache controls. A report runs inert HTML extraction and real image inference for discovered images; it is not a crawler, and remote URL fetching must be protected by the calling application's network policy. Model output may be inaccurate; never use it for security or authorization decisions.

The model is pinned to `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`, with Q4 embeddings/decoder and an FP16 vision encoder. Initial use downloads approximately 871 MB. Each cached asset is verified against pinned size and SHA-256 values before use. Runtime inference bundles Transformers.js 4.2.0; Node uses ONNX Runtime 1.30.0.

## Install and build

- Requires Node.js 22 or newer; the package is ESM and is not published to npm.
- For Git repository use, clone the repository, then run `npm ci` and `npm run build`. This explicit local build also works with npm 11's approval of lifecycle scripts; avoid installing an unbuilt Git dependency when its `prepare` script has not been approved.
- Build the archive locally from the current working tree with `npm pack`; it includes `dist/` even though generated output is git-ignored. Install that local file in another project with `npm install /path/to/neko.js-1.1.0.tgz`. The `1.1.0` filename/version comes from this checkout's `package.json`. This project is not published to npm. Node bundles Transformers.js but keeps pinned `onnxruntime-node`, `sharp`, and `parse5` as direct runtime dependencies. If npm 11 blocks native postinstall scripts, approve only `onnxruntime-node@1.30.0` and `sharp@0.35.4`; do not grant blanket script approval.
- `npm test`, `npm run typecheck`, and `npm run lint` run the Node tests, typecheck, and ESLint. `npm run test:package:artifact` exercises the actual packed artifact in an isolated consumer, including public imports and TypeScript declarations. `npm run test:package` additionally runs text, image, and report inference using the packed consumer and may download about 871 MB on a cold cache.
- `npm run test:browser` runs the browser contract tests. `npm run smoke:browser` is an explicit real-model browser UI run and may download about 871 MB; use a secure HTTP(S) origin for the static demo.

Browser CPU/WASM is unsupported for this model because ONNX Runtime Web lacks `GatherBlockQuantized(1)`; creation reports an unsupported-backend error and never falls back. Node and browser providers are selected explicitly and must be validated separately.

## Historical prototype measurements

Measurements used Node.js 22.23.3, Transformers.js 4.2.0, native ONNX Runtime 1.30.0, and the pinned Qwen revision. The hybrid provider profile recorded 128 WebGPU events for embedding, 536 WebGPU + 50 CPU for vision, and 169,472 WebGPU + 12,288 CPU for decoder; this is not an all-GPU execution claim.

- Populated verified cache: model load 6,414.542 ms, first non-empty decoded output 1,487.161 ms, generation 9,353.701 ms, inference total 9,486.618 ms; zero network requests.
- Cold empty cache: 13 files / 871,364,778 bytes downloaded; outer load 37,118.820 ms, inference load 37,114.206 ms, first output 1,228.452 ms, generation 7,728.604 ms, inference total 7,796.944 ms.
- Fresh offline process using the same cache: outer load 4,655.131 ms, inference load 4,652.725 ms, first output 2,447.610 ms, generation 9,874.194 ms, inference total 9,997.882 ms; zero network requests.

`loadMs` includes cache verification and processor/model loading. `InferenceResult.timings.firstTokenMs` measures the first model-token callback from generation start; it is not the first non-empty decoded text chunk or kernel latency. The prototype measurements below separately recorded the first non-empty decoded output. Process RSS was about 2.147 GB after load, 1.966 GB during inference, and 0.731 GB after disposal. GPU memory is unknown. These are measured runs, not cross-platform performance guarantees.


## SDK usage

```js
import { createNeko } from 'neko.js';

const neko = await createNeko({ device: 'webgpu' });
try {
  // Image input is optional: without it, this is text-only inference.
  const answer = await neko.infer({
    image: './photo.png',
    prompt: 'Describe the visible scene.',
    maxNewTokens: 128,
    onToken: (text) => process.stdout.write(text),
  });
  console.log(answer.usage, answer.finishReason);

  const report = await neko.describe('https://example.test/article', {
    language: 'en',
    format: 'json',
    imageFailurePolicy: 'omit',
    maxNewTokens: 256,
  });
  console.log(report.page.summary, report.images);
  console.log(await neko.cache.model.status(), neko.cache.engine.status());
} finally {
  await neko.dispose();
}
```

`createNeko(options?)` sets up the selected backend and verified cache, then returns without loading the model. `cache.model.prefetch()` downloads and verifies model files but does not load the inference engine; the first `infer()` or `describe()` loads it lazily. The default device is `webgpu`; there is no automatic fallback. `cacheDir` overrides the Node cache location. Otherwise Node uses `~/Library/Caches/neko.js` on macOS, `%LOCALAPPDATA%/neko.js` on Windows, or `$XDG_CACHE_HOME/neko.js` / `~/.cache/neko.js` on Linux. Browsers use Cache Storage. `localFilesOnly: true` forbids network downloads and fails if a required verified asset is absent.

`infer({ prompt, image?, maxNewTokens?, contextWindowTokens?, signal?, onToken? })` accepts a Node path/file URL, an HTTP(S) image URL, a browser `Blob`/`File` or local `blob:` object URL, a supported raster data URL, or a structurally decoded image. In Node, supply `validateDestination`/network controls for remote images; browser remote images remain subject to CORS. A non-empty prompt is required; `maxNewTokens` defaults to 128. The result includes decoded text, `finishReason`, token usage, pinned model identity, observed backend/session configuration, timings, and memory values (unavailable memory is `null`). `firstTokenMs` is measured from the first model-token callback, not the first decoded text chunk. `onToken` receives decoded text chunks; callback failures fail the inference.

`describe(urlOrHtml, options?)` extracts page text and images, performs actual image inference, and generates a structured report by default. `maxNewTokens` defaults to 256 per generation phase. `format: 'markdown'` returns Markdown; `format: 'json'` (the default) returns the typed report object. Markdown section labels are English by default and Traditional Chinese for `zh-*` tags. `language` is a BCP 47 language tag. `imageFailurePolicy: 'error'` is the default and rejects on an image failure; `'omit'` preserves a typed failed-image entry with its error instead. The report retains source image IDs/provenance, requires every extracted paragraph and image to be accounted for, and validates all generated summary, section, and image-description fields against the requested language. Obvious English and Traditional Chinese script mismatches are rejected; this heuristic does not guarantee fluency or accurate content, and other languages are best-effort. `onToken(text, phase)` reports chunks from `image`, `section`, `summary`, or `conclusion` generation. Cancellation rejects the operation without returning a partial report; streamed chunks already delivered to `onToken` remain available to the caller. Truncated generations fail with `NekoError` code `INCOMPLETE_GENERATION`, and detected language mismatches use `LANGUAGE_MISMATCH`; neither is automatically retried.

`contextWindowTokens` is checked against the pinned model configuration; the default is a conservative 4096-token working window, not a claim about practical maximum context. Input and output budgets must fit together. Reports split paragraph text using the actual tokenizer and Unicode-preserving boundaries, then hierarchically reduce intermediate evidence when it cannot fit the summary budget. Schema validation retains every paragraph/image source ID, but neither those references nor successful reduction guarantees faithful meaning: summaries can omit facts and conclusions can contradict their source.

The `omit` policy is limited to image-inference failures. Exceptions thrown by the caller's `onToken` callback (even `throw undefined`) and `AbortSignal` cancellation still reject the report.

`neko.cache.model.prefetch/status/clear` respectively download, inspect, or remove only this pinned model's files; each returns a promise, so await the status call. `neko.cache.engine.status/release` reports or releases the reusable live engine. The default `cache: { engine: true, engineTtlMs: 1_800_000 }` reuses it until 30 minutes idle; disable with `cache: { engine: false }` or configure the TTL. `neko.backend.current()` reports the selected backend; `neko.backend.detect(device?)` checks model/runtime compatibility and, for browser WebGPU, adapter availability. Its `supported` flag does not prove that a native provider/driver is installed or a session works. Only one `Neko` instance may own the process-global Transformers runtime hooks at a time (`RUNTIME_BUSY`). Calls are serialized per instance; `dispose()` aborts queued work, waits for current work, releases the engine, and restores those shared hooks. Failures are `NekoError` values with `stage`, `code`, and `cause` when available.

`npm run smoke:browser` drives the static demo through image, text-only, and page-report inference. `--offline-reload` repeats the selected path using the persistent browser cache while blocking model-network requests. For cache reuse, `NEKO_MODEL_CACHE` must be the SDK cache **root** for `test:package`, but the specific **revision directory** for `smoke:browser` (directly containing `tokenizer.json` and `onnx/`). These long real-model runs are explicit rather than part of default CI. Using the macOS default cache:

```sh
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js" npm run test:package
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0" npm run smoke:browser -- webgpu 16 --headed --offline-reload
```

This headed smoke requires an available WebGPU adapter; headed mode alone does not guarantee one. The runner's local static server remains necessary. `--offline-reload` blocks model-download traffic only, not all website networking or the need for that server.

## Extraction, images, and reports

```js
import { extractPage, loadImage, renderMarkdown, validateStructuredReport } from 'neko.js';

const page = await extractPage('https://example.test/article', {
  maxHtmlBytes: 2 * 1024 * 1024,
  maxImages: 20,
  timeoutMs: 10_000,
  validateDestination: (url) => { /* apply the application's outbound-network policy */ },
});
const prepared = await loadImage(page.images[0]);
// prepared.data is normalized PNG bytes; prepared.imageId preserves provenance.
```

`extractPage(input, options?)` accepts an HTTP(S) URL or an HTML string; set `baseUrl` to resolve relative URLs in HTML. Defaults limit HTML to 2 MiB, images to 20 unique URLs, image downloads to 10 MiB, and each request to 10 seconds. Parsing is inert: it extracts semantic text and image sources without running scripts or fetching linked resources. It discovers `img`, `picture/source[srcset]`, inline styles, `background` attributes, and `og:image`, preserving source metadata.

Remote URL fetching creates SSRF risk when callers accept untrusted URLs. Supply `validateDestination` to check every destination/redirect and also apply application-level outbound network controls; this SDK cannot determine which private or internal destinations are safe for a particular deployment.

`loadImage(image, options?)` decodes an extracted image using the Node native decoder or browser bitmap/canvas path. It checks raster bytes/MIME, caps input and decoded dimensions, applies orientation, scales to at most 1280×1280 without enlarging, and emits PNG bytes. SVG and mismatched/invalid content are rejected. It preprocesses only; `Neko.describe()` performs the model inference. Browser remote image requests remain subject to CORS.

`renderMarkdown(report)` renders a `StructuredReport`, escapes untrusted text, and only creates provenance links for HTTP(S) URLs. `validateStructuredReport(report, page)` validates field types, requested-language checks, complete paragraph coverage, image count/identity/status, failure policy, and provenance. `Neko.describe()` connects extraction, image inference, summaries, and validation into the end-to-end report workflow.

## Cache and backend notes

The model manifest fixes the Hugging Face revision and SHA-256/size of required files. Every cache hit is verified before use; mismatches fail instead of silently becoming misses. `neko.cache.model.prefetch/status/clear` operate on only these pinned files. Browser Cache Storage remains subject to browser user actions and eviction.

`device: 'cpu'` and `device: 'webgpu'` are explicit choices; there is no automatic provider fallback or cross-platform parity guarantee. A packed-consumer run verified Node native CPU text, path-image, and HTML-report inference. In observed pinned model/runtime runs, saturated red/blue regions were described as pink or pinkish-red. Do not treat generated image descriptions as reliable ground truth; this observation does not isolate the model itself or establish Ollama parity. `backend.detect()` reports model/runtime compatibility; browser WebGPU also probes for an adapter with `shader-f16`, but Node does not probe native provider/driver availability until actual inference. Browser CPU/WASM is unsupported because ONNX Runtime Web lacks `GatherBlockQuantized(1)`; Neko reports this rather than silently falling back. Session provider/configuration data does not prove that every operator ran on a GPU.

## Security and privacy

Page HTML and generated model text are untrusted data. Parsing is inert, and the extraction helper does not load page scripts or subresources. Markdown output is escaped but must still be rendered with a safe Markdown renderer. Local model inference does not prevent the model from generating harmful or incorrect content. See [SECURITY.md](../../SECURITY.md) for the threat model and reporting process.
