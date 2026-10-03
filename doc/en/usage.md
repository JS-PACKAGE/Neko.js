# Usage

## Current scope

Neko.js is a local multimodal SDK for Node.js and supported WebGPU browsers. `createNeko()` lazily owns the pinned Qwen model and exposes text/image inference, website-to-structured-report generation, backend status, and explicit model/engine cache controls. A report runs inert HTML extraction and real image inference for discovered images; it is not a crawler, and remote URL fetching must be protected by the calling application's network policy. Model output may be inaccurate; never use it for security or authorization decisions.

The model is pinned to `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`, with Q4 embeddings/decoder and an FP16 vision encoder. Initial use downloads approximately 871 MB. Each cached asset is verified against pinned size and SHA-256 values before use. Runtime inference bundles Transformers.js 4.2.0; Node uses ONNX Runtime 1.30.0.

## Install and build

- Requires Node.js 22 or newer; the package is ESM and is not published to npm.
- For Git repository use, clone the repository, then run `npm ci` and `npm run build`. This explicit local build also works with npm 11's approval of lifecycle scripts; avoid installing an unbuilt Git dependency when its `prepare` script has not been approved.
- Build the archive locally from the current working tree with `npm pack`; it includes `dist/` even though generated output is git-ignored. Use the filename printed by `npm pack`, then install it in another project with `npm install /absolute/path/to/<printed-filename>.tgz`. The archive name/version derives from the current `package.json`, not this guide. This project is not published to npm. Node bundles Transformers.js but keeps pinned `onnxruntime-node`, `sharp`, and `parse5` as direct runtime dependencies. If npm 11 blocks native postinstall scripts, approve only `onnxruntime-node@1.30.0` and `sharp@0.35.4`; do not grant blanket script approval.
- `npm test`, `npm run typecheck`, and `npm run lint` run the Node tests, typecheck, and ESLint. `npm run test:package:artifact` exercises the actual packed artifact in an isolated consumer, including public imports and TypeScript declarations. `npm run test:package` additionally runs text, image, and report inference using the packed consumer and may download about 871 MB on a cold cache.
- `npm run test:browser` runs the browser contract tests. `npm run smoke:browser` is an explicit real-model browser UI run and may download about 871 MB; use a secure HTTP(S) origin for the static demo.

Browser CPU/WASM is unsupported for this model because ONNX Runtime Web lacks `GatherBlockQuantized(1)`; creation reports an unsupported-backend error and never falls back. Node and browser providers are selected explicitly and must be validated separately.

### First-use Node cache, then offline inference

Run this ESM example in the installed consumer (`node first-use.mjs`). Choose an absolute cache root owned by your user, not a symlink or filesystem root. The first instance needs model-network access; prefetch verifies the selected profile without creating inference sessions. Keep the entire model-ID/revision directory structure if moving the cache to an offline machine.

```js
import { createNeko } from 'neko.js';
import { resolve } from 'node:path';

const cacheDir = resolve('./neko-model-cache');
const setup = await createNeko({ device: 'cpu', cacheDir, modelProfile: 'default' });
try {
  await setup.cache.model.prefetch(); // Online once: about 871 MB for default.
} finally {
  await setup.dispose(); // Release the inline runtime owner before creating another.
}

const offline = await createNeko({
  device: 'cpu', cacheDir, modelProfile: 'default', localFilesOnly: true,
});
try {
  const result = await offline.infer({ prompt: 'Write one short greeting.', maxNewTokens: 32 });
  console.log(result.text);
} finally {
  await offline.dispose();
}
```

Repeat only the second instance in a fresh process without model-network access. Missing or corrupt assets fail closed; offline mode does not install Node native dependencies or make remote page/image inputs available. For local images, explicitly approve their canonical paths with `policy.localFiles`, as below. `all-q4` requires its own selected-profile prefetch. Offline operation, valid JSON and quoted provenance are not quality or factual-accuracy promises.

## Historical prototype measurements

Measurements used Node.js 22.23.3, Transformers.js 4.2.0, native ONNX Runtime 1.30.0, and the pinned Qwen revision. The hybrid provider profile recorded 128 WebGPU events for embedding, 536 WebGPU + 50 CPU for vision, and 169,472 WebGPU + 12,288 CPU for decoder; this is not an all-GPU execution claim.

- Populated verified cache: model load 6,414.542 ms, first non-empty decoded output 1,487.161 ms, generation 9,353.701 ms, inference total 9,486.618 ms; zero network requests.
- Cold empty cache: 13 files / 871,364,778 bytes downloaded; outer load 37,118.820 ms, inference load 37,114.206 ms, first output 1,228.452 ms, generation 7,728.604 ms, inference total 7,796.944 ms.
- Fresh offline process using the same cache: outer load 4,655.131 ms, inference load 4,652.725 ms, first output 2,447.610 ms, generation 9,874.194 ms, inference total 9,997.882 ms; zero network requests.

`loadMs` includes cache verification and processor/model loading. `InferenceResult.timings.firstTokenMs` measures the first model-token callback from generation start; it is not the first non-empty decoded text chunk or kernel latency. The prototype measurements below separately recorded the first non-empty decoded output. Process RSS was about 2.147 GB after load, 1.966 GB during inference, and 0.731 GB after disposal. GPU memory is unknown. These are measured runs, not cross-platform performance guarantees.


## SDK usage

```js
import { createNeko } from 'neko.js';
import { realpath } from 'node:fs/promises';

const photo = await realpath('./photo.png');
const neko = await createNeko({ device: 'cpu', policy: { localFiles: (path) => path === photo } });
try {
  // Image input is optional: without it, this is text-only inference.
  const answer = await neko.infer({
    image: './photo.png',
    prompt: 'Describe the visible scene.',
    maxNewTokens: 128,
    onToken: (text) => process.stdout.write(text),
  });
  console.log(answer.usage, answer.finishReason);

  const report = await neko.describe('<p>The sky is blue.</p>', {
    language: 'en',
    format: 'json',
    imageFailurePolicy: 'omit',
    maxNewTokens: 256,
  });
  console.log(report.page.summary, report.images);
  console.log(await neko.cache.model.status(), await neko.cache.engine.status());
} finally {
  await neko.dispose();
}
```

`createNeko(options?)` configures the backend and verified cache without loading the model. `modelProfile: 'default'` loads q4 embeddings/decoder and fp16 vision; `'all-q4'` selects the separately pinned q4 vision assets. Loaded session metadata records the actual configuration, not just the requested profile. `cache.model.prefetch()` verifies files without creating sessions; inference loads lazily. `load()` loads sessions, `warmup()` also exercises text and vision, and `runtimeStatus()` reports their readiness. The default device is `webgpu`, with no automatic fallback. Node `cacheDir` defaults to macOS `~/Library/Caches/neko.js`, Windows `%LOCALAPPDATA%/neko.js`, or Linux `$XDG_CACHE_HOME/neko.js` / `~/.cache/neko.js`; browsers use Cache Storage. `localFilesOnly: true` forbids uncached model downloads and remote page/image requests; trusted package-local worker/runtime bootstrap remains permitted.

`infer({ prompt?, messages?, image?, generation?, maxNewTokens?, contextWindowTokens?, signal?, onToken? })` supports a nonempty text prompt or ordered `system`/`user`/`assistant` messages with text and image content. Multiple images are processed jointly in one generation, preserving their individual dimensions and content hashes. Inputs include Node paths/file URLs, HTTP(S) image URLs, browser `Blob`/`File` and local `blob:` URLs, supported raster data URLs, or structurally decoded images. Instance policy approval is required for local files and remote page/image URLs; browser CORS still applies. `maxNewTokens` defaults to 128. Results contain text, finish reason, usage, model/profile, observed sessions, execution, timings, and memory (`null` when unavailable). `onToken` receives decoded chunks; `firstTokenMs` measures the first model-token callback.

`generation` supports `sampling`, `temperature`, `topK`, `topP`, `repetitionPenalty`, `noRepeatNgramSize`, `stop`, and `stopTokenIds`. Temperature/top-K/top-P require `sampling: true`; greedy decoding remains the default. String stops work across decoded chunk boundaries without leaking the stop text. `inferStructured({ ...inferenceOptions, schema })` includes the validated JSON Schema in prompt/context planning and returns parsed `value` alongside inference metadata. Cheap prompt/message/generation/budget validation and schema compilation happen before model acquisition, including worker execution; invalid requests do not need a populated model cache.

Structured generation stops deterministically at the boundary of the first complete JSON value, rather than extracting a valid-looking substring after generation. Boundary detection uses raw generated token IDs; any extra content emitted inside the completing token is still validated and rejected, not sliced off. Numeric roots wait for whitespace or EOS to establish their boundary. It rejects malformed prefixes, incomplete JSON and schema violations; it does not repair output or retry. Full supported Draft-07 runtime validation remains fail-closed (`SCHEMA_INVALID` for invalid/unsupported schemas; `STRUCTURED_OUTPUT` for invalid generated JSON/schema output). The result's `structured.mode` is `'json-boundary-runtime-validation'` and `structured.dialect` is `'draft-07'`. This is **not** schema-grammar-constrained decoding, semantic validation or a promise that generation succeeds.

### Exact inference planning

```ts
import type { InferencePlan } from 'neko.js';

const request = { prompt: 'Write one short greeting.', maxNewTokens: 32 };
const plan: InferencePlan = await neko.planInference(request);
console.log(plan.inputTokens, plan.maxNewTokens, plan.contextLimit,
  plan.availableOutputTokens, plan.fits);
if (plan.fits) console.log((await neko.infer(request)).text);
```

`planInference(options)` uses the same actual chat-template/tokenizer/image preprocessing as inference, including image-token expansion. Optional `schema` includes the structured-output instruction in the count. It returns model/image observations and execution metadata, but generates no tokens. `availableOutputTokens` is `max(0, contextLimit - inputTokens)`; `fits` compares the requested output budget with that capacity. A valid cold planning request **loads the model/processor and may download the selected assets**; this is not a lightweight tokenizer-only API. Planning an oversized context returns `fits: false`, whereas inference rejects it. A plan does not reserve queue capacity or guarantee later generation success.

```ts
const schema = {
  type: 'object',
  properties: { greeting: { type: 'string' } },
  required: ['greeting'],
  additionalProperties: false,
};
const structuredRequest = {
  prompt: 'Return a short greeting in the requested JSON shape.',
  maxNewTokens: 64, schema,
};
const structuredPlan = await neko.planInference(structuredRequest);
if (structuredPlan.fits) {
  const result = await neko.inferStructured(structuredRequest);
  console.log(result.value, result.structured.mode, result.structured.dialect);
}
```

`describe(urlOrHtmlOrPage, options?)` accepts an HTTP(S) URL, inert HTML, or an owned copy of a validated `Page`. `sources` selects paragraph/image IDs and optionally asynchronous `paragraph`/`image` predicates; IDs must be known and unique, and predicates receive readonly owned source records. Selection and extraction happen before model loading. Every selected source must be accounted for; deliberately excluded sources are not required. Image preprocessing is reused for its staged inference. Each report stage produces evidence-linked claims, then summaries/conclusions are reduced hierarchically as needed. `maxNewTokens` defaults to 256 per generation. `format: 'json'` returns the typed report; `'markdown'` returns escaped Markdown. `language` is BCP 47; English and Traditional Chinese script checks are heuristic, not fluency/factual guarantees. `onToken(text, phase)` identifies `image`, `section`, `summary`, or `conclusion`. `imageFailurePolicy: 'error'` rejects image failures; `'omit'` records a typed failed image but never suppresses policy violations, callback exceptions (including `throw undefined`), cancellation, or budget errors. Invalid/truncated generated output and language mismatches fail without automatic retries.

`contextWindowTokens` is checked against the pinned model configuration; the default is a conservative 4096-token working window, not a claim about practical maximum context. Input and output budgets must fit together. Reports split paragraph text using the actual tokenizer and Unicode-preserving boundaries. Each section request handles at most four quote spans and asks for one to four evidence-linked claims, rather than collapsing an arbitrarily large source into one claim. `sourceFacts` retains the full selected paragraph text as exact contiguous quotes with UTF-16 offsets independently of generated summaries. The ledger is retained source text, **not** extracted or verified real-world facts. Summaries/conclusions use retained source evidence when it fits; otherwise generated claims are reduced hierarchically. Neither references nor successful reduction proves faithful meaning.

Generation schemas are phase-specific: sections and intermediate reduction request one to four `{ text, evidenceIds }` claims; image descriptions, `page.summary` and `conclusion` request one concise evidence-linked passage that may contain multiple supported facts. This bounds output shape, not factuality or completeness. Checkpoints bind request hashes to the actual phase schema.

`budget: { maxTotalTokens, maxDurationMs }` bounds aggregate input/output usage and elapsed request time, including queue wait, load, extraction, preprocessing, generation, and supported asynchronous source predicates, resource approvals, `onEvent`, and `onCheckpoint`. A deadline cancels pending user awaits; it does not terminate caller-owned side effects or forcibly preempt a native ORT/OS call. `onEvent` reports stage transitions; `onCheckpoint` receives cloneable saved state. `resume: checkpoint` reuses only compatible completed stages, preserves spent budget, and verifies source/model/settings/checksum and image content versions. After a report checkpoint exists, failures expose its final accounting through `ReportError.checkpoint`; earlier extraction/selection/load failures may be plain `NekoError`. Checkpoints contain selected source text/metadata, not image pixels; treat persisted data as potentially sensitive.

All cache/backend/status methods return promises. `cache.model.prefetch/status/clear` affect only the selected pinned profile; `cache.engine.status/release` inspect or release the live engine. `cache: { engine: true, engineTtlMs: 1_800_000 }` reuses it until 30 minutes idle. `backend.detect()` checks compatibility, not successful native driver/session creation. `execution: 'inline'` is the default and permits one process-global runtime owner; `'worker'` creates a real Node thread or browser module worker with independent ownership. Calls are bounded FIFO per instance (`queue: { maxPending: 8 }`), report queue wait, and reject excess admission with `QUEUE_FULL`; `queueStatus()` exposes current state. Worker callbacks retain ordering and typed errors/checkpoints. `dispose()` cancels queued work, awaits safe active-work cleanup, releases resources, and restores hooks.

`policy.network(url, kind)` approves normally (`undefined` or `true`) or denies by throwing/returning `false`; `kind` is `model`, `runtime`, `worker`, `page`, or `image`. `policy.localFiles(canonicalPath)` explicitly approves Node input-file access. Defaults allow only exact pinned model transfers and package bootstrap assets; arbitrary page/image destinations and local files are denied. Checks happen before each visible redirect hop, and per-call `validateDestination` adds restrictions without replacing instance policy. The SDK cannot identify deployment-specific private/SSRF-safe destinations; application approval must enforce those boundaries.

The factory captures known policy hook references, including class prototype methods, and binds their original receiver. Defined nonfunction hooks and invalid policy records fail preflight. The owned hook record is frozen, not the caller's policy object.

`npm run smoke:browser` drives the static demo through image, text-only, and page-report inference. `--offline-reload` repeats the selected path using the persistent browser cache while blocking model-network requests. With `NEKO_MODEL_CACHE`, the smoke runner verifies the selected pinned profile through the local mirror and seeds browser Cache Storage before running the UI in local-only mode. For cache reuse, `NEKO_MODEL_CACHE` must be the SDK cache **root** for `test:package`, but the specific **revision directory** for `smoke:browser` (directly containing `tokenizer.json` and `onnx/`). These long real-model runs are explicit rather than part of default CI. Using the macOS default cache:

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

`renderMarkdown(report)` escapes untrusted text and links only HTTP(S) provenance. `await validateStructuredReport(report, selectedPage?)` audits the persisted selected source snapshot and SHA-256 versions, exact quoted offsets/content, claim spans and evidence references, complete selected-source coverage, image status/provenance, language, loaded model/session identity, and aggregate metadata. An optional matching selected `Page` adds an external-source comparison; no fetch is needed to audit saved JSON. It detects inconsistent references/accounting, not whether a model claim is entailed by its source or image pixels. Hashes are not source authentication.

### Versioned reports and checkpoints

Reports use `schemaVersion: 2` and carry `sourceFacts` plus `integrity: { algorithm: 'sha256', checksum }`. `metadata.coverage` records selected paragraph IDs/character count, retained quote/character count, model- and summary-cited fact IDs, `conclusionBasis` (`'retained-source'` or `'reduced-generated-claims'`), and `semanticRetention: 'not-measured'`. Source coverage is structural accounting, not a semantic recall score; generated claims are not fact-checked.

`renderMarkdown(report)` includes the retained quote ledger and coverage as well as generated sections, so exact source text remains inspectable even when a model summary omits it. Escaping does not make quoted text trustworthy or private.

```ts
import {
  serializeStructuredReport, parseStructuredReport,
  serializeReportCheckpoint, parseReportCheckpoint,
} from 'neko.js/report';
import type { StructuredReport, ReportCheckpoint } from 'neko.js';

const savedReport = await serializeStructuredReport(report);
const restored: StructuredReport = await parseStructuredReport(savedReport);
console.log(restored.schemaVersion, restored.metadata.coverage);

// In describe's onCheckpoint callback, persist the returned string securely.
async function saveCheckpoint(checkpoint: ReportCheckpoint): Promise<string> {
  return serializeReportCheckpoint(checkpoint);
}
async function resumeSaved(saved: string): Promise<StructuredReport> {
  const checkpoint = await parseReportCheckpoint(saved);
  return neko.describe('<p>The sky is blue.</p>', {
    language: 'en', format: 'json', imageFailurePolicy: 'omit',
    maxNewTokens: 256, resume: checkpoint,
  });
}
```

Serialization and parsing validate before accepting data; optionally pass a matching selected `Page` as the second report-helper argument. `validateReportCheckpoint(value)` also validates an in-memory checkpoint. Checkpoints use `version: 2` and `plan: 'evidence-first-v2'`, persist the quote ledger, and resume only with matching input/model/settings. **Breaking persistence policy:** unversioned/older/future reports and checkpoints are rejected, not migrated or interpreted through aliases. Report rejection is a `TypeError`; invalid checkpoint versions use `CHECKPOINT_INVALID`. Recreate reports/checkpoints from the original inputs under the current contract. Checksums detect inconsistent/tampered saved content but do not authenticate its author. Saved source quotes, metadata and generated text may be sensitive.

Checkpoint `sectionPlan` records ordered groups of one to four source-fact IDs covering the ledger exactly. Validation rejects section citations into other groups even if a checksum is recomputed; resume also verifies the deterministic plan and request hashes. Persist SDK-issued checkpoints through the helpers rather than constructing saved state manually.

## Cache and backend notes

The model manifest fixes the Hugging Face revision and SHA-256/size of required files. Every cache hit is verified before use; mismatches fail instead of silently becoming misses. `neko.cache.model.prefetch/status/clear` operate on only these pinned files. Browser Cache Storage remains subject to browser user actions and eviction.

### Browser model sources

#### First-use browser workflow

1. Complete the online Node prefetch above in the checkout or installed consumer. It produces the same verified files the mirror needs. Build the checkout with `npm run build`.
2. For the existing demo, install the browser harness with `npx playwright install chromium`, then run the existing runner; it creates its own loopback static server and exact-origin CORS mirror, verifies/seeds browser Cache Storage, and runs the UI:

   ```sh
   NEKO_MODEL_CACHE="/absolute/path/to/neko-model-cache/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0" npm run smoke:browser -- webgpu 256 --headed --offline-reload
   ```

   The environment variable is the **revision directory**, not the cache root. `256` is the explicit per-generation output budget, not a guarantee that every report completes. The runner closes its temporary server/profile afterward; it is a real-inference check, not a persistent development server. Use a WebGPU-capable browser/driver; `--headed` does not create an adapter. The shipped demo itself has no mirror URL field, so merely opening it with an empty cache does not configure a first-online mirror.
3. For your own application, serve the ESM bundles and all `dist/browser/assets/` files from HTTPS or trusted loopback HTTP, **not** `file:`. Keep `neko.js`, `worker.js` and `assets/` in their emitted relative layout. Serve `.js`/`.mjs` as JavaScript and `.wasm` as `application/wasm`. WebGPU execution still needs the ONNX WASM runtime assets; it is not CPU fallback. CSP must permit the SDK's fetched/blob runtime modules and module workers if used.
4. Prefer a same-origin `/models/` reverse proxy to the loopback mirror. Preserve the complete pinned path unchanged and forward upstream with `Host: 127.0.0.1:8787` (the helper rejects arbitrary Host headers). Use the exact application origin in `--allow-origin` when forwarding an `Origin` header. Never expose arbitrary cache files or a generic URL proxy. For example, in an existing nginx server listening at `http://127.0.0.1:4173`:

   ```nginx
   location /dist/browser/ {
     root /absolute/path/to/Neko.js;
     types { application/javascript js mjs; application/wasm wasm; }
   }
   location /models/ {
     proxy_pass http://127.0.0.1:8787;
     proxy_set_header Host 127.0.0.1:8787;
   }
   ```

   Start the mirror with the command below, substituting your actual cache **root**. Your page imports `/dist/browser/neko.js` and sets `modelSource.baseUrl` to `new URL('/models/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0/', location.href).href`. Same-origin model/runtime assets need no cross-origin CORS grant. If you instead call the separate `8787` mirror directly, the page origin must exactly match `--allow-origin` (scheme, host and port); do not use wildcard CORS. Remote image/page CORS remains independent of model CORS.

   ```js
   import { createNeko } from '/dist/browser/neko.js';

   const neko = await createNeko({
     device: 'webgpu',
     modelSource: {
       baseUrl: new URL('/models/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0/', location.href).href,
     },
   });
   try {
     await neko.cache.model.prefetch();
     document.querySelector('#output').textContent =
       (await neko.infer({ prompt: 'Write one short greeting.', maxNewTokens: 32 })).text;
   } finally {
     await neko.dispose();
   }
   ```

After initial caching, dispose the online instance and create a new same-profile instance with `localFilesOnly: true`. Keep serving package/runtime assets even when model traffic is blocked: model-cache offline use is not a fully offline website. Cache Storage can be evicted, and different origins/profiles do not implicitly share it. Browser CPU/WASM is rejected with no automatic fallback; this workflow does not promise accurate output.

Browsers conceal cross-origin redirect destinations (`opaqueredirect`); a Hugging Face redirect therefore fails closed with `POLICY_DENIED`, rather than silently following an unapproved hop. For first-online downloads use an explicit `modelSource: { baseUrl }` mirror/broker serving the selected profile's exact pinned relative paths. Every body retains manifest size/SHA-256 verification and the original canonical Hugging Face Cache Storage key. The base must be absolute HTTP(S), without credentials/query/fragment. Visible redirects still require authorization; CORS and secure-context requirements remain.

The factory captures and freezes a plain normalized `baseUrl` record, including values from class getters, before worker serialization. Cross-origin mirror remapping and visible redirect hops strip `Authorization`, `Cookie`, and `Proxy-Authorization`; stripped credentials never return on a later same-origin hop. Nonsensitive headers and same-origin requests are preserved.

From a built checkout with an existing verified Node cache, the loopback-only helper serves no arbitrary filesystem paths:

```sh
npm run build
node scripts/serve-model-mirror.mjs --cache-dir="$HOME/Library/Caches/neko.js" --model-profile=default --port=8787 --allow-origin=http://127.0.0.1:4173
```

In a browser served from that exact approved origin:

```js
const neko = await createNeko({
  device: 'webgpu',
  modelProfile: 'default',
  modelSource: { baseUrl: 'http://127.0.0.1:8787/models/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0/' },
});
await neko.cache.model.prefetch();
```

Alternatively reverse-proxy that directory under the application origin. Match `--model-profile=all-q4` with the same factory profile when using all-q4. The helper verifies the selected cache at startup and each requested asset before serving; it never fetches Hugging Face or follows browser redirects. After caching, a new `localFilesOnly: true` instance reuses verified files without model-network access.

### Backend compatibility

`device: 'cpu'` and `device: 'webgpu'` are explicit choices; there is no automatic provider fallback or cross-platform parity guarantee. `backend.detect()` reports model/runtime compatibility; browser WebGPU also probes for an adapter with `shader-f16`, but Node does not probe native provider/driver availability until actual inference. Browser CPU/WASM is unsupported because ONNX Runtime Web lacks `GatherBlockQuantized(1)`. Requested/session provider configuration is not proof of successful inference or that every operator runs on the GPU. Historical hybrid operator profiling above actually observed CPU operations alongside WebGPU.

Evidence snapshot (2026-10-03): macOS (Darwin 27), arm64, with fresh CPU smoke runs on Node 22.23.3 and Node 26.7.0. Fresh outcomes below verify only their recorded runtime/backend/profile/input, not the other rows or every Node `>=22` release.

| Runtime/platform | API/contract status | Historical real-inference evidence | Fresh verification of current changes |
| --- | --- | --- | --- |
| Node on macOS/arm64, CPU | Native CPU path; Node `>=22` required by package, build targets Node 22 | Node 22.23.3 cold/download/offline prototype and packed-consumer text/path-image/HTML-report runs | Passed Node 22.23.3 and 26.7.0 offline CPU/default worker inference, planning, inline/worker preflight, report persistence and 4-stage checkpoint resume |
| Node on macOS/arm64, WebGPU | Explicit native provider selection; driver/session must work | Hybrid provider profiling recorded GPU **and CPU** operators | Not established for this change |
| Node on Windows or Linux, CPU/WebGPU | Platform-dependent native dependency/provider paths; cache locations implemented | No platform-specific real-inference evidence recorded here | Unverified; no Windows/Linux or GPU parity claim |
| Chromium with WebGPU | Secure context, usable adapter and `shader-f16`, model cache/mirror and runtime assets required | Browser demo inference and model-network-blocked reload recorded in prior releases | Passed on macOS/arm64, full Chromium 153.0.0.0 headless with `--enable-unsafe-webgpu`: exact text/image/schema plans, structured generation, image/report inference, persistence and actual demo worker text; not every Chromium/OS combination |
| Firefox/Safari/other browsers | Must independently satisfy the same runtime requirements | No browser-specific real-inference evidence recorded here | Unverified; API presence is insufficient |
| Any browser, CPU/WASM | Unsupported; rejected with no fallback | Missing `GatherBlockQuantized(1)` observed | Remains an unsupported configuration |

The fresh Node 22.23.3 and 26.7.0 CPU/default worker runs matched planned token counts for text/chat/schema, streamed the exact structured response, returned `fits: false` for overflow, and verified report/checkpoint serialize/parse, worker execution identity and four resumed stages. Report sections contained all five test facts (1987, an 18-meter-tall oak, no fountain, Plot A 12–19, Plot B 24–31), with 5 exact quotes retaining 134/134 UTF-16 characters. **The generated summary omitted the opening year and used `q1` rather than the no-fountain quote `q3`; the red image fixture was described as pink.** The quote ledger is lossless for selected text; generated summaries and citations are not semantically verified (`semanticRetention: 'not-measured'`).

The final packed consumer also passed on Node 22.23.3: it installed the locally generated tarball, typechecked the public package exports, and exercised offline native CPU text, path-image, exact inference planning, structured generation, HTML/image reports, validated persistence, rejection of edited source quotes and checkpoint resume. Browser inline and worker API smokes both passed separately, including report checksum validation after worker execution identity was recorded; the demo worker text path was visually checked again.

The fresh browser run seeded a new persistent profile with 13 locally verified pinned SHA-256 assets; it did **not** verify a fresh Internet download. Exact planned counts matched inference (text 25, image 90, plus a schema-bearing request); invalid cold requests failed preprocessing before model acquisition. Structured output was `{ answer: 7 }`; the report retained all three test facts (1987, an 18-meter-tall oak, no fountain) in its key points/overview and 3 exact quotes covering 75/75 UTF-16 characters, then passed validation and persistence round-trip. The actual demo's worker text path returned `7` and was visually checked.

This was a headless test configuration with an unsafe-WebGPU enablement flag, not ordinary unflagged browser support or hardware-GPU/operator proof; software/headless adapters may differ. The default headless-shell adapter lacked `shader-f16` and was correctly rejected before model acquisition. Chromium/Firefox/WebKit contract tests also passed (15/15), but the latter two results are **not** real-model inference evidence. Successful fixture output is not a general report-quality gate pass.

Contract tests, typechecks and adapter preflight are not real-inference evidence. Historical observations of the pinned pipeline include saturated red/blue regions described as pink or pinkish-red, omitted report facts and contradictory conclusions. These are not an isolated model diagnosis or an Ollama comparison. No platform row promises reliable color recognition, semantic retention, factuality or performance.

## Security and privacy

Page HTML and generated model text are untrusted data. Parsing is inert, and the extraction helper does not load page scripts or subresources. Markdown output is escaped but must still be rendered with a safe Markdown renderer. Local model inference does not prevent the model from generating harmful or incorrect content. See [SECURITY.md](../../SECURITY.md) for the threat model and reporting process.
