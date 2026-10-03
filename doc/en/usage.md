# Usage

## Current scope

Neko.js is a local multimodal SDK for Node.js and supported WebGPU browsers. `createNeko()` lazily owns a registered pinned Qwen model and exposes text/image inference, structured output, streams, sessions, document QA/reports, planning, diagnostics and cache controls. Generated reports perform real image inference; extractive reports do not. Extraction is inert and is not a crawler. Application network policy must protect remote inputs. Model output may be inaccurate; never use it for security or authorization decisions.

The model is pinned to `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`, with Q4 embeddings/decoder and an FP16 vision encoder. Initial use downloads approximately 871 MB. Each cached asset is verified against pinned size and SHA-256 values before use. Runtime inference bundles Transformers.js 4.2.0; Node uses ONNX Runtime 1.30.0.

Select either registered model with `createNeko({ model: 'onnx-community/Qwen3.5-2B-ONNX-OPT', modelProfile: 'all-q4', device: 'cpu' })`. The 2B revision is `2ea7886f48b926aca97de8b0e041ffca7e3ebaa9`. Both models offer only the fixed `default` and `all-q4` profiles. `MODEL_REGISTRY` and `getRegisteredModelProfile(profile?, model?)` expose their immutable identities and asset manifests. Arbitrary model IDs/revisions are unsupported; `modelSource` changes only the approved asset mirror, never model identity. The download size above belongs to the default 0.8B profile.

## Install and build

- Requires Node.js 22.13 or newer; the package is ESM and is not published to npm.
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

Default `inferStructured` uses a real tokenizer-aware JSON grammar and returns `structured.mode: 'tokenizer-constrained-runtime-validation'`. Its supported subset is primitive types, primitive `enum`/`const`, string `minLength`/`maxLength`, closed objects with `properties`, `required` and `additionalProperties: false`, and homogeneous arrays with `minItems`, `maxItems` and `uniqueItems`. Numeric bounds, composition and other unsupported grammar keywords fail with `SCHEMA_UNSUPPORTED` before engine acquisition; invalid schemas fail with `SCHEMA_INVALID`. There is no silent fallback.

Explicit `structuredMode: 'validation-only'` opts into the existing full supported Draft-07 runtime validator and JSON-boundary decoding, returning `'json-boundary-runtime-validation'`. Both modes report dialect `'draft-07'`, validate the full result, reject malformed/incomplete JSON and schema violations (`STRUCTURED_OUTPUT`), and do not repair output or guarantee factuality. Boundary handling never slices a valid-looking substring from invalid output. `inferStructured<const S>` returns `SchemaValue<S>`; retain literal schema types with `as const` for precise TypeScript inference.

### Exact inference planning

```ts
import type { InferencePlan } from 'neko.js';

const request = { prompt: 'Write one short greeting.', maxNewTokens: 32 };
const plan: InferencePlan = await neko.planInference(request);
console.log(plan.inputTokens, plan.maxNewTokens, plan.contextLimit,
  plan.availableOutputTokens, plan.fits);
if (plan.fits) console.log((await neko.infer(request)).text);
```

`planInference(options)` uses the same rendered chat template, tokenizer and image-token expansion as inference, including optional `schema` and `structuredMode`. Text-only planning reads/downloads verified tokenizer/config/template assets but creates **no ONNX sessions**. Image planning preprocesses images and may acquire the engine. It generates no tokens. `availableOutputTokens` is `max(0, contextLimit - inputTokens)`; `fits` compares requested output with capacity. Oversized planning returns `fits: false`, whereas inference rejects it. Plans reserve neither queue capacity nor successful generation.

The `preprocessing` observations describe exact owned rendered-prompt tokenization reuse only, with `kvReuse: false`. Repeating an identical plan/inference request can reuse those tokens; appending conversation turns does not reuse a growing prefix or model KV state. Caller prompt text is preserved. Engine reuse and normalized image-pixel reuse are separate mechanisms.

```ts
const schema = {
  type: 'object',
  properties: { greeting: { type: 'string' } },
  required: ['greeting'],
  additionalProperties: false,
} as const;
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

### Streams and conversations

```js
for await (const event of neko.inferStream({
  prompt: 'Write a short greeting.', maxNewTokens: 32,
  maxBufferedEvents: 64, maxBufferedCharacters: 1_048_576,
})) {
  if (event.type === 'token') process.stdout.write(event.text);
  else console.log(event.result.usage);
}
const conversation = neko.session({
  system: 'Answer concisely.', contextPolicy: 'drop-oldest',
  maxHistoryMessages: 16, maxNewTokens: 64,
});
try {
  console.log(await conversation.plan('Hello.'));
  console.log((await conversation.send('Hello.')).text);
  console.log(conversation.history());
  const fork = await conversation.branch();
  try {
    await fork.import(await conversation.export());
    await fork.reset();
  } finally { await fork.dispose(); }
} finally { await conversation.dispose(); }
```

`inferStream` is a bounded `AsyncIterable` of decoded `token` events followed by one `result` with final usage. Slow-consumer overflow cancels generation with `STREAM_OVERFLOW`; breaking iteration/early return also cancels. Chunk counts are not token usage.

Session operations are serialized and transactional: only successful completed turns enter history. `plan` does not commit; failure/cancellation does not append partial history. Default `contextPolicy: 'error'` rejects overflow; `'drop-oldest'` removes whole oldest turns while preserving system. `maxHistoryMessages` includes system and is at most 128. History/branches are owned copies. `export()`/`import(snapshot)` use model-bound version-1 snapshots with portable raw pixels/raster data URLs (raster Blobs are embedded); external URL/path images are rejected on export without IO. Reset preserves system/defaults. Session disposal never disposes its host. Sessions do not provide KV/prefix reuse.

For an unsupported grammar constraint, explicitly opt in:

```ts
const bounded = await neko.inferStructured({
  prompt: 'Return a positive integer.', maxNewTokens: 32,
  schema: { type: 'integer', minimum: 1 } as const,
  structuredMode: 'validation-only',
});
console.log(bounded.value, bounded.structured.mode);
```

`describe(urlOrHtmlOrPage, options?)` accepts an HTTP(S) URL, inert HTML, or an owned copy of a validated `Page`. `sources` selects known, unique paragraph/image IDs and optional asynchronous readonly predicates. Selection/extraction precede model loading; every selected source must be accounted for. Default `mode: 'generated'` performs staged text/image inference and hierarchical reductions when needed. `maxNewTokens` defaults to 256 per generation. `format: 'json'` returns the typed report; `'markdown'` returns escaped Markdown. `language` is BCP 47; language/script checks are heuristic, not fluency/factual guarantees. `onToken(text, phase)` identifies `image`, `section`, `summary`, or `conclusion`. `imageFailurePolicy: 'error'` rejects image failures; `'omit'` records a typed failure but does not suppress policy violations, callback exceptions, cancellation or budget errors.

`contextWindowTokens` is checked against the pinned model configuration; the default is a conservative 4096-token working window, not a claim about practical maximum context. Input and output budgets must fit together. Reports split paragraph text using the actual tokenizer and Unicode-preserving boundaries. Each section request handles at most four quote spans and asks for one to four evidence-linked claims, rather than collapsing an arbitrarily large source into one claim. `sourceFacts` retains the full selected paragraph text as exact contiguous quotes with UTF-16 offsets independently of generated summaries. The ledger is retained source text, **not** extracted or verified real-world facts. Summaries/conclusions use retained source evidence when it fits; otherwise generated claims are reduced hierarchically. Neither references nor successful reduction proves faithful meaning.

Generation schemas are phase-specific: sections and intermediate reduction request one to four `{ text, evidenceIds }` claims; image descriptions, `page.summary` and `conclusion` request one concise evidence-linked passage that may contain multiple supported facts. This bounds output shape, not factuality or completeness. Checkpoints bind request hashes to the actual phase schema.

`budget: { maxTotalTokens, maxDurationMs }` bounds aggregate input/output usage and elapsed request time, including queue wait, load, extraction, preprocessing, generation, and supported asynchronous source predicates, resource approvals, `onEvent`, and `onCheckpoint`. A deadline cancels pending user awaits; it does not terminate caller-owned side effects or forcibly preempt a native ORT/OS call. `onEvent` reports stage transitions; `onCheckpoint` receives cloneable saved state. `resume: checkpoint` reuses only compatible completed stages, preserves spent budget, and verifies source/model/settings/checksum and image content versions. After a report checkpoint exists, failures expose its final accounting through `ReportError.checkpoint`; earlier extraction/selection/load failures may be plain `NekoError`. Checkpoints contain selected source text/metadata, not image pixels; treat persisted data as potentially sensitive.

### Structured, report and question streams

`inferStructuredStream(options)`, `describeStream(input, options?)`, `askStream(input, question, options?)`, and `askDocumentsStream(index, question, options?)` use the corresponding non-streaming request options plus `StreamBufferOptions`. All expose unvalidated `{ type: 'provisional', text }` deltas before a terminal `{ type: 'result', result, usage }`. Only the final result has completed schema/citation validation; provisional JSON is not an answer or evidence. Report deltas also carry `phase`, and reports emit `{ type: 'stage', event }` progress. With `format: 'markdown'`, the terminal report is rendered Markdown, not streamed Markdown fragments.

```ts
for await (const event of neko.inferStructuredStream({
  prompt: 'Return a greeting.', schema, maxNewTokens: 64,
  maxBufferedEvents: 64, maxBufferedCharacters: 1_048_576,
})) {
  if (event.type === 'result') console.log(event.result.value);
}
for await (const event of conversation.sendStream('Hello.')) {
  if (event.type === 'token') process.stdout.write(event.text);
}
```

Buffer defaults are 64 events and 1,048,576 UTF-16 characters; accepted limits are 1–10,000 events and 1–16,777,216 characters. Queued terminal results also count toward the bounds. Overflow fails with `STREAM_OVERFLOW` and cancels the operation; early `break`, iterator return, or cancellation does not yield a successful terminal result.

`session.sendStream(content, options?)` emits `token` and `result`, rather than `provisional`. History commits **only when iteration reaches normal exhaustion after the result**. Receiving the result then breaking still rolls back; use a complete `for await` loop. Failure, overflow and cancellation append neither the user turn nor partial assistant output. Session ownership remains held until exhaustion or cancellation.

### Tool selection and approved execution

```ts
import { defineTool, executeToolCalls, runToolLoop } from 'neko.js';

const tools = [defineTool({
  name: 'openingHours',
  description: 'Read the application-owned opening hours.',
  parameters: { type: 'object', properties: {}, additionalProperties: false } as const,
  result: { type: 'string' } as const,
})] as const;
const messages = [{ role: 'user' as const, content: 'What are the opening hours?' }];
const execution = {
  approve: (call: { name: string }) => call.name === 'openingHours',
  handlers: { openingHours: () => 'Monday 09:00–17:00' },
};
const selected = await neko.inferTools({ tools, messages, maxNewTokens: 256 });
const results = await executeToolCalls(tools, selected.toolCalls, execution);
const completed = await runToolLoop(neko, tools, {
  messages, ...execution, maxRounds: 4, maxNewTokens: 256,
});
console.log(results, completed.stopReason, completed.message);
```

`defineTool({ name, description?, parameters, result?, structuredMode? })` validates schemas and owns the definition. `neko.inferTools({ tools, messages, maxToolCalls?, ...inferenceOptions })` performs structured model selection and returns validated `toolCalls` and an assistant `message`; it executes no handlers. `maxToolCalls` defaults to 8 (0–64). Tool conversations use their separate `ToolConversationMessage` contract, including `role: 'tool'` results, not ordinary inference chat messages.

`executeToolCalls(tools, calls, { approve, handlers, signal? })` validates calls and processes approval/handlers sequentially. Only literal `true` from application approval authorizes execution; the model never authorizes anything. Results have `status: 'ok' | 'denied' | 'error' | 'cancelled'`; successful outputs must be JSON-compatible and satisfy an optional result schema. Handlers own their side effects and must observe their cancellation signal.

`runToolLoop(neko, tools, options)` feeds tool results back as **untrusted data**. `maxRounds` bounds executed rounds (1–16, default 4), not inference count. It makes one terminal selection after the last executed round, unless an earlier selection has no calls. `stopReason` is `'no-calls'` or `'max-rounds'`; `rounds`, `roundResults`, `messages` and aggregate `usage` record the work. Calls in the terminal `message.toolCalls` are returned but **never executed**. Do not execute them without a new explicit application decision.

### Independent worker pool

```ts
import { createNekoPool } from 'neko.js';

const reservation = 3 * 1024 ** 3; // Application estimate, not measured memory.
const pool = await createNekoPool({
  workers: [
    { options: { device: 'cpu' }, memoryBytes: reservation },
    { options: { device: 'cpu' }, memoryBytes: reservation },
  ],
  budget: { memoryBytes: 2 * reservation }, maxPending: 8,
});
try {
  const results = await pool.inferBatch([
    { prompt: 'Write a greeting.', maxNewTokens: 32 },
    { prompt: 'Write a farewell.', maxNewTokens: 32 },
  ]);
  console.log(results, pool.status());
} finally { await pool.dispose(); }
```

Each configuration owns an independent worker/runtime; concurrent owners may each load a model. FIFO pending requests are scheduled across available owners. This is **not tensor batching** or shared model memory. `maxWorkers` defaults to 4 (1–32); the nonempty `workers` list cannot exceed it. `maxPending` defaults to 8 (0–10,000), and excess admission fails with `QUEUE_FULL`. Positive `memoryBytes` reservations must fit `budget.memoryBytes`; this is application-declared accounting, not a measured or OS-enforced memory cap.

`pool.infer(request)` returns a result promise; `pool.submit(request)` returns `{ id, result, cancel, dispose }`. Item cancellation/disposal cancels only that request, not its owner or the pool. `inferBatch(requests, { signal? })` returns input-ordered fulfilled/rejected records with IDs, retaining per-item errors. Pool disposal cancels outstanding work and disposes all owners; it should always be awaited.

### Decoder state and vision reuse

`infer`/`inferStructured` accept `reuse: { retainState?, state?, vision? }`. With `retainState: true`, a successful result may expose `result.reuse.state`; pass that opaque handle as `state` with a compatible, exactly extending token prefix. `vision: true` opts into processed vision-encoder feature reuse. This is separate from normalized-pixel and rendered-prompt tokenization caches, and is not automatic session KV reuse.

```ts
const retained = await neko.infer({
  prompt: 'Write a greeting.', maxNewTokens: 32,
  reuse: { retainState: true, vision: true },
});
try {
  console.log(retained.reuse, await neko.reuseCacheInfo());
  // A continuation must match the exact token prefix and compatibility key.
} finally {
  if (retained.reuse?.state) await neko.releaseGenerationState(retained.reuse.state);
}
await neko.clearReuseCaches();
```

Handles are engine-owned, not portable snapshots; released, evicted or foreign handles cannot continue generation. Compatibility binds model/profile, schema/constraint mode, instructions, processed image identity and stop settings. Similar text is insufficient. Results report `reusedDecoderTokens`, `visionEncoderHits` and `visionEncoderMisses`, not a speed guarantee.

Set `createNeko({ reuseCache: { stateEntries, stateBytes, visionEntries, visionBytes } })` to bound entries and bytes (positive safe integers). Defaults are 4 states / 512 MiB and 8 vision entries / 64 MiB. Oversized states/features fail; retained entries can be evicted. `reuseCacheInfo()` returns cache accounting or `null` before engine acquisition; `releaseGenerationState(handle)` releases one state and `clearReuseCaches()` clears both caches. Engine release/disposal invalidates handles. `scripts/smoke-reuse.mjs` is an explicit real-model reuse exercise; its presence is not a claim that it passed on your backend.

Structured calls (`inferStructured` and report stages) additionally keep up to four engine-private decoder states for their fixed system instruction when it is at least 32 tokens, the request is text-only and no `reuse` option is given. They are never handles, do not change the result shape, count against the `stateBytes` limit separately from handles (an oversized one is skipped) and appear in `reuseCacheInfo()` as `prefixEntries`, `prefixBytes`, `prefixHits` and `prefixMisses`; `clearReuseCaches()` clears them.

### Generation diagnostics and lifecycle events

```ts
import { getGenerationDiagnostic } from 'neko.js';

try {
  await neko.inferStructured({
    prompt: 'Return a greeting.', schema, maxNewTokens: 64,
    diagnostics: { capture: { maxCharacters: 2048 } },
  });
} catch (error) {
  const diagnostic = getGenerationDiagnostic(error);
  console.log(diagnostic?.code, diagnostic?.usage);
  throw error;
}
```

`diagnostics` on `infer`, `inferStructured` and `describe` enables error metadata: stage/attempt, finish reason, usage, output length, and available JSON/schema details. `true` captures metadata without raw text; `{ capture: { maxCharacters } }` opts into at most 1–65,536 characters. `getGenerationDiagnostic(error)` returns metadata or `undefined` without replacing the original error. Captured model output is untrusted and can contain user/source content: do not indiscriminately log, upload or persist it.

`createNeko({ onEvent: (event) => { /* record content-free metrics */ } })` observes `NekoEvent`: request start/end IDs, `operation`, queue/duration timings, outcome/error code, token usage, model/execution identity, or engine-loaded timing. `NekoOperation` covers `infer`, `inferStructured`, `planInference`, `ask`, `describe`, `load`, and `warmup`; it is not a promise of a distinct outer event for every composed helper. Events contain no prompts, generated text or source documents. Observer exceptions/rejections never affect results, observers are not awaited, and worker forwarding is best-effort. This factory observer differs from per-report `DescribeOptions.onEvent`, which reports report stages.

### Report modes, planning and recovery

```js
const source = '<main><p>The park opened in 1987.</p></main>';
const extractedReport = await neko.describe(source, {
  mode: 'extractive', sourceLanguage: 'en', content: 'main',
});
const reportPlan = await neko.planReport(source, {
  maxNewTokens: 256, retries: { section: 1 },
});
console.log(extractedReport.sourceFacts, reportPlan.stages, reportPlan.reduction);
```

Extractive mode retains exact whole-paragraph quotes without tokenizer/ONNX loading, has zero token usage and `metadata.backend: null`, and retains image metadata without image fetching or vision. `sourceLanguage` defaults to `'und'`; it does not translate.

`planReport(input, options?)` reports source/context segmentation, `sectionCount`, `imageCount`, and stages with `id`, `phase`, `inputTokens: number | null`, `maxOutputTokens`, `maxAttempts`. `knownInputTokens`/`maxKnownOutputTokens` cover known stages only. `reduction.required` is `boolean | null`; `reduction.stageCount`, `estimatedDurationMs` and `totalTokensUpperBound` are `null` where unknown. Generated planning may acquire the engine; extractive planning does not. Plans do not promise duration, final total or successful output.

`retries: { image?, section?, summary?, conclusion? }` allows 0–3 extra attempts per phase, only for `STRUCTURED_OUTPUT`/`MODEL_OUTPUT`. Failed generation charges cumulative input/output usage. Budget-induced incomplete JSON is `BUDGET_EXCEEDED`, not retryable structured failure. Budget/retry authorization is excluded from immutable resume identity and may increase; spending never resets.

```js
import { ReportError } from 'neko.js';
try {
  await neko.describe(source, {
    maxNewTokens: 256, budget: { maxTotalTokens: 1024 },
    retries: { section: 0 },
  });
} catch (error) {
  if (!(error instanceof ReportError)) throw error;
  console.log(error.partial.completedStages, error.partial.usage);
  const resumed = await neko.describe(source, {
    maxNewTokens: 256, budget: { maxTotalTokens: 8192 },
    retries: { section: 0 }, resume: error.checkpoint,
  });
  console.log(resumed.metadata.resumedStages);
}
```

`ReportError.partial` is a typed version-1 partial result with snapshot, source facts, completed audited stages and cumulative usage/elapsed time, not a successful report. Partial/checkpoint survive worker transport. Increasing budget alone can resume with zero retries; incompatible immutable settings still fail. The example requires a compatible checkpoint and sufficient new authorization.

Cache/backend/status methods return promises except `cache.model.exportBundle()`, which synchronously returns a readable stream. `cache.model.prefetch/status/clear` affect only the selected model/profile; `cache.engine.status/release` inspect/release the live engine. `cache: { engine: true, engineTtlMs: 1_800_000 }` reuses it until 30 minutes idle. `backend.detect()` checks compatibility, not successful native driver/session creation. Default `execution: 'inline'` permits one realm-local inline owner; `'worker'` creates a real Node thread/browser module worker with independent ownership. Bounded FIFO admission (`queue: { maxPending: 8 }`) rejects excess work with `QUEUE_FULL`; `queueStatus()` exposes state. Worker callbacks preserve ordering and typed errors, partial reports and checkpoints. Streams retain queue ownership until EOF/cancellation. `dispose()` cancels queued work, awaits safe active cleanup, releases resources and restores hooks.

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

### Main content, tables and document questions

```js
const document = await extractPage(
  '<main><p>The park opened in 1987.</p><table><caption>Hours</caption>' +
  '<tr><th scope="col">Day</th><th scope="col">Time</th></tr>' +
  '<tr><td>Monday</td><td>09:00</td></tr></table></main>',
  { content: 'main' },
);
console.log(document.extraction, document.containers, document.tables);
const answer = await neko.ask(document, 'When did the park open?', {
  maxNewTokens: 256,
});
console.log(answer.status, answer.claims);
```

Default extraction is full content. Opt-in `content: 'main'` selects a conservative unique visible `main`/`article`; ambiguous/empty candidates fall back to full body, recorded in `extraction`. Paragraph `containerId`/`sectionId` and containers preserve relations. Tables preserve captions, geometry/spans, cells, row/column/header references and paragraph IDs. Selection may mark tables `partial`, retaining empty geometry without pretending omitted text remains. Page inputs/snapshots are owned and validated.

`ask(input, question, options?)` uses constrained model selection of known paragraph IDs and extractive claim text. SDK citations quote exact whole paragraphs with UTF-16 offsets, not model-computed offsets. Unknown/duplicate IDs and invalid output are rejected; unsupported claims return `insufficient-evidence`. Selected evidence is not silently omitted/truncated to fit context. Supported quotes guarantee neither truth nor question relevance; images are not visual evidence in QA.

### Local document indexes and exact-substring answers

```ts
import {
  createDocumentIndex, importDocumentIndex, documentFromPage, askDocuments,
} from 'neko.js';

const page = await extractPage('<main><p>The park opened in 1987.</p></main>');
const index = await createDocumentIndex([
  documentFromPage(page, 'park'),
  { id: 'hours', text: 'Monday opening hours are 09:00–17:00.' },
], { chunkSize: 1200, maxDocuments: 1000 });
const found = index.search('opening hours', { topK: 4 });
const answer = await neko.askDocuments(index, 'When did the park open?');
// Equivalent helper with an explicit inference/planning host:
const other = await askDocuments(neko, index, 'What are Monday opening hours?');
const restored = await importDocumentIndex(JSON.parse(JSON.stringify(index.exportSnapshot())));
console.log(found.coverage, answer.claims, other.retrieval, restored.id);
```

`createDocumentIndex(documents?, options?)` asynchronously owns `{ id, text, title?, url?, blocks? }` documents and creates bounded BM25/CJK chunks. Defaults: `chunkSize: 1200`, `maxDocuments: 1000`, `maxCharacters: 8_000_000`, `maxChunks: 100_000`. `documentFromPage(page, id?)` joins paragraph text and retains paragraph provenance; its default ID is `page.url`. `search(question, { topK?, documentIds?, maxScoredChunks? })` returns hits, quotes and explicit coverage, always `exhaustive: false`. Retrieval does not establish document-wide absence, relevance or completeness.

`updateDocument(document)`, `replaceDocuments(documents)` and `removeDocument(id)` atomically update the index; successful mutations change its identity and make old answers stale. `exportSnapshot()` returns a version-1 immutable snapshot suitable for caller-managed JSON persistence; `importDocumentIndex(snapshot)` validates identities and rebuilds retrieval structures. Snapshots contain source text, not authentication or embedding vectors; protect sensitive persisted content.

`askDocuments(host, indexOrSnapshot, question, options?)` / `neko.askDocuments(...)` plan whole retrieved chunks against the actual tokenizer/context budget before structured generation. Options include `search`, `maxNewTokens` (default 512, 1–2048), `contextWindowTokens`, `generation`, `signal`, `hardDeadlineMs`, and provisional `onToken`. `retrieval` records selected/context-omitted chunk IDs and planning coverage; no full-document fallback is submitted. Claims must be exact contiguous substrings of **every** cited chunk. SDK citations carry document/chunk version IDs, source provenance and UTF-16 offsets into canonical document text; offsets do not refer to original PDF/HTML bytes. `status` is `'answered'` or `'insufficient-evidence'`. Exact quotes are not fact-checking, OCR verification or a relevance guarantee.

### Caller-owned hybrid retrieval

Neko.js bundles **no embedding model**; the pinned Qwen generation model is not an embedder. Applications own the embedding model's provenance, licensing and integrity checks. Supply a real application-owned `DocumentEmbedder`; the following function receives that dependency rather than inventing vectors:

```ts
import type { DocumentEmbedder, DocumentIndex, Neko } from 'neko.js';

async function queryWithEmbeddings(
  neko: Neko, index: DocumentIndex, embedder: DocumentEmbedder,
) {
  const found = await index.searchHybrid('opening hours', {
    embedder, topK: 4, batchSize: 32, maxEmbeddedChunks: 4096,
  });
  const answer = await neko.askDocuments(index, 'What are the opening hours?', {
    embedder, embedding: { batchSize: 32, maxEmbeddedChunks: 4096 },
    search: { topK: 4 },
  });
  return { found, answer };
}
```

`DocumentEmbedder` has readonly `id`, `dimensions` (1–8192), and `embed(texts, { kind: 'query' | 'document', signal? }): Promise<readonly ArrayLike<number>[]>`. Return one finite, nonzero vector of the declared dimensions per input, in order; asymmetric models can apply their own query/passage prefixes. Change `embedder.id` whenever the model, prompt format or anything else could change vectors.

`DocumentIndex.searchHybrid(question, { embedder, ...searchOptions, batchSize?, maxEmbeddedChunks?, minSimilarity?, signal? })` combines lexical BM25 and cosine-ranked candidates using reciprocal-rank fusion (RRF). Returned `score` values are **rank-based fusion scores, not similarities**. `batchSize` is 1–256 (default 32); `maxEmbeddedChunks` is 1–100,000 (default 4096) and limits newly embedded chunks, rejecting excess rather than silently omitting them. `minSimilarity` is an optional cosine floor in [-1, 1] for semantic candidates only. Coverage includes semantic embedded/cached/scored counts and still reports `exhaustive: false`.

`AskDocumentsOptions.embedder` enables the same hybrid path; `embedding` supplies those semantic settings and requires an embedder. Vectors are cached per `DocumentIndex` instance by embedder ID and content-derived chunk version in a bounded 128 MiB cache; unchanged chunks can reuse them after mutation. Passing a snapshot re-imports a fresh index each call, so reuse requires passing the **DocumentIndex object**, not its snapshot. Queries are embedded each search. Hybrid retrieval still does not promise exhaustive evidence or semantic recall.

### PDF extraction and image OCR

```ts
import { extractPdf, documentForIndex, ocrImage } from 'neko.js';
import { readFile } from 'node:fs/promises';

const pdfBytes = new Uint8Array(await readFile('./article.pdf'));
const native = await extractPdf(pdfBytes, { id: 'article', ocr: 'none', maxPages: 20 });
const withOcr = await neko.extractPdf(pdfBytes, { id: 'article-ocr', ocr: 'scanned' });
const imageBytes = new Uint8Array(await readFile('./scan.png'));
const scan = await neko.ocr(imageBytes, { id: 'scan', maxNewTokens: 2048 });
const sameApi = await ocrImage(imageBytes, (request) => neko.inferStructured(request));
const documents = await createDocumentIndex([
  documentForIndex(native), documentForIndex(scan),
]);
console.log(withOcr.pages, sameApi.provenance, documents.id);
```

These helpers accept owned `Uint8Array`, `ArrayBuffer` or `Blob` data, not paths or URLs; `ocrImage` / `neko.ocr` also accept decoded raster pixels. The example's file reading is application IO. `extractPdf(bytes, options?)` parses inert PDF data without activating scripts, links, attachments or XFA. Default `ocr: 'none'` extracts native text with zero inference usage. `'scanned'` performs OCR only on pages with no native layout blocks; `'all'` replaces each page's native blocks with OCR output. Standalone PDF OCR requires `infer: (request) => neko.inferStructured(request)`; `neko.extractPdf` supplies the host automatically.

PDF defaults (upper bounds in parentheses): `maxBytes` 32 MiB (256 MiB), `maxPages` 100 (1000), `maxPagePixels` 8 million (40 million), `maxTotalPixels` 40 million (400 million), `maxItems` 100,000 (1 million), `maxTextCharacters` 2 million (10 million), `maxCells` 4096 (16,384). Limits reject excess; they do not silently truncate. `renderScale` defaults to 1.5 (0.25–4); `password`, `id`, `title` and `signal` are available. OCR additionally accepts `maxBlocks`, `maxNewTokens`, `contextWindowTokens` and `hardDeadlineMs`. Worker hard-deadline restrictions of the inference host still apply.

Image OCR defaults: `maxBytes` 64 MiB (maximum 256 MiB), `maxPixels` 40 million (hard ceiling), `maxBlocks` 128 (2048), `maxCells` 512 (4096), `maxTextCharacters` 100,000 (2 million), `maxNewTokens` 2048 (maximum 2048). Raster decoding rejects SVG. Transcription uses actual supplied pixels and structured inference; **no real-model OCR accuracy is asserted**. Text, geometry, reading order and table grouping remain untrusted. Native PDF blocks use `provenance: 'native-text'`; OCR uses `'model-ocr-untrusted'` and `accuracy: 'not-verified'`. Native provenance is not a truth guarantee. PDF geometry uses top-left PDF points; image OCR uses top-left pixels. `documentForIndex(pdfOrOcr)` preserves canonical text spans and source provenance while creating an owned index payload.

Node loads local `pdfjs-dist` assets and native canvas lazily; a custom `assetBase` must be a local `file:` directory URL. Browser `assetBase` is an application-controlled, trailing-slash directory containing `pdf.mjs`, `pdf.worker.mjs`, `cmaps/`, `standard_fonts/`, `wasm/` and `iccs/` (default `./assets/pdf/` relative to the module). Browser raster rendering requires a **main-thread document/2D canvas**: compose `extractPdf(bytes, { ocr: 'scanned', infer: request => worker.inferStructured(request) })` on the main thread instead of rendering inside a worker. Browser CPU/WASM inference remains unsupported; browser OCR needs a supported WebGPU inference host.

### Image regions and preprocessing reuse

```js
const regional = await neko.infer({
  image: './photo.png', prompt: 'Describe these regions.', maxNewTokens: 64,
  region: { unit: 'normalized', x: 0, y: 0, width: 1, height: 0.5 },
  tiling: { tileWidth: 640, tileHeight: 640, overlap: 32, maxTiles: 16 },
  maxDimension: 1280,
});
console.log(regional.images);
```

Use approved paths as above. `region` takes `unit: 'pixels' | 'normalized'`, `x`, `y`, `width`, `height`, after EXIF orientation and before crop/resize. Tiling partitions that region; inference caps source images at 16 and total regions at 64. Decoded images are capped at 40 million pixels; output dimensions never exceed 1280. Browser raw pixels support 1/2/3/4 channels.

The engine's byte/count-bounded cache owns normalized pixels, not vision embeddings. Inputs are re-read, re-authorized and digested before hits. `ImageObservation` records `sourceVersionId`, source dimensions, pixel/normalized region, processed `versionId`, and preprocessing pipeline/cache/reuse kind. Hashes detect changes, not authenticity or vision accuracy.

Remote URL fetching creates SSRF risk when callers accept untrusted URLs. Supply `validateDestination` to check every destination/redirect and also apply application-level outbound network controls; this SDK cannot determine which private or internal destinations are safe for a particular deployment.

`loadImage(image, options?)` decodes an extracted image using the Node native decoder or browser bitmap/canvas path. It checks raster bytes/MIME, caps input and decoded dimensions, applies orientation, scales to at most 1280×1280 without enlarging, and emits PNG bytes. SVG and mismatched/invalid content are rejected. It preprocesses only; `Neko.describe()` performs the model inference. Browser remote image requests remain subject to CORS.

`renderMarkdown(report)` escapes untrusted text and links only HTTP(S) provenance. `await validateStructuredReport(report, selectedPage?)` audits the persisted selected source snapshot and SHA-256 versions, exact quoted offsets/content, claim spans and evidence references, complete selected-source coverage, image status/provenance, language, loaded model/session identity, and aggregate metadata. An optional matching selected `Page` adds an external-source comparison; no fetch is needed to audit saved JSON. It detects inconsistent references/accounting, not whether a model claim is entailed by its source or image pixels. Hashes are not source authentication.

### Versioned reports and checkpoints

Reports use `schemaVersion: 3`, `mode: 'generated' | 'extractive'`, and carry `sourceFacts` plus `integrity: { algorithm: 'sha256', checksum }`. `metadata.coverage` records selected paragraph IDs/character count, retained quote/character count, model- and summary-cited fact IDs, `conclusionBasis` (`'retained-source'` or `'reduced-generated-claims'`), and `semanticRetention: 'not-measured'`. Source coverage is structural accounting, not semantic recall. Each claim's `audit.status` is `supported`, `contradicted` or `unknown`, using `conservative-lexical-v1`: exact quote retention and narrowly aligned lexical conflicts, not semantic truth, confidence, relevance, translations or image-pixel verification.

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

Serialization and parsing validate before accepting data; optionally pass a matching selected `Page` as the second report-helper argument. `validateReportCheckpoint(value)` validates an in-memory checkpoint. Current checkpoints use `version: 3` and `plan: 'evidence-first-v3'`, persist the quote ledger, and require matching immutable source/model/settings identity. **Breaking persistence policy:** unversioned/older/future reports and checkpoints are rejected, not migrated or interpreted through aliases. Report rejection is a `TypeError`; invalid checkpoint versions use `CHECKPOINT_INVALID`. Recreate old persisted data from original inputs. Checksums do not authenticate authors; saved source text, metadata and generated text may be sensitive.

Checkpoint `sectionPlan` records ordered groups of one to four source-fact IDs covering the ledger exactly. Validation rejects section citations into other groups even if a checksum is recomputed; resume also verifies the deterministic plan and request hashes. Persist SDK-issued checkpoints through the helpers rather than constructing saved state manually.

## Cache and backend notes

The model manifest fixes the Hugging Face revision and SHA-256/size of required files. Every cache hit is verified before use; mismatches fail instead of silently becoming misses. `neko.cache.model.prefetch/status/clear` operate on only these pinned files. Browser Cache Storage remains subject to browser user actions and eviction.

On Node, each ONNX payload file is hashed once per installed runtime: Transformers.js requests these files several times per load, so an unchanged file (same device, inode, size, mtime and ctime; ctime cannot be set by callers) is not re-hashed within that installation. Every new process, any changed file identity and every non-ONNX file are verified in full. Node sessions use `os.availableParallelism()` intra-op threads. On one Apple M-series machine (4 performance + 6 efficiency cores) this raised 416-token prefill from about 213-238 to about 290 tokens/s and lowered decode from about 28 to about 25 tokens/s; these are single-machine observations, not a guarantee, and the best thread count depends on the hardware.

### Concurrent downloads and validator-bound resume

```ts
const downloading = await createNeko({
  device: 'cpu', downloadConcurrency: 3, resumeDownloads: true,
  onCacheProgress: (event) => {
    console.log(event.file, event.phase, event.loaded, event.total, event.resetReason);
  },
});
try { await downloading.cache.model.prefetch(); }
finally { await downloading.dispose(); }
```

`NekoOptions.downloadConcurrency` bounds simultaneous pinned-file downloads (1–16, default 3). `resumeDownloads` defaults to `true`, preserving interrupted staging for the next explicit installation; it does not automatically retry a failed request. `onCacheProgress` receives `CacheProgress` with `file`, byte `loaded`/`total`, phase `'download' | 'verify' | 'resume'`, optional `resumedFrom` and `resetReason`. Resets may report `'source-changed'`, `'resume-disabled'`, `'validator-unavailable'`, `'invalid-partial'`, `'range-rejected'`, `'validator-changed'` or `'integrity'`.

Resume binds partial bytes to the pinned source/size/hash and a strong ETag or usable Last-Modified validator, sending `Range` with `If-Range`. Invalid validators, changed objects or rejected ranges reset staging; complete content is still freshly size- and SHA-256-verified before promotion. Signed CDN query strings are ignored only when matching a partial's destination by object location (origin + path), not when validating content or authorizing arbitrary destinations. Same-origin Hub `/api/resolve-cache/` redirects are accepted only for the exact registered pinned model/revision/file.

Node install locks coordinate across processes; browser installation uses the Web Locks API and requires its availability. Locks coordinate cache installation, not inference ownership or application network policy. Local-only mode still fails on missing/corrupt assets rather than using unverified partial data.

### Portable offline bundles and diagnostics

```js
// Independent worker hosts with the same model/profile; source is prefetched.
const controller = new AbortController();
// Returns a stream immediately; pass the signal itself, not { signal }.
await target.cache.model.importBundle(
  source.cache.model.exportBundle(controller.signal), controller.signal,
);
console.log(await target.cache.model.diagnostics());
console.log(await target.diagnostics());
console.log(await target.health({ timeoutMs: 1000 }));
```

`importBundle(Blob | ReadableStream<Uint8Array>, signal?)` validates strict versioned manifests, pins, sizes and digests, stages before publication and supports cancellation. Import preserves verified/unrelated files; corrupt cache fails closed instead of silent replacement. Bundle transfer is not inference/quality validation. Node filesystem quota is unknown (`null`); browser diagnostics use actual `navigator.storage.estimate()` when available, not a memory guarantee. Browser storage availability/quota/eviction remain host-dependent.

Bundles contain selected model assets, not native dependencies or browser runtime/bootstrap files. Before browser `localFilesOnly: true` inference, separately deliver the matching shipped `.mjs`/WASM assets and cache the requested WASM URL; a missing runtime asset remains an offline-cache error. The existing browser runners seed these assets. Static app/worker modules still need delivery; this is model-network-offline, not a complete offline website. Same-origin browser hosts share CacheStorage. Exported chunks own only their visible bytes, preventing transferable streams from cloning oversized upstream backing buffers.

Diagnostics exclude prompts, image bytes and model output. `health()` is lightweight readiness/transport metadata, not generation or quality testing. Worker restart is explicit:

```js
const worker = await createNeko({ device: 'cpu', execution: 'worker' });
try {
  await worker.infer({ prompt: 'Hello.', maxNewTokens: 16, hardDeadlineMs: 30_000 });
  console.log(await worker.health());
  await worker.restart();
} finally { await worker.dispose(); }
```

Worker-only `hardDeadlineMs` terminates the realm on expiry and fails **all pending/queued calls**, without automatic replay. Explicitly restart before reuse; model sessions must load again and interrupted calls are not replayed. Inline hard deadlines/restart are unsupported; dispose/create instead. Ordinary abort/duration budgets are cooperative, not native-call preemption.

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

**Current SDK verification**, macOS/arm64:

- Node 22.23.3 CPU/default: cold text planning without ONNX, Unicode/integral-exponent/unique-enum constrained generation, bounded iterators, transactional session eviction/branch/reset/import, two ROI tiles (152 input / 24 output tokens), report audits and cumulative budget-only resume. Full 871,364,778-byte streaming bundles transferred between workers into a new filesystem cache and produced actual local-only inference; returned-stream abort preserved health.
- Both registered 2B profiles completed actual Node text, image and report inference. This does not establish 2B browser support or quality parity.
- Full Chromium 153.0.8010.12 headless with `--enable-unsafe-webgpu`: real 0.8B text, Unicode grammar, iterators, sessions, ROI and paragraph-cited QA. A separately shipped/cached runtime plus full exported Blob was imported into **empty** CacheStorage, then a new local-only worker with model-network denial generated 8 tokens. Integral-exponent/unique-enum output used 17 tokens; a one-output-token budget failure resumed with zero output retries (1,212 cumulative tokens). Hard expiry rejected both inference and queued planning with `DEADLINE_EXCEEDED`; explicit restart recovered health/light planning. The final browser evidence surface was visually checked.
- Typecheck/lint/build, 130 Node contracts, 27 deterministic quality-tool tests, 21 browser contracts and packed-consumer real inference passed. The independent [full four-fixture quality gate still fails](quality.md#current-measured-outcome). Provider evidence remains loaded-session configuration, not per-operator/hardware GPU certification.


Historical evidence snapshot retained from earlier validation (2026-10-03), macOS (Darwin 27), arm64, Node 22.23.3/26.7.0. The table and detailed runs below verify only their recorded runtime/backend/profile/input. They are not revalidation of every newly documented API, browser full-bundle transfer, a full quality gate, other platforms or every Node `>=22` release.

| Runtime/platform | API/contract status | Historical real-inference evidence | Recorded prior verification |
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
