# Changelog

## Unreleased

- **Fix (affects 1.5.0):** a first-use model download from Hugging Face could not complete. Observed against the live Hub on 2026-10-03: non-LFS files redirect same-origin to `/api/resolve-cache/models/<id>/<revision>/<file>`, which the default network policy denied (`POLICY_DENIED`), and interrupted CDN downloads could never resume because the signed redirect URL changes per request and was compared as part of the object location. The exact pinned model/revision/file `resolve-cache` path is now trusted (size/SHA-256 verification unchanged) and resume matches by origin and path, including partial staging written by 1.5.0. Verified by a complete live prefetch of the pinned default profile with every file SHA-256 checked.
- **Fix (CI, Windows):** the 1.5.0 remote CI run failed 3 of 22 jobs, all `windows-2025`, in `npm run test:quality`. The CLI test built its script path from `URL.pathname` (`D:\D:\a\...`), and Git's CRLF conversion changed the byte-hashed quality fixtures (`LOCAL_FIXTURE_INTEGRITY`). The path now uses `fileURLToPath`, and `.gitattributes` marks `scripts/quality/fixtures/**` as `-text`. Linux and macOS locally pass 27/27; the Windows fix is not yet verified on a Windows runner.
- Add opt-in hybrid document retrieval: caller-supplied `DocumentEmbedder`, `DocumentIndex.searchHybrid` and `askDocuments({ embedder, embedding })` combine BM25 with cosine ranking by reciprocal-rank fusion. Neko.js bundles no embedding model; embedder output is validated as untrusted, vectors are cached per index by content version under a fixed 128 MiB bound, and retrieval remains non-exhaustive.
- Add `runToolLoop`: a bounded select, approve/execute, feed-back and re-infer loop over `inferTools` and `executeToolCalls`. Approval stays mandatory and sequential; calls requested by the terminal selection are returned, never executed.
- Add opt-in `onEvent` observability: content-free request lifecycle and engine-load events (ids, timings, token counts, error codes, model identity). Observer failures never change request results; worker forwarding is best-effort.
- Document every 1.5.0 API and the items above in the English and Traditional Chinese usage guides.
- Packaging checks now fail on sync-conflict duplicate files (the working tree is iCloud-synced and produced `* 2`/`* 3` copies) and bundle `neko.js`, `neko.js/documents`, `neko.js/web` and `neko.js/report` with esbuild under browser conditions.
- Verified on macOS arm64, Node.js 26.7.0 CPU, default profile: 217 Node tests, 27 quality-tool tests, 21 browser contracts (Chromium, Firefox, WebKit), site smoke, packed-artifact check, the state-reuse smoke against the real model (cached continuation equal to baseline text and usage, branch isolation, abort leaves the cache unchanged, vision feature hits), one synthetic image OCR, one synthetic scanned-PDF OCR and native PDF extraction without inference. Observed, not asserted as quality: document QA on the PDF fixture returned `insufficient-evidence`, and a 128-token budget truncated structured JSON (`STRUCTURED_OUTPUT`). Not verified: browser WebGPU, browser PDF extraction, the new hybrid retrieval with a real embedding model, other platforms, and the remote CI matrix.

## 1.5.0 — 2026-10-03

- Add concurrent, resumable pinned-model downloads with validator-bound staging, cross-process/browser installation locks and explicit cache progress; preserve size/SHA-256 verification before promotion.
- Add engine-local, entry/byte-bounded decoder-state and vision-feature reuse, opaque retained-state handles and explicit release/clear controls. Add opt-in generation/report diagnostics with bounded output capture; reuse does not imply backend parity or output-quality guarantees.
- Add typed tool definitions, structured tool selection and separately invoked, approval-gated tool execution; model-selected calls never authorize handlers.
- Add inert native PDF extraction, bounded PDF rendering and model-backed OCR composition, provenance-aware document indexes, local retrieval and multi-document questions with exact validated citations. Require Node.js 22.13 or newer and package browser PDF assets.
- Add bounded structured/report/web-question/document-question streams and transactional session streaming; commit conversation history only after successful stream consumption.
- Add bounded worker pools, cancellation-aware scheduling and batch results with independent worker ownership; this is not tensor batching.
- Add a bilingual static project website with bounded preview routes. Expand CI contract coverage across OS/Node/browser matrices, packed-artifact checks and explicitly opt-in real CPU/WebGPU/quality lanes.
- Verify typecheck, lint, build, 191 Node tests, 27 quality-tool tests, the static-site HTTP smoke, packed-consumer artifact checks and built public native-PDF extraction on macOS ARM64 with Node.js 26.7.0. These checks do not exercise real model inference, model-backed OCR, browser PDF extraction or the new remote CI matrix; no fresh model-quality or cross-platform parity pass is claimed.

## 1.4.0 — 2026-10-03

- Add tokenizer-aware JSON constrained decoding for an explicit supported schema subset, inferred `SchemaValue` results and explicit validation-only fallback. Reject unsupported constrained schemas before model acquisition; handle integral exponent constants, finite unique-enum domains and UTF-8 token boundaries without weakening runtime validation.
- Add bounded AsyncIterable inference streams, transactional conversation sessions, context eviction, branching/reset and model-bound snapshots. Cache only exact rendered-prompt tokenization; do not claim growing-prefix or model KV-cache reuse.
- Make text-only inference planning tokenizer/configuration-only, without ONNX sessions; add staged report planning with explicitly unknown reduction, duration and total-cost bounds.
- **Breaking persisted contracts:** migrate reports/checkpoints to version 3 and `evidence-first-v3`. Add conservative claim–evidence audits, zero-inference extractive reports, typed partial reports, cumulative failed-attempt accounting, increased-budget/retry authorization and bounded per-stage retries. Budget-induced incomplete JSON is `BUDGET_EXCEEDED`, not an output retry.
- Add inert main-content extraction, structured tables/paragraph relations and document questions with SDK-constructed exact citations; reject unsupported claims without pretending citations certify truth or relevance.
- Add EXIF-oriented image regions, bounded tiling, preprocessing provenance and byte/count-bounded owned pixel caches; reauthorize/read/validate before reuse. Support browser raw images with all four channel layouts.
- Add integrity-checked streaming offline bundles with staged imports/cancellation, installation/quota diagnostics, privacy-safe health/diagnostics, explicit worker restart and realm-wide hard deadlines without replay. Bound transferred chunk backing buffers rather than cloning whole upstream allocations; tighten POSIX cache-ancestor trust checks.
- Register immutable Qwen3.5-2B revision `2ea7886f48b926aca97de8b0e041ffca7e3ebaa9` alongside the default 0.8B model, with pinned default/all-q4 assets and provenance; exercise both 2B profiles with actual Node text/image/report inference.
- Migrate callers, packed-consumer declarations and quality hierarchy contracts; advance the finite evaluator to `quality-claims-v5` without lowering acceptance thresholds. Consolidate English/Traditional Chinese API and security documentation after implementation.
- Verify 130 Node/27 quality-tool/21 browser contracts, packed-consumer inference, full Node streaming bundle transfer and Chromium 153 WebGPU cold Blob import followed by actual local-only inference, budget-only resume and deadline/restart. Runtime assets remain separately deployed/cached. The full four-fixture quality gate still fails; retain its diagnostics without lowering policy or claiming model-quality acceptance.

## 1.3.0 — 2026-10-03

- Validate cheap inference/report options before acquiring a model in inline and worker execution; expose `planInference()` with exact chat/schema/image-expanded input tokens, output capacity and context-fit metadata. Valid cold planning calls still load the model/processor.
- Stop structured generation at a deterministic single-JSON-value boundary and retain fail-closed Draft-07 runtime validation. Expose `json-boundary-runtime-validation` evidence; no schema grammar constraints, repair, retry or factual guarantees.
- Preserve all selected paragraph text in an exact source-quote ledger independent of generated summaries; bound section planning to multiple evidence-linked claims and expose structural coverage, conclusion basis and explicitly unmeasured semantic retention.
- **Breaking persisted contracts:** reports use `schemaVersion: 2`; checkpoints use `version: 2` and `evidence-first-v2`. Add validated report/checkpoint serialize/parse helpers and report integrity checksums. Reject older, unversioned and unknown versions without automatic migration; checksums are not authentication.
- Record trusted worker execution identity before report checksum generation, so packed Node and browser-worker reports pass validated persistence without post-generation metadata mutation.
- Add quality-regression CI gates separate from informational benchmark output; do not equate valid schemas/provenance with model quality.
- Complete bilingual first-use Node cache/offline and browser mirror/runtime/CORS instructions; derive local tarball filenames from `npm pack`. Separate platform API contracts, historical real inference and fresh verification instead of claiming Node/browser/OS parity.
- Exercise current changes on macOS/arm64 Node 22.23.3 and 26.7.0 CPU, the packed Node consumer, and Chromium 153 WebGPU inline/worker inference. Document that the strict four-fixture quality gate still fails for missing format prefixes and malformed hierarchical model JSON; no quality or cross-platform parity pass is claimed.

## 1.2.0 — 2026-10-03

- Upgrade the pinned Node image-processing dependency to `sharp@0.35.4` to include upstream libheif and libvips security fixes; retain an exact-version install-script approval.
- Add source-aware Page selection, full content snapshots, versioned citations and claim-evidence auditing; add aggregate report token/time budgets, resumable checkpoints, events, and deadline-bound async selector/resource/event/checkpoint hooks.
- Add bounded FIFO request admission, real Node/browser workers, cancellation and streaming, queue status, and actual load/warmup/runtime readiness APIs.
- Apply instance-scoped fail-closed policy to network and local inputs; preserve bound class policy hooks without freezing caller objects, capture model-source getters once before worker serialization, strip sensitive headers on cross-origin hops without restoring them, and align trusted worker URLs across source and bundled runtime layouts. Fail closed on opaque browser redirects; support pinned model mirrors and a verified loopback cache-mirror helper.
- Add typed conversations, joint multi-image inference, bounded generation/sampling/stop controls, and Draft-07 structured-output runtime validation; this does not claim constrained decoding.
- Register the pinned all-q4 profile with native/browser execution evidence without implying output-quality guarantees.
- Seed offline browser quality runs from the verified Node cache into the SDK CacheStorage contract, then enforce `localFilesOnly` and block external network requests; report seed and zero-HF-request evidence.
- Seed headed browser UI smokes from the verified Node-cache mirror into browser Cache Storage, then require local-only inference and block model-host requests.
- Correct quality claim scoring so supported cross-ID context does not invalidate a matching fact, and accept both equivalent circle-left-square / square-right-circle relations while rejecting opposite or negated positions.

## 1.1.0 — 2026-10-02

- Deliver the `createNeko()` SDK: text/image inference, URL/HTML structured reports, model/backend inspection, explicit cache controls, cancellation and streaming.
- Bundle conditionally for Node/browser and include browser WASM assets in prebuilt `npm pack` archives; pin Node native `onnxruntime-node@1.30.0`, `sharp@0.34.5`, and `parse5@8.0.1` as runtime dependencies.
- Validate report structure, complete paragraph/image provenance, and generated-field languages. English and Traditional Chinese script mismatches are rejected with `LANGUAGE_MISMATCH`; incomplete generations return `INCOMPLETE_GENERATION` without automatic retry.
- Exercise the actual packed consumer on Node native CPU and drive the current headed browser UI through image, text and report inference, then repeat from the same browser profile with model network access blocked.
- Verify native CPU hierarchical reporting on 300 paragraphs with a 1536-token context and 512-token output budget, preserving all source IDs across eight generated sections.
- Keep `imageFailurePolicy: 'omit'` limited to image-inference failures; caller `onToken` exceptions and aborts still propagate, including callback `throw undefined`.
- Document observed inference limits: saturated red/blue regions were described as pink or pinkish-red, and a hierarchical report omitted facts and contradicted its source in the conclusion. Valid schema/provenance does not guarantee faithful meaning; these observations do not isolate the model or establish Ollama parity.

## 1.0.0 — 2026-10-02

- Pin the Node native ONNX Runtime to 1.30.0 and use matching browser ONNX Runtime Web GPU/WASM assets.
- Verify the pinned Qwen model with real Node inference, including cold-cache download and fresh-process offline-cache runs; exact measurements are in the usage guides.
- Complete real WebGPU inference in the browser demo and verify offline reload in the same persistent profile with `localFilesOnly`. Label the browser CPU/WASM UI option and smoke helper as unsupported for this pinned model; observed session creation fails at `GatherBlockQuantized(1)` with no fallback.
- Preserve image extraction provenance and align extraction tests with observed browser image discovery behavior.
