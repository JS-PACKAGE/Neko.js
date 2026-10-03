# Changelog

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
