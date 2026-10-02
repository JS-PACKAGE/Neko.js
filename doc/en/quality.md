# Quality benchmark

The repository benchmark exercises three fixed, reproducible inputs against the pinned SDK model: a five-fact text fixture, a rasterized image fixture, and a 300-paragraph hierarchy fixture. It runs the real Node or browser SDK and retains the SDK's raw per-case results and errors. It is an oracle-based regression benchmark, not a general-purpose factuality test or a guarantee that model output is correct.

## Run

```sh
MODEL_CACHE=/absolute/path/to/verified-node-cache

# Node, CPU, default profile; offline unless the flag below is supplied
npm run quality:benchmark -- --runtime node --device cpu --model-profile default \
  --cache-dir "$MODEL_CACHE" --output artifacts/quality-node.json

# Browser, WebGPU; use a dedicated persistent Chromium data directory
npm run quality:benchmark -- --runtime browser --device webgpu \
  --cache-dir "$MODEL_CACHE" --browser-profile .cache/neko-quality-chromium \
  --browser-port 4173 --output artifacts/quality-browser.json

# Deterministic tests for the evaluator itself (does not run a model)
npm run test:quality
```

The benchmark defaults to `--allow-network` off. Node runs use the selected verified filesystem cache. Offline browser runs require `--cache-dir PATH`: the runner verifies that Node cache, serves the selected profile only from a temporary loopback mirror, seeds complete responses into the SDK's actual `env.cacheKey` CacheStorage plus the pinned runtime WASM asset, then enables `localFilesOnly`, takes Chromium offline, and blocks external requests. A missing or unverified profile prevents inference; no HF model requests should occur. Artifacts record cache and network evidence. Pass `--allow-network` only when you explicitly intend to let the SDK use network model sources. There are no automatic retries. Cache files are verified against the SDK's pinned manifest.

`--cache-dir PATH` selects a Node filesystem cache and is also the verified seed source for offline Browser runs; model inference still uses the browser Cache API. Browser cache is origin-scoped: use a dedicated `--browser-profile` directory and keep the same `--browser-port` between runs to reuse it. Port `4173` is the default and must be available; choose another stable port with `--browser-port` if it is occupied. A browser run without `--browser-profile` uses an ephemeral context, so its cache does not persist between runs. Do not point `--browser-profile` at a personal browser profile.

Other options include `--case text|image|hierarchy` to run one fixture, `--model-profile default|all-q4`, `--context-window-tokens N`, `--max-new-tokens N`, and `--headed`. Native ONNX profiling can be requested with `--profile-prefix PATH` (Node only). `--output PATH` writes the complete JSON artifact and prints a short summary; without it, the complete artifact is printed to stdout. `--help` lists the CLI options.

## Reading the artifact

Each run records a `stimulusKey` derived from fixture hashes, actual input/prompt hashes, expected record IDs, and effective token/context limits; it identifies the model inputs, not the scoring rules. `oracleSha256` and `evaluatorVersion` identify scoring inputs and semantics separately. The `caseInputs` entries expose each submitted-input SHA-256 and generation budget. Run configuration records runtime, device, profile, and network permission, so a matching stimulus key alone does not imply matching runtime or backend. Offline rescored artifacts also record `rescoreSource` (the prior evaluator version and stimulus key); raw model outputs, errors, and `caseInputs` remain unchanged.

The artifact contains:

- Model ID/revision, requested and observed model profile, requested runtime/device, and observed backend. Inference results report profile/backend directly; structured reports report them in `metadata`. A missing or mismatched reported profile is `profile-unverified`, prevents a comparable result, and stops later cases rather than silently substituting another profile.
- Fixture source/raster hashes, extracted source-ID count, cache status, bounded per-file/phase progress summaries, browser cache-seed and external-request evidence when applicable, environment versions, and (when requested) native ONNX trace summaries.
- Raw SDK output for each attempted inference/report and structured error details for failed or blocked cases. Inspect the raw output alongside the metrics; a metric is not a substitute for the generated text or report.

`status` is `completed` only when all selected cases complete with verified fixture integrity; `incomplete` indicates a failed inference, unverified profile, or offline network-policy violation; `blocked` indicates cases were not attempted (for example, offline with a missing cache); `fixture-invalid` indicates fixture integrity failed. `comparable` is true only for a fully completed run that satisfies the network policy. For meaningful cross-run comparison, require the same `stimulusKey`, model revision/profile, and observed backend identity, and account for the runtime/browser version recorded in the artifact.

The image case submits the same manifest-pinned PNG bytes to Node as a `Blob` and to Browser as a data URL. `imageRenderer.rasterSha256` records the raster bytes; `caseInputs.inputSha256` hashes the submitted data-URL string. The SDK image `versionId`, however, hashes decoded pixels plus dimensions and channel count: Node uses Sharp to produce sRGB raw pixels without alpha, while Browser uses `createImageBitmap` and Canvas `ImageData` (RGBA). Thus identical encoded PNG bytes do not guarantee equal decoded/preprocessor representations or `versionId`s. Do not attribute cross-runtime/profile score differences solely to quantization; this benchmark does not isolate decoder or preprocessing effects.

## What the metrics mean

### Text and image cases

The evaluator scores explicit-ID claim lines against the fixture's finite regex oracle. The `FACT` prefix is optional for factual scoring; omitting it is reported in `formatCompliance` and does not turn a matching fact into a false claim. Non-empty lines without a recognizable explicit ID are format errors, not false factual claims. A true positive must match its ID's expected signature and must not be duplicated, contradicted, or marked unsupported. Other supported facts may appear as context without invalidating a matching claim; a cross-ID false claim is reported only when the assigned ID misses its expected fact and another known fact matches. Unknown IDs, duplicates, mismatches, contradictions, and unsupported-pattern hits remain false-positive claims; expected IDs without credited true-positive claims are false negatives.

`claimPrecision = TP / (TP + FP)` and `factRecall = TP / (TP + FN)`; a metric is `null` when its denominator is zero. `formatCompliance` reports missing prefixes and unparsed lines separately from these fact metrics. The artifact includes claim lines, format errors, misses, contradictions, and unsupported-pattern hits. The negative pepper fact is expected; negated positive facts must not earn credit. The image position oracle accepts either “circle left of square” or the equivalent “square right of circle”; the opposite and explicitly negated relations do not match.

These are deliberately closed-set lexical checks, not semantic entailment or free-form claim extraction. Regexes cover only the documented fixture vocabulary. Paraphrases, scope, coreference, mixed statements, and negation can be misclassified; contradiction/unsupported detection and negation scope are heuristics, not exhaustive linguistic analysis. A high score does not prove arbitrary statements true, and a low score does not establish why a model failed.

### Hierarchy case

`sourceIds.recall` and `sourceIds.precision` measure exact retained source paragraph IDs (unique IDs); the artifact separately reports missing, extra, and duplicate IDs. The closed-set record evaluator counts a record as matched only when its expected source ID is cited by a section and that section's `keyPoints` contain regex matches for the record ID, site, field, value, and unit. `recordSignatureMetrics.factRecall` is matched expected records divided by all expected records. `recordSignatureMetrics.claimPrecision` is the fraction of emitted record-ID mentions that pass that exact signature-and-citation check; duplicate, unknown, unlinked, or mismatched record IDs do not pass.

Those record measures do not parse or certify arbitrary report prose, omitted facts outside the oracle, or the truth of claims that are not represented by the expected record signatures. Source-ID retention measures citation coverage, not factual correctness.
