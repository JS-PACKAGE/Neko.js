# Quality benchmark and gate

The benchmark runs the pinned SDK model against four fixed inputs: five garden facts, visible image geometry/colors, eight boundary ledger records, and 300 dated station observations. Raw SDK outputs/errors are retained. This finite-oracle regression suite is not generic semantic entailment, factuality proof, or prompt-injection protection.

## Run

Run `npm ci` first (its `prepare` builds the SDK). After source changes run `npm run build`; the benchmark and gate use the compiled pinned model manifest. Deterministic quality tests never download or execute a model.

```sh
MODEL_CACHE=/absolute/path/to/verified-node-cache
npm run quality:benchmark -- --runtime node --device cpu --model-profile default \
  --cache-dir "$MODEL_CACHE" --output artifacts/quality-node.json
npm run quality:gate -- --input artifacts/quality-node.json \
  --output artifacts/quality-node-diagnostics.json

# Deterministic evaluator, programmatic gate, comparison and CLI regression tests; no model
npm run test:quality

# Browser WebGPU, using a dedicated persistent Chromium directory
npm run quality:benchmark -- --runtime browser --device webgpu \
  --cache-dir "$MODEL_CACHE" --browser-profile .cache/neko-quality-chromium \
  --browser-port 4173 --output artifacts/quality-browser.json
```

Benchmark completion and throughput do **not** imply quality acceptance. The gate prints structured JSON, optionally writes it to `--output`, and exits nonzero for malformed, stale, incomplete, noncomparable or below-policy results. It re-evaluates raw SDK output rather than trusting saved scores. The default requires **all four** cases. `--cases text,image` deliberately scopes a diagnostic gate; it is not full release acceptance. Benchmark `--case text|image|boundaries|hierarchy` selects a single fixture; the default is `all`.

Node defaults to CPU; Browser requires WebGPU. Browser availability is not evidence of execution-provider correctness or successful model quality. No new platform verification is claimed here. Network is disabled unless `--allow-network` explicitly permits model downloads. There are no automatic retries. Missing verified offline assets block inference. Offline Browser runs require `--cache-dir`: a temporary loopback mirror verifies and seeds the selected pinned model files into the SDK CacheStorage, seeds the pinned WASM asset, then takes Chromium offline and blocks external routes. Artifacts preserve cache/network evidence. Model/cache integrity policy is unchanged.

Browser cache is origin-scoped: retain the same `--browser-port` and dedicated `--browser-profile` to reuse it. Default port is `4173`. Never use a personal browser profile. Without a profile the browser context is ephemeral. Other benchmark options: `--model-profile default|all-q4`, `--context-window-tokens N`, `--max-new-tokens N`, `--headed`, and Node-only `--profile-prefix PATH`. `--help` lists all options. Default text budget is 512 tokens for five separate facts, boundary budget 512, image budget 256; the text prompt explicitly requests all five assigned facts instead of a single summary.

Only public synthetic fixtures belong in uploaded artifacts. Raw prompts, model text and reports are untrusted. Never upload private user inputs, credentials, images or cache contents; see [Security](../../SECURITY.md).

## Versioned acceptance policy

Artifact `schemaVersion: 3`, fixture `quality-fixtures-v2`, evaluator `quality-claims-v5`, and policy `quality-thresholds-v1` are explicit contracts. Hierarchy artifacts now contain version-3 reports; the benchmark/gate targets the default pinned 0.8B model, not every registered model. The gate requires matching current oracle SHA-256, fixture hashes, effective case inputs/budgets and stimulus key. Missing/wrong versions, missing/duplicate cases, unattempted/failed cases and mismatched profile evidence fail closed. Run and raw-result model ID/revision must equal the pinned manifest; raw result and case backend evidence must agree with declared runtime/device. Old artifacts must be regenerated; relabeling versions is not a migration.

The policy in `scripts/quality/policy.mjs` declares:

| Scope | Minimum claim precision | Minimum fact recall | Maximum contradictions / unsupported / format errors |
| --- | --- | --- | --- |
| Text, image, boundary claims | 1.0 | 1.0 | 0 / 0 / 0 |
| Hierarchy retained source quotes | 1.0 | 1.0 | 0 / 0 / 0 |
| Hierarchy generated section record claims | 1.0 | 0.9 | 0 / 0 / 0 |
| Hierarchy overview/conclusion record mentions | Not required | Not required | 0 / 0 / 0 |

Where precision/recall thresholds apply, an empty-denominator (`null`) metric fails; it is not perfect quality. These are acceptance targets, **not measured current-model capabilities**. A completed model run with low recall, pink instead of red, contradictory numbers, injected text or lossy sections can fail. Do not lower targets to make the current model green.

## Scoring boundaries

Text/image lines use `FACT <ID>: <statement>`. Missing `FACT` affects format compliance separately from factual scoring. Unparsed nonempty lines are format errors; unknown/duplicate IDs and wrong signatures are false claims. True positives must match the assigned finite signature without contradiction/unsupported hits. Precision is `TP/(TP+FP)`; recall is `TP/(TP+FN)`. Supported extra known facts can appear as context; this is not unrestricted claim extraction. Garden negation and image-relative position rules remain vocabulary-bounded heuristics.

The boundary ledger is deliberately stricter: each assigned record must match one explicitly enumerated original-language sentence (NFC normalization; the temperature range permits an en dash or ASCII hyphen). It covers negative facts, three versus five varieties bound to different plots, both 18–22 °C endpoints, Traditional Chinese and Japanese entities/colors/counts, the accented name José, and a quoted malicious note that must remain data. Added clauses, swapped entities, wrong numbers or following the injected password instruction fail. Exact matching provides meaningful closed-set Unicode boundaries, not multilingual understanding or general injection resistance.

Hierarchy **retention and generation are separate**. `sourceFacts[].citation.quote` must equal the cited source paragraph slice; retained content receives credit only for full expected record tuples. Generated section `keyPoints` are scanned for the declared tuple syntax: `Observation H001: station S001 at Alder plot measured soil moisture 10 percent on 2025-01-01.` The word `Observation` is optional. Every record's code, station, site, field, value, unit and date must match together. The union of section records must meet 0.9 recall and perfect precision. `page.summary` and `conclusion` are scored separately for contradictory/unsupported tuples and malformed record mentions, without a full-catalog recall requirement: compressed overviews should not repeat all 300 records and cannot compensate for missing section facts. An H-ID mention without a parsed tuple is a format error. A wrong known tuple is a contradiction; an unknown tuple is unsupported. Repeated matching record mentions are reported but do not inflate recall or invalidate repeated correct content. Source/citation IDs **never** supply semantic signatures. A perfectly retained source ledger with empty or lossy generated sections fails generated recall.

Non-record hierarchy prose is not semantically certified. Tuple paraphrases outside this grammar may fail even when true, and prose without record IDs may be unscored. The gate remains a finite regression oracle, not proof that arbitrary report prose is supported. Version-3 report audits are conservative lexical evidence checks; tokenizer-aware constrained JSON decoding, provenance/schema validation and exact quote preservation neither fact-check summaries nor replace this independent quality gate.

## Comparison and diagnostics

`stimulusKey` identifies current encoded fixture inputs and effective budgets, not model quality or decoded pixels. Each artifact records model/revision/profile, requested/observed backend, runtime/environment, fixture hashes, raw outputs/errors and case inputs. `status: completed` and `comparable: true` describe execution only. Gate diagnostics expose `completion` separately from `quality` and include exact threshold failures and findings.

```sh
npm run quality:gate -- --input artifacts/new.json --baseline artifacts/prior.json \
  --comparison-mode strict --output artifacts/comparison.json
```

Strict comparison requires matching inputs, versions, model/revision, profile, runtime/device, observed backend, environment and browser evidence. `--comparison-mode paired-inputs` explicitly permits runtime/profile/backend differences but still requires the same model/revision, fixtures and encoded stimuli, with complete structurally valid artifacts. Both modes exit nonzero when either quality gate fails. Programmatic exports are `evaluateQualityArtifact(artifact, { requiredCases? })` and `compareQualityArtifacts(left, right, { mode? })` in `scripts/quality/gate.mjs`.

Paired-input comparison is **not** an isolated quantization or platform experiment: Node uses Sharp raw sRGB pixels without alpha while Browser uses browser decoding and RGBA Canvas pixels. Identical manifest PNG bytes do not guarantee identical decoded/preprocessed representations or SDK image `versionId`s. Differences must not be attributed solely to quantization.

## Current measured outcome

The latest macOS/arm64 Node 22.23.3 CPU/default offline run, using `quality-claims-v5`, attempted all four fixtures. Text (5 facts), image (3 facts) and boundaries (8 records) each scored precision/recall 1.0 with no finite-oracle contradictions or unsupported findings, but all omitted the required `FACT` prefix. The 300-paragraph hierarchy failed with `STRUCTURED_OUTPUT` (`Generated text is not one complete JSON value`). The full gate exited 1 with incomplete execution and quality failures. Passing contract tests, packed-consumer inference, constrained-output smokes and smaller reports does not meet the strict release-quality target.

This run's artifact records the hierarchy error, not its failed raw token stream or checkpoint; it does not establish the exact failure cause or stage. Constraints do not guarantee completion within a finite output budget, and incomplete output still fails closed. The gate thresholds have not been reduced.

**Historical validation-only diagnostic:** a prior version-2 run located a malformed escaped closing quote at `section:67`, after 149 of 512 output tokens. That older checkpoint retained all 300 exact quotes and 67 completed stages, with 33,490 input / 11,485 output tokens including the failed attempt; validated persistence passed. This is historical failure/provenance evidence, not the diagnosis of the current constrained run, successful 300-paragraph inference or a factuality guarantee.

## CI

Every push/PR runs deterministic `test:quality` in the existing CI verification job. The model-heavy `real-model-quality` job runs only on `workflow_dispatch`, using the existing macOS 15 / Node 22 CPU runtime, explicit network permission and a temporary model cache. It executes benchmark **and gate**; bad model results fail the job. Raw benchmark output and structured gate diagnostics are uploaded with `if: always()` even when quality fails. There is no publishing or new claimed platform certification.
