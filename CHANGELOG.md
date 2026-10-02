# Changelog

## Unreleased

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
