# Changelog

## Unreleased

- Pin the Node native ONNX Runtime to 1.30.0 and use matching browser ONNX Runtime Web GPU/WASM assets.
- Verify the pinned Qwen model with real Node inference, including cold-cache download and fresh-process offline-cache runs; exact measurements are in the usage guides.
- Complete real WebGPU inference in the browser demo and verify offline reload in the same persistent profile with `localFilesOnly`. Label the browser CPU/WASM UI option and smoke helper as unsupported for this pinned model; observed session creation fails at `GatherBlockQuantized(1)` with no fallback.
- Preserve image extraction provenance and align extraction tests with observed browser image discovery behavior.
