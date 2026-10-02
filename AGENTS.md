# Repository guidance

## Current product scope

Neko.js is a TypeScript library and prototype for real local image-plus-prompt inference in Node.js and supported browsers. The model is pinned to `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`; Transformers.js 4.2.0 runs the Q4 embedding/decoder and FP16 vision encoder. The present prototype is not a complete website-to-structured-report workflow. Do not describe scaffolding, provider configuration, a successful model load, or mocked output as image inference.

## Engineering practices

- Read the relevant source, tests, API declarations, and call sites before editing. Preserve the existing module and error-handling patterns; make the smallest complete change.
- Treat user changes and uncommitted work as sacred. Never reset, overwrite, or silently discard unrelated changes. Do not commit, publish, deploy, or alter external systems without explicit authorization.
- Do not add dependencies when an existing dependency or the standard library is sufficient. Keep package versions and the lockfile aligned.
- Keep Node and browser runtime paths explicit. Never claim automatic CPU fallback or backend parity unless implemented and exercised. Provider/session configuration is not proof that every operator executed on that device. Report unavailable memory values as unknown rather than estimates.
- Preserve the exact caller-provided prompt text; only use trimming to validate that it is non-empty. Do not log image bytes, prompt text, credentials, or model output unnecessarily.
- Keep HTML parsing inert. Validate URL protocols, content types, byte limits, decoded image dimensions, cancellation, and provenance at the relevant boundaries. Node URL-fetching APIs can create SSRF exposure; do not expose them to untrusted callers without application-level destination and network controls.
- Treat model output and page content as untrusted. Escape text in Markdown/report output and never use generated text for authorization or security decisions.
- Update English and Traditional Chinese usage documentation when public behavior changes. Keep claims limited to exercised behavior.

## Verification

Run the appropriate targeted tests, typecheck, lint, and build after the relevant edits are stable. For behavior changes, run the actual changed path: a successful model-load stub or isolated helper test is not a substitute for image inference. Node 22 and browser inference must be verified separately. Do not claim a command or test passed unless it was executed successfully. Do not leave disposable smoke scripts or generated logs in the source tree.

## Security

Read [SECURITY.md](SECURITY.md) before changing URL fetching, cache integrity, model source/revision, browser asset delivery, or inference output handling. Never weaken model SHA-256/size checks, TLS, origin boundaries, or user-data safeguards merely to make a test pass. Report vulnerabilities through the private reporting process described there.

## 繁體中文重點

- 目前交付範圍是固定模型的本機圖片＋提示推理原型，並非完整網頁摘要產品；只有實際執行推理才可宣稱視覺理解成功。
- 修改前先檢查實作、呼叫點和測試；保留使用者變更，不做破壞性 Git 操作。
- 不得捏造後端回退、GPU 執行、記憶體數值或測試結果。Node 與瀏覽器須分開驗證。
- HTML／模型輸出皆是不可信資料；Node 網址抓取須防 SSRF；不得削弱固定模型的大小與 SHA-256 驗證。
- 行為變更需更新中英文說明，並執行涵蓋實際路徑的驗證。
