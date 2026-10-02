# Neko.js

## 繁體中文

Neko.js 是一個以 TypeScript 撰寫的本機多模態推理原型，可在 Node.js 或支援的瀏覽器中，用固定版本的 Qwen3.5 視覺模型回答使用者提供的圖片與提示。此原型目前提供真實圖片推理與後端／耗時觀測；**尚不是完整的網頁擷取與結構化網頁報告產品**。

模型固定為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，embedding 與 decoder 為 Q4，vision encoder 為 FP16。首次載入需下載約 871 MB 模型與處理器資產；本機快取以 pinned 清單的檔案大小和 SHA-256 驗證。推理使用 Transformers.js 4.2.0 / ONNX Runtime；不會自動切換所選後端。模型輸出仍可能出錯，請勿用於安全或權限判斷。

### Node.js

需要 Node.js 22 以上。

```sh
npm ci
npm run build
```

```js
import { createPrototype } from 'neko.js';

const prototype = await createPrototype({ device: 'webgpu' });
try {
  const result = await prototype.infer({
    image: './photo.png',
    prompt: 'Describe the image.',
    maxNewTokens: 128,
  });
  console.log(result.text);
  console.log(result.model, result.backend, result.timings, result.memory);
} finally {
  await prototype.dispose();
}
```

Node 原生 provider 是否可用取決於 ONNX Runtime 版本；此專案將 Transformers.js 使用的 `onnxruntime-node` 覆寫為 1.30.0（瀏覽器端 ONNX Runtime Web 未更動）。舊版 1.24.3 缺少模型所需的 `com.microsoft:CausalConvWithState`。Node 22.23.3、Transformers.js 4.2.0、runtime 1.30.0 與固定 Qwen revision 的實測 profile：embed 128 個 WebGPU 事件，vision 536 WebGPU + 50 CPU，decoder 169,472 WebGPU + 12,288 CPU，因此並非所有運算子都在 GPU。已填妥驗證快取：載入 6,414.542 ms、第一個非空輸出 1,487.161 ms、生成 9,353.701 ms、總推理 9,486.618 ms，網路 0。空快取冷啟動下載 13 檔、871,364,778 位元組；外層載入 37,118.820 ms，推理載入 37,114.206 ms、首個輸出 1,228.452 ms、生成 7,728.604 ms、總推理 7,796.944 ms。全新離線程序重用同一快取：外層載入 4,655.131 ms、推理載入 4,652.725 ms、首個輸出 2,447.610 ms、生成 9,874.194 ms、總推理 9,997.882 ms，網路 0。RSS 約為載入後 2.147 GB、推理中 1.966 GB、dispose 後 0.731 GB；GPU 記憶體未知。Node 僅選 CPU 的端對端路徑尚未驗證。詳見[使用說明](doc/zh/usage.md)與[安全性](SECURITY.md)。

### 瀏覽器

執行 `npm run build` 後，從安全的 HTTP(S) 靜態來源開啟 `examples/browser-prototype.html`。已用持久化 headed Chromium 153 的實際 demo UI 上傳圖片並完成 WebGPU 推理，畫面輸出：`A football player in a Paris Saint-Germain uniform is dribbling the ball while being pursued by a defender in a dark blue kit.` 瀏覽器回報 Apple／Metal 3、`fallback:false`、支援 `shader-f16`，所有已載入 ONNX session 設定為 WebGPU。這證明模型推理確實完成，但不代表每個運算子都在 GPU，也未量得 GPU 記憶體。此 UI run 的載入／前處理／第一個非空輸出／生成／總耗時為 5,238.6／271.6／3,963.9／4,717.6／4,989.2 ms。獨立的 headed browser smoke instrumentation 觀察到 2,656 次 `dispatchWorkgroups` 和 1,104 次 `queue.submit`。同一持久 profile 重載後啟用 `localFilesOnly`，輸出相同且未觀察到 Hugging Face 請求；Cache Storage 約 895 MB。明確選用的 CPU/WASM 選項標示為此固定模型不支援；實際 session 建立因缺少 `GatherBlockQuantized(1)` 實作而失敗，且不會 fallback。`npm run test:browser` 只涵蓋瀏覽器行為／backend 能力，不包含模型推理。

```ts
import { createPrototype } from './dist/neko.js';

const prototype = await createPrototype({ device: 'webgpu' });
const result = await prototype.infer({ image: selectedFile, prompt: 'Describe the image.' });
```

首次執行會從 Hugging Face 載入固定模型版本，並使用瀏覽器 Cache API 快取；用持久化 profile 可在重載後以 `localFilesOnly: true` 驗證離線推理，必要檔案不在快取時載入會失敗。上方已記錄 headed Chromium 的實際結果與限制。瀏覽器 URL 圖片仍受來源 CORS 政策限制；使用者選取的 `File` 不需要跨來源圖片請求。請勿在不可信頁面中載入此 demo。

### 開發與驗證

```sh
npm run build
npm run typecheck
npm run lint
npm test
npm run test:browser
```

`npm run test:browser` 使用 Playwright，需先安裝其 Chromium、Firefox 和 WebKit 瀏覽器。昂貴且可能下載約 871 MB 權重的真實 Chromium demo smoke **不屬於預設測試**；需明確執行 `npm run build && npm run smoke:browser`。`npm run smoke:browser -- cpu 8` 保留為明確 CPU/WASM 診斷 helper，但此固定模型在瀏覽器 CPU/WASM 下已確認因 `GatherBlockQuantized(1)` 無實作而於 session 建立失敗；它不是受支援的替代後端，也不會 fallback。若已有 Node cache，可將 `NEKO_MODEL_CACHE` 設為包含 `config.json` 與 `onnx/` 的固定模型 revision 目錄，避免重複下載；未設定時會照常使用固定 Hugging Face URL。此 smoke 使用生成的合成紅色方塊圖片，不是 Ollama 對照；相同 Ollama 測試圖片尚未提供。Node 推理和瀏覽器推理需分開驗證；本專案不宣稱跨平台效能或結果一致。

## English

Neko.js is a local multimodal inference prototype written in TypeScript. It runs a pinned Qwen3.5 vision model in Node.js or a supported browser to answer a user-supplied image and prompt. It provides real image inference and reports observed backend/timing data; **it is not yet a complete website extraction or structured website-report product**.

The model is pinned to `onnx-community/Qwen3.5-0.8B-ONNX-OPT`, revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`, with Q4 embeddings/decoder and an FP16 vision encoder. Initial loading downloads approximately 871 MB of model and processor assets. The pinned cache manifest verifies file sizes and SHA-256 digests. Inference uses Transformers.js 4.2.0 / ONNX Runtime and does not automatically switch the selected device. Model output can be wrong; do not use it for security or authorization decisions.

### Node.js

Requires Node.js 22 or newer.

```sh
npm ci
npm run build
```

```js
import { createPrototype } from 'neko.js';

const prototype = await createPrototype({ device: 'webgpu' });
try {
  const result = await prototype.infer({
    image: './photo.png',
    prompt: 'Describe the image.',
    maxNewTokens: 128,
  });
  console.log(result.text);
  console.log(result.model, result.backend, result.timings, result.memory);
} finally {
  await prototype.dispose();
}
```

Node provider availability depends on the ONNX Runtime version. This project overrides Transformers.js's native `onnxruntime-node` to 1.30.0; the browser's separate ONNX Runtime Web dependency is unchanged. The older 1.24.3 runtime lacks the model's `com.microsoft:CausalConvWithState` operator. Measurements used Node 22.23.3, Transformers.js 4.2.0, runtime 1.30.0, and the pinned Qwen revision. The profile recorded 128 WebGPU events for embed, 536 WebGPU + 50 CPU for vision, and 169,472 WebGPU + 12,288 CPU for decoder: not every operation ran on GPU. With a populated verified cache: load 6,414.542 ms, first non-empty output 1,487.161 ms, generation 9,353.701 ms, total inference 9,486.618 ms, zero network requests. A cold empty-cache run downloaded 13 files (871,364,778 bytes): outer load 37,118.820 ms, inference load 37,114.206 ms, first output 1,228.452 ms, generation 7,728.604 ms, total inference 7,796.944 ms. A fresh offline process using that cache: outer load 4,655.131 ms, inference load 4,652.725 ms, first output 2,447.610 ms, generation 9,874.194 ms, total inference 9,997.882 ms, zero network requests. Process RSS was about 2.147 GB after load, 1.966 GB during inference, and 0.731 GB after dispose; GPU memory is unknown. Node CPU-only end-to-end inference remains unverified. See the [English usage guide](doc/en/usage.md) and [Security](SECURITY.md).

### Browser

After `npm run build`, open `examples/browser-prototype.html` from a secure HTTP(S) static origin. An actual demo UI run in persistent headed Chromium 153 uploaded an image and completed WebGPU inference, rendering: `A football player in a Paris Saint-Germain uniform is dribbling the ball while being pursued by a defender in a dark blue kit.` The browser reported Apple/Metal 3, `fallback:false`, and `shader-f16`; all loaded ONNX session configurations used WebGPU. This demonstrates a completed model inference but does not establish that every operator ran on GPU; GPU memory was not measured. UI load/preprocess/first non-empty output/generation/total timings were 5,238.6/271.6/3,963.9/4,717.6/4,989.2 ms. Separate headed-browser smoke instrumentation observed 2,656 `dispatchWorkgroups` and 1,104 `queue.submit` calls. After reloading the same persistent profile with `localFilesOnly` enabled, inference produced the same output and no Hugging Face requests were observed; Cache Storage used about 895 MB. The explicit CPU/WASM option is labeled unsupported for this pinned model; the observed session creation failure is `GatherBlockQuantized(1)`, with no fallback. `npm run test:browser` covers browser behavior/backend capability, not model inference.

```ts
import { createPrototype } from './dist/neko.js';

const prototype = await createPrototype({ device: 'webgpu' });
const result = await prototype.infer({ image: selectedFile, prompt: 'Describe the image.' });
```

The first run loads the pinned model revision from Hugging Face and caches it with the browser Cache API. With a persistent profile, reloading with `localFilesOnly: true` verified offline inference; loading fails if required files are absent. The headed Chromium result and CPU limitation are described above. Browser image URLs remain subject to the source's CORS policy; a user-selected `File` does not need a cross-origin image request. Do not load this demo into an untrusted page.

### Development and verification

```sh
npm run build
npm run typecheck
npm run lint
npm test
npm run test:browser
```

`npm run test:browser` uses Playwright and requires its Chromium, Firefox, and WebKit browsers to be installed. The expensive real Chromium demo smoke, which may download approximately 871 MB of weights, is **not part of the default suite**. Explicitly run `npm run build && npm run smoke:browser`. `npm run smoke:browser -- cpu 8` remains an explicit CPU/WASM diagnostic helper, but this pinned model is confirmed to fail at browser CPU/WASM session creation because no `GatherBlockQuantized(1)` implementation is available; it is not a supported alternative backend and does not fall back. To reuse a Node cache, set `NEKO_MODEL_CACHE` to the model-revision directory containing `config.json` and `onnx/`; otherwise the smoke uses the pinned Hugging Face URLs normally. This smoke uses a generated synthetic red-square image, not an Ollama parity fixture; the same Ollama test image has not been supplied. Node and browser inference must be verified separately; this project makes no cross-platform performance or output-equivalence claim.

## License

The project is licensed under Apache-2.0. The model and runtime dependencies have their own licenses; review the upstream terms before use.
