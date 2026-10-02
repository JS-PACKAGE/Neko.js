# Neko.js

## 繁體中文

Neko.js 是本機多模態推理 SDK，以固定版本的 Qwen3.5 ONNX 模型提供文字／圖片推理，以及從網址或 HTML 產生結構化網頁報告。Node.js 與受支援的瀏覽器都可執行；瀏覽器需支援 WebGPU。這不是爬蟲，也不會執行擷取頁面的腳本。模型輸出可能不正確，請勿用於安全或授權判斷。

模型為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，embedding／decoder 使用 Q4，vision encoder 使用 FP16。首次使用約需下載 871 MB；固定模型檔案會在使用前驗證大小和 SHA-256。執行時 bundle 內含 Transformers.js 4.2.0；Node 使用原生 ONNX Runtime 1.30.0。

### Node.js 安裝

需要 Node.js 22 以上。本專案不發佈 npm；支援的 Git 倉庫用法是 clone 後在本機建置：

```sh
git clone https://github.com/YueyuHoshizora/Neko.js.git
cd Neko.js
npm ci
npm run build
```

- 使用方也可在本機 clone 並建置目前工作樹後執行 `npm pack`，產生含預建 `dist/` 的 tarball，再安裝該檔案。tarball 的 `1.1.0` 套件版本反映工作樹的 `package.json`；本專案不發佈至 npm。因 npm 11 對 Git 相依套件的 `prepare` 有腳本核准限制，不要假設直接安裝 Git URL 會完成建置；clone、`npm ci`、`npm run build` 的本機流程仍支援。Node bundle 已包含 Transformers.js；執行依賴為 pinned `onnxruntime-node@1.30.0`、`sharp@0.35.4`、`parse5@8.0.1`。若 npm 11 封鎖原生安裝腳本，僅核准這兩個有原生程式碼的精確版本，不要核准所有套件腳本。

```js
import { createNeko } from 'neko.js';
import { realpath } from 'node:fs/promises';

const photo = await realpath('./photo.png');
const neko = await createNeko({ device: 'cpu', policy: { localFiles: (path) => path === photo } });
try {
  const answer = await neko.infer({
    image: './photo.png', // 可省略以進行純文字推理
    prompt: 'Describe the visible scene.',
    maxNewTokens: 128,
  });
  console.log(answer.text, answer.usage);

  const report = await neko.describe('<p>The sky is blue.</p>', {
    language: 'zh-TW',
    format: 'json',
    imageFailurePolicy: 'omit',
  });
  console.log(report.page.summary, report.images);
} finally {
  await neko.dispose();
}
```

模型採懶載入。Node 預設快取位於 macOS `~/Library/Caches/neko.js`、Windows `%LOCALAPPDATA%/neko.js`、Linux `$XDG_CACHE_HOME/neko.js` 或 `~/.cache/neko.js`；可用 `cacheDir` 覆寫。瀏覽器使用 Cache Storage。`neko.cache.model.prefetch/status/clear` 管理固定模型檔案，`neko.cache.engine.status/release` 管理載入的引擎。詳見[繁體中文使用說明](doc/zh/usage.md)。

### 瀏覽器

建置後從安全的 HTTP(S) 來源提供 `examples/browser-prototype.html`。Browser bundle 位於 `dist/browser/`，ONNX WASM 資產隨 bundle 一起輸出。瀏覽器 CPU/WASM 不支援此模型：ONNX Runtime Web 缺少 `GatherBlockQuantized(1)`，Neko 會回報錯誤，不會自動 fallback。瀏覽器 URL 圖片仍受 CORS 限制。

首次線上模型下載請使用經驗證的 `modelSource: { baseUrl }` mirror／broker；瀏覽器隱藏 Hugging Face 跨來源 redirect 時會 fail closed，不會繞過逐 hop 核准。設定及 loopback helper 見[使用說明](doc/zh/usage.md#瀏覽器模型來源)。

### 驗證與安全

```sh
npm run build
npm run typecheck
npm run lint
npm test
npm run test:package:artifact
npm run test:browser
```

昂貴的真實模型驗證明確執行：`npm run test:package` 會在 packed consumer 實際執行 Node 原生 CPU 文字、路徑圖片及 HTML 報告推理；`npm run smoke:browser` 透過瀏覽器 demo 驗證圖片、文字和報告流程。冷快取可能下載約 871 MB。兩個 script 的 `NEKO_MODEL_CACHE` 路徑層級不同：`test:package` 接受 SDK 快取根目錄（其下含 model ID／revision）；`smoke:browser` 接受該 revision 目錄本身（直接含 `tokenizer.json`、`onnx/`）。以下以 macOS 預設快取為例。觀察到的 pinned model/runtime pipeline 曾把飽和紅／藍色描述為粉紅／紫色；圖片描述不應視為可靠事實，這也不是模型單獨品質或 Ollama parity 的結論。後端/session 組態不是所有算子都在 GPU 執行的證據。

```sh
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js" npm run test:package
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0" npm run smoke:browser -- webgpu 16 --headed --offline-reload
```

瀏覽器 smoke 需要實際可用的 WebGPU adapter；headed 不保證裝置可用。runner 的本機靜態伺服器仍須可連線；`--offline-reload` 只封鎖模型下載，不代表整個網站離線可用。

Node 網址擷取可能產生 SSRF 風險；對不可信 URL 必須自行設定 `validateDestination` 和 outbound network policy。HTML 與模型輸出均是不可信資料。更多威脅模型見 [SECURITY.md](SECURITY.md)。

## English

Neko.js is a local multimodal inference SDK. It uses a pinned Qwen3.5 ONNX model for text/image inference and structured website reports from a URL or HTML. It runs in Node.js and supported WebGPU browsers. It is not a crawler and does not execute page scripts. Model output may be wrong; do not use it for security or authorization decisions.

The model is `onnx-community/Qwen3.5-0.8B-ONNX-OPT`, revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`, with Q4 embeddings/decoder and an FP16 vision encoder. First use downloads about 871 MB; each pinned model asset is size- and SHA-256-verified before use. Transformers.js 4.2.0 is bundled; Node uses native ONNX Runtime 1.30.0.

### Install for Node.js

Requires Node.js 22 or newer. This project is not published to npm. Clone the Git repository and build locally:

```sh
git clone https://github.com/YueyuHoshizora/Neko.js.git
cd Neko.js
npm ci
npm run build
```

Consumers may clone and build the current working tree locally, then run `npm pack` to create a tarball containing the prebuilt `dist/`. The `1.1.0` package version in that locally built archive comes from the working tree's `package.json`. This project is not published to npm. Do not assume an npm Git-URL dependency runs `prepare`: npm 11 may require lifecycle-script approval. The supported Git workflow is clone, `npm ci`, and `npm run build`. The Node bundle includes Transformers.js; runtime dependencies are pinned `onnxruntime-node@1.30.0`, `sharp@0.35.4`, and `parse5@8.0.1`. If npm 11 blocks native install scripts, approve only those exact native package versions rather than allowing all dependency scripts.

```js
import { createNeko } from 'neko.js';
import { realpath } from 'node:fs/promises';

const photo = await realpath('./photo.png');
const neko = await createNeko({ device: 'cpu', policy: { localFiles: (path) => path === photo } });
try {
  const answer = await neko.infer({
    image: './photo.png', // omit for text-only inference
    prompt: 'Describe the visible scene.',
    maxNewTokens: 128,
  });
  console.log(answer.text, answer.usage);

  const report = await neko.describe('<p>The sky is blue.</p>', {
    language: 'en',
    format: 'json',
    imageFailurePolicy: 'omit',
  });
  console.log(report.page.summary, report.images);
} finally {
  await neko.dispose();
}
```

Model loading is lazy. The default Node cache is macOS `~/Library/Caches/neko.js`, Windows `%LOCALAPPDATA%/neko.js`, or Linux `$XDG_CACHE_HOME/neko.js` / `~/.cache/neko.js`; `cacheDir` overrides it. Browsers use Cache Storage. `neko.cache.model.prefetch/status/clear` manages only the pinned model files; `neko.cache.engine.status/release` manages the loaded engine. See the [English usage guide](doc/en/usage.md).

### Browser

After building, serve `examples/browser-prototype.html` from a secure HTTP(S) origin. Browser bundles and their ONNX WASM assets are emitted under `dist/browser/`. Browser CPU/WASM is unsupported for this model because ONNX Runtime Web lacks `GatherBlockQuantized(1)`; Neko reports an error and does not automatically fall back. Browser image URLs remain subject to CORS.

For first-online model downloads use a verified `modelSource: { baseUrl }` mirror/broker. Concealed Hugging Face cross-origin redirects fail closed rather than bypassing per-hop approval. See the [usage guide and loopback helper](doc/en/usage.md#browser-model-sources).

### Verification and security

```sh
npm run build
npm run typecheck
npm run lint
npm test
npm run test:package:artifact
npm run test:browser
```

Expensive real-model verification is explicit: `npm run test:package` performs actual Node native CPU text, path-image, and HTML-report inference in a packed consumer; `npm run smoke:browser` exercises image, text, and report flows through the browser demo. A cold run may download approximately 871 MB. The scripts interpret `NEKO_MODEL_CACHE` differently: `test:package` expects the SDK cache root containing the model ID/revision directories; `smoke:browser` expects the revision directory itself, directly containing `tokenizer.json` and `onnx/`. The examples below use the macOS default cache. The pinned model/runtime pipeline described saturated red/blue regions as pink or pinkish-red in observed runs, so generated image descriptions are not reliable ground truth. This observation does not isolate the model or establish Ollama parity. Backend/session configuration does not prove that every operator ran on GPU.

```sh
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js" npm run test:package
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0" npm run smoke:browser -- webgpu 16 --headed --offline-reload
```

The browser smoke requires an available WebGPU adapter; headed mode does not guarantee one. The runner's local static server must remain reachable; `--offline-reload` blocks model downloads, not all site-network traffic or the need for that server.

Node URL extraction can create SSRF risk. For untrusted URLs, provide `validateDestination` and application-level outbound network controls. Treat HTML and model output as untrusted. See [SECURITY.md](SECURITY.md) for the threat model.

## License

Apache-2.0. The model and runtime dependencies have their own licenses; review upstream terms before use.
