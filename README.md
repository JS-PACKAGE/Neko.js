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

- 使用方也可在本機 clone 並建置目前工作樹後執行 `npm pack`，產生含預建 `dist/` 的 tarball，再以 `npm install /absolute/path/to/<printed-filename>.tgz` 安裝印出的檔案；檔名與版本取自當前 `package.json`，不使用固定版本檔名。本專案不發佈至 npm。因 npm 11 對 Git 相依套件的 `prepare` 有腳本核准限制，不要假設直接安裝 Git URL 會完成建置；clone、`npm ci`、`npm run build` 的本機流程仍支援。Node bundle 已包含 Transformers.js；執行依賴為 pinned `onnxruntime-node@1.30.0`、`sharp@0.35.4`、`parse5@8.0.1`。若 npm 11 封鎖原生安裝腳本，僅核准這兩個有原生程式碼的精確版本，不要核准所有套件腳本。

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

首次使用先以 `cache.model.prefetch()` 連線建立驗證快取，`dispose()` 後再以相同 `cacheDir`／profile 和 `localFilesOnly: true` 在全新程序推理。完整可執行範例見[Node 首次使用](doc/zh/usage.md#node-首次建立快取再離線推理)。`planInference()` 提供精確輸入（含圖片展開）／輸出／上下文預算；有效冷請求會載入模型，不是無載入成本的 tokenizer-only 規劃。結構化生成以 JSON 邊界停止及 Draft-07 runtime 驗證，不是 schema grammar 約束，也不修補／重試。報告 `schemaVersion: 2` 保存精確來源引用 ledger 與 coverage；checkpoint `version: 2`／`evidence-first-v2` 的持久化 helper 拒絕舊版／未知版本，不自動遷移。來源涵蓋不等於生成 claims 已查核。

### 瀏覽器

建置後從安全的 HTTP(S) 來源提供 `examples/browser-prototype.html`。Browser bundle 位於 `dist/browser/`，ONNX WASM 資產隨 bundle 一起輸出。瀏覽器 CPU/WASM 不支援此模型：ONNX Runtime Web 缺少 `GatherBlockQuantized(1)`，Neko 會回報錯誤，不會自動 fallback。瀏覽器 URL 圖片仍受 CORS 限制。

首次線上模型下載請使用經驗證的 `modelSource: { baseUrl }` mirror／broker；瀏覽器隱藏 Hugging Face 跨來源 redirect 時會 fail closed，不會繞過逐 hop 核准。[首次使用流程](doc/zh/usage.md#瀏覽器首次使用流程)涵蓋 Node prefetch、現有 runner 自建 server／mirror、同 origin reverse proxy、runtime asset MIME／佈局和精確 CORS origin。隨附 demo 沒有 mirror URL 欄位，空快取時單純開啟不代表首次下載已設定。

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

平台 API／契約、歷史實測和目前全新驗證分開列於[平台矩陣](doc/zh/usage.md#後端相容性)。Node `>=22` 是套件要求；macOS／arm64 的 Node 22.23.3 與 26.7.0 已完成全新 CPU smoke；完整 Chromium 153 headless WebGPU 亦完成真實推理，但使用測試旗標。Windows／Linux、其他瀏覽器與各平台 GPU driver 實際推理未獲跨平台驗證，不主張 parity。Provider 組態不等於 GPU 執行證據；離線、schema 和 provenance 均不承諾模型品質。

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

Consumers may clone and build the current working tree locally, then run `npm pack` to create a tarball containing the prebuilt `dist/`. Install the printed filename with `npm install /absolute/path/to/<printed-filename>.tgz`; its name/version derives from the current `package.json`, not a fixed version in this guide. This project is not published to npm. Do not assume an npm Git-URL dependency runs `prepare`: npm 11 may require lifecycle-script approval. The supported Git workflow is clone, `npm ci`, and `npm run build`. The Node bundle includes Transformers.js; runtime dependencies are pinned `onnxruntime-node@1.30.0`, `sharp@0.35.4`, and `parse5@8.0.1`. If npm 11 blocks native install scripts, approve only those exact native package versions rather than allowing all dependency scripts.

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

First prefetch the verified cache online, dispose that instance, then infer in a fresh process with the same `cacheDir`/profile and `localFilesOnly: true`; see the [complete first-use example](doc/en/usage.md#first-use-node-cache-then-offline-inference). `planInference()` reports exact input (including image expansion), output and context budgets; valid cold calls load the model, so planning is not tokenizer-only or load-free. Structured generation uses JSON boundary stopping and Draft-07 runtime validation, not schema grammar constraints, repair or retry. Report `schemaVersion: 2` preserves an exact source-quote ledger and coverage; persistence helpers reject older/unknown report and checkpoint (`version: 2` / `evidence-first-v2`) schemas without automatic migration. Source coverage does not fact-check generated claims.

### Browser

After building, serve `examples/browser-prototype.html` from a secure HTTP(S) origin. Browser bundles and their ONNX WASM assets are emitted under `dist/browser/`. Browser CPU/WASM is unsupported for this model because ONNX Runtime Web lacks `GatherBlockQuantized(1)`; Neko reports an error and does not automatically fall back. Browser image URLs remain subject to CORS.

For first-online model downloads use a verified `modelSource: { baseUrl }` mirror/broker. Concealed Hugging Face cross-origin redirects fail closed rather than bypassing per-hop approval. The [first-use workflow](doc/en/usage.md#first-use-browser-workflow) covers Node prefetch, the existing runner's automatic server/mirror, same-origin reverse proxying, runtime-asset MIME/layout and exact-origin CORS. The shipped demo has no mirror URL field; simply opening it with an empty cache does not configure first-online downloads.

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

The [platform matrix](doc/en/usage.md#backend-compatibility) separates API/contracts, historical inference and fresh verification. Node `>=22` is the package requirement; fresh macOS/arm64 CPU smokes passed on Node 22.23.3 and 26.7.0, and full Chromium 153 headless WebGPU completed real inference with a test flag. Windows/Linux, other browsers and platform-specific GPU drivers lack cross-platform real-inference verification here; no parity is claimed. Provider configuration is not GPU execution proof; offline use, schemas and provenance do not promise model quality.

Node URL extraction can create SSRF risk. For untrusted URLs, provide `validateDestination` and application-level outbound network controls. Treat HTML and model output as untrusted. See [SECURITY.md](SECURITY.md) for the threat model.

## License

Apache-2.0. The model and runtime dependencies have their own licenses; review upstream terms before use.
