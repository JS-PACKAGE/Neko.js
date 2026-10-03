# Neko.js

## 繁體中文

Neko.js 是本機多模態推理 SDK，以固定版本的 Qwen3.5 ONNX 模型提供文字／圖片推理，以及從網址或 HTML 產生結構化網頁報告。Node.js 與受支援的瀏覽器都可執行；瀏覽器需支援 WebGPU。這不是爬蟲，也不會執行擷取頁面的腳本。模型輸出可能不正確，請勿用於安全或授權判斷。

預設模型為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，embedding／decoder 使用 Q4，vision encoder 使用 FP16，首次完整下載約 871 MB。Registry 另提供固定 revision 的 Qwen3.5-2B，兩個模型均有 `default`／`all-q4` profile；不是任意 Hub 模型載入器。固定檔案會在使用前驗證大小和 SHA-256。執行時 bundle 內含 Transformers.js 4.2.0；Node 使用原生 ONNX Runtime 1.30.0。模型選擇與 revision 詳見[使用說明](doc/zh/usage.md)。

### Node.js 安裝

需要 Node.js 22.13 以上。本專案不發佈 npm；支援的 Git 倉庫用法是 clone 後在本機建置：

```sh
git clone https://github.com/JS-PACKAGE/Neko.js.git
cd Neko.js
npm ci
npm run build
```

- 使用方也可在本機 clone 並建置目前工作樹後執行 `npm pack`，產生含預建 `dist/` 的 tarball，再以 `npm install /absolute/path/to/<printed-filename>.tgz` 安裝印出的檔案；檔名與版本取自當前 `package.json`。本專案不發佈至 npm。npm 11 的 Git dependency `prepare` 可能需要腳本核准，請使用明確的 clone／`npm ci`／`npm run build` 流程。Node bundle 包含 Transformers.js；直接執行依賴為 pinned ONNX Runtime、sharp、parse5、PDF.js 與預建 native canvas。原生 postinstall 僅核准 `onnxruntime-node@1.30.0`、`sharp@0.35.4`；canvas 不需新增腳本核准，勿一律核准所有依賴。

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

首次使用可先以 `cache.model.prefetch()` 建立驗證快取，再用相同 model／profile、`localFilesOnly: true` 離線推理；也可匯出／匯入經驗證的離線 bundle。純文字 `planInference()` 只載入 tokenizer／設定，不建立 ONNX session；圖片規劃仍需前處理。`inferStructured()` 預設使用 tokenizer-aware JSON grammar 約束明確 schema 子集，完整 Draft-07 驗證須明確選用 `structuredMode: 'validation-only'`；兩者都會 runtime 驗證，不保證事實正確。

- 有限緩衝的 `inferStream()` AsyncIterable、可分支／重設／持久化的對話 session；目前只重用精確相同 prompt 的 tokenization，**不重用 model KV cache**。
- 報告／checkpoint 版本 3，支援 extractive 零推理模式、`planReport()`、累計預算、明確階段重試、型別化部分報告與增加授權後續跑。Claim–evidence audit 是保守詞彙檢查，不是語意／真偽認證。
- 可選主內容擷取、表格與段落關聯、帶精確引用的文件問答、圖片 ROI／tiling 與有界前處理快取。
- 離線 bundle、安裝／配額診斷、worker health／restart／硬期限；硬期限會終止同一 worker 所有待處理請求，需明確 restart，不自動重播。

完整契約與可執行範例見[繁體中文使用說明](doc/zh/usage.md)。

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

最新變更通過 130 個 Node 契約、27 個品質工具、21 個 browser 契約測試，以及 packed consumer 真實 CPU 文字／圖片／報告推理。macOS／arm64 Node 22.23.3 實測串流、session、約束生成、ROI、恢復及完整離線 bundle；2B 的兩種 profile 也完成真實 Node 文字／圖片／報告。Chromium 153 WebGPU 另通過 0.8B 完整 bundle 冷匯入後離線推理、預算續跑與硬期限／restart。[平台矩陣](doc/zh/usage.md#後端相容性)區分最新與歷史證據；browser 使用測試旗標，未主張 Windows／Linux、其他瀏覽器、2B browser 或各 driver parity。Provider 組態不是所有算子的 GPU 證據；[完整品質 gate 仍未通過](doc/zh/quality.md#目前實測結果)，不以 schema／引用／audit 取代品質驗收。

Node 網址擷取可能產生 SSRF 風險；對不可信 URL 必須自行設定 `validateDestination` 和 outbound network policy。HTML 與模型輸出均是不可信資料。更多威脅模型見 [SECURITY.md](SECURITY.md)。

## English

Neko.js is a local multimodal inference SDK. It uses a pinned Qwen3.5 ONNX model for text/image inference and structured website reports from a URL or HTML. It runs in Node.js and supported WebGPU browsers. It is not a crawler and does not execute page scripts. Model output may be wrong; do not use it for security or authorization decisions.

The default is `onnx-community/Qwen3.5-0.8B-ONNX-OPT`, revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`, with Q4 embeddings/decoder and an FP16 vision encoder; a full first download is about 871 MB. The registry also provides an immutable Qwen3.5-2B revision, with `default` and `all-q4` profiles for both models—not arbitrary Hub model loading. Each pinned asset is size- and SHA-256-verified before use. Transformers.js 4.2.0 is bundled; Node uses native ONNX Runtime 1.30.0. See the [usage guide](doc/en/usage.md) for model selection and revisions.

### Install for Node.js

Requires Node.js 22.13 or newer. This project is not published to npm. Clone the Git repository and build locally:

```sh
git clone https://github.com/JS-PACKAGE/Neko.js.git
cd Neko.js
npm ci
npm run build
```

Consumers may clone and build the working tree, then run `npm pack` and install the printed tarball with `npm install /absolute/path/to/<printed-filename>.tgz`. Its name/version derives from `package.json`; this project is not published to npm. npm 11 may require approval for Git dependency `prepare`, so use explicit clone, `npm ci`, and `npm run build`. The Node bundle includes Transformers.js; direct runtime dependencies are pinned ONNX Runtime, sharp, parse5, PDF.js, and prebuilt native canvas. Approve native postinstall only for `onnxruntime-node@1.30.0` and `sharp@0.35.4`; canvas needs no additional script approval. Never blanket-approve dependency scripts.

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

First prefetch a verified cache, then infer with the same model/profile and `localFilesOnly: true`, or import a verified offline bundle. Text-only `planInference()` loads verified tokenizer/configuration assets without creating ONNX sessions; image planning still requires preprocessing. `inferStructured()` defaults to tokenizer-aware JSON grammar for an explicit schema subset; full Draft-07 validation requires explicit `structuredMode: 'validation-only'`. Both modes validate at runtime without factual guarantees.

- Bounded `inferStream()` AsyncIterable and transactional conversation sessions with branching/reset/persistence. Reuse is exact-prompt tokenization only, **not model KV-cache reuse**.
- Version-3 reports/checkpoints, zero-inference extractive mode, `planReport()`, cumulative budgets, explicit stage retries, typed partial reports and increased-authorization resume. Claim–evidence audits are conservative lexical checks, not semantic or truth certification.
- Opt-in main-content extraction, tables/paragraph relations, citation-grounded document questions, image ROI/tiling and bounded preprocessing caches.
- Offline bundles, installation/quota diagnostics and worker health/restart/hard deadlines. A hard deadline terminates all pending work in that worker; explicit restart is required, with no automatic replay.

See the [English usage guide](doc/en/usage.md) for complete contracts and examples.

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

The latest changes pass 130 Node contract, 27 quality-tool and 21 browser contract tests plus real packed-consumer CPU text/image/report inference. Fresh macOS/arm64 Node 22.23.3 runs exercise streams, sessions, constrained generation, ROI, recovery and full offline bundles; both 2B profiles complete real Node text/image/report inference. Chromium 153 WebGPU additionally passes 0.8B offline inference after full-bundle cold import, budget resume and hard deadline/restart. The [platform matrix](doc/en/usage.md#backend-compatibility) separates current and historical evidence; browser runs use a test flag, without Windows/Linux, other-browser, 2B-browser or driver parity claims. Provider configuration is not per-operator GPU proof. The [full quality gate still fails](doc/en/quality.md#current-measured-outcome); schemas, citations and audits do not replace quality acceptance.

Node URL extraction can create SSRF risk. For untrusted URLs, provide `validateDestination` and application-level outbound network controls. Treat HTML and model output as untrusted. See [SECURITY.md](SECURITY.md) for the threat model.

## License

Apache-2.0. The model and runtime dependencies have their own licenses; review upstream terms before use.
