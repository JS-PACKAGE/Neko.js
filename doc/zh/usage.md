# 使用說明

## 目前範圍

Neko.js 目前提供真實的本機「單張圖片＋提示」推理原型，尚不是完整的網頁摘要工作流程。`createPrototype()` 對使用者提供的一張圖片和一段提示執行推理；頁面抽取與報告渲染是獨立工具，目前尚未串接成完整網頁摘要器。模型輸出可能不正確，請勿用於安全、授權或其他重大決策。

模型固定為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，embedding 與 decoder 使用 Q4，vision encoder 使用 FP16。推理使用 Transformers.js 4.2.0 和 ONNX Runtime。首次載入會從 Hugging Face 取得模型資產；使用或快取前會依固定清單檢查檔案大小和 SHA-256。

## 安裝與建置

- 需要 Node.js 22 以上，套件使用 ESM。
- `npm ci` 安裝鎖定的依賴。為支援此模型的 `com.microsoft:CausalConvWithState` 運算子，Node 路徑將 Transformers.js 的原生 `onnxruntime-node` 覆寫為 1.30.0；瀏覽器推理仍使用原本獨立鎖定的 ONNX Runtime Web，並未更動。`npm run build` 將 JavaScript、型別宣告、瀏覽器 bundle 及 ONNX Runtime 瀏覽器 WASM 資產輸出至 `dist/`。
- `npm test`、`npm run typecheck`、`npm run lint` 分別執行 Node 測試、TypeScript 型別檢查和 ESLint。
- `npm run test:browser` 以 Playwright 執行 Chromium、Firefox、WebKit 測試；執行前需安裝 Playwright 瀏覽器。
- `examples/browser-prototype.html` 是靜態瀏覽器範例；建置後需從安全的 HTTP(S) 來源提供。此專案不提供應用程式 HTTP server。

`npm run smoke:browser` 會以生成的紅色方塊圖片及保留空白的提示開啟真實 Chromium demo；預設明確選擇 WebGPU。`cpu`／WASM helper 模式仍保留供診斷使用，但此固定模型不支援瀏覽器 CPU/WASM：session 建立已確認會因缺少 `GatherBlockQuantized(1)` 實作而失敗，且不會 fallback。若要重用本機 Node 快取、避免重複下載，可將 `NEKO_MODEL_CACHE` 設為含有 `config.json` 與 `onnx/` 的模型 revision 目錄；未設定則照常使用固定 Hugging Face URL。`--headed` 會使用全新的隔離 persistent profile 啟動 Chromium；`--offline-reload` 則先執行再重新載入 UI，同時封鎖 Hugging Face 模型請求。真實推理可能下載約 871 MB 模型資產；此長時間、需網路的 smoke 不屬於 `npm run test:browser` 預設測試。若只想查看 Chromium WebGPU adapter、而不載入權重，可執行 `npm run smoke:browser -- webgpu 8 --enable-unsafe-webgpu --preflight-only`；這只提供 capability 資訊，不會推理，也不代表使用硬體執行。

## 實測結果

量測使用 Node.js 22.23.3、Transformers.js 4.2.0、原生 ONNX Runtime 1.30.0 及固定 Qwen revision。Hybrid provider profile 記錄 embedding 128 個 WebGPU 事件、vision 536 WebGPU + 50 CPU、decoder 169,472 WebGPU + 12,288 CPU；這不代表全部運算均在 GPU 執行。

- 已填妥驗證快取：模型載入 6,414.542 ms、第一個非空 decoded 輸出 1,487.161 ms、生成 9,353.701 ms、推理總計 9,486.618 ms；網路請求 0。
- 空快取冷啟動：下載 13 個檔案／871,364,778 位元組；外層載入 37,118.820 ms、推理載入 37,114.206 ms、首個輸出 1,228.452 ms、生成 7,728.604 ms、推理總計 7,796.944 ms。
- 全新離線程序重用相同快取：外層載入 4,655.131 ms、推理載入 4,652.725 ms、首個輸出 2,447.610 ms、生成 9,874.194 ms、推理總計 9,997.882 ms；網路請求 0。

`loadMs` 包含快取預取／驗證及 processor／模型載入；首個輸出時間不是原始 token 或 kernel 延遲。Process RSS 約為載入後 2.147 GB、推理中 1.966 GB、dispose 後 0.731 GB。GPU 記憶體未知；以上是實測而非跨平台效能保證。


## 圖片推理

Node.js 範例：

```js
import { createPrototype } from 'neko.js';

const model = await createPrototype({ device: 'webgpu' });
try {
  const result = await model.infer({
    image: './photo.png',
    prompt: 'Describe the image.',
    maxNewTokens: 128,
  });
  console.log(result.text);
  console.log(result.model, result.backend, result.timings, result.memory);
} finally {
  await model.dispose();
}
```

瀏覽器範例：

```js
import { createPrototype } from './dist/neko.js';

const model = await createPrototype({ device: 'webgpu' });
try {
  const result = await model.infer({ image: fileInput.files[0], prompt: 'Describe the image.' });
  output.textContent = result.text;
  details.textContent = JSON.stringify({ model: result.model, backend: result.backend, timings: result.timings, memory: result.memory }, null, 2);
} finally {
  await model.dispose();
}
```

`createPrototype(options?)` 會在回傳前載入固定版本的 processor 和模型。可設定 `device: 'cpu' | 'webgpu'`、`localFilesOnly`、`progressCallback`；瀏覽器可設定 `wasmPaths`。Node.js 可用 `cacheDir` 指定原生快取目錄。瀏覽器使用 Cache API，需在安全環境中執行，並依網站網路／CSP 政策允許模型來源。`localFilesOnly: true` 禁止網路下載，必要檔案不在快取時會明確失敗。WebGPU 無法使用時會回報錯誤，不會靜默切換後端；CPU 是明確選項，不會在 WebGPU 失敗時自動切換。

已用持久化 headed Chromium 153 的實際 demo UI 選取 WebGPU、上傳足球圖片並產生輸出：`A football player in a Paris Saint-Germain uniform is dribbling the ball while being pursued by a defender in a dark blue kit.` 瀏覽器回報 Apple／Metal 3、`fallback:false`、支援 `shader-f16`；所有已載入 session 設定皆使用 WebGPU。載入／前處理／第一個非空輸出／生成／總耗時為 5,238.6／271.6／3,963.9／4,717.6／4,989.2 ms。獨立 headed-browser smoke instrumentation 觀察到 2,656 次 `dispatchWorkgroups` 及 1,104 次 `queue.submit`。同一持久 profile 重載後啟用 `localFilesOnly`，產生相同輸出且未觀察到 Hugging Face 請求；Cache Storage 約 895 MB。這證明已完成實際推理，但不代表所有運算子均在 GPU，亦未量得 GPU 記憶體。瀏覽器 CPU/WASM 路徑已確認不支援此固定模型：session 建立因缺少 `GatherBlockQuantized(1)` 實作而失敗，demo 不會 fallback。

`infer({ image, prompt, maxNewTokens? })` 接受 Node.js 圖片路徑或支援的圖片輸入；瀏覽器可將使用者選取的 `Blob`／`File` 傳入。提示不可為空白。`maxNewTokens` 預設 128，必須是 1 至 2048 的整數。瀏覽器取得圖片 URL 仍受 CORS 限制；使用者選取的 `File` 不需跨來源圖片請求。

結果包含固定模型身分與 dtype、實際回報的執行環境／裝置、已設定的 execution providers、已載入 ONNX session 組態、耗時及記憶體欄位。`providerEvidence: 'loaded-session-configuration'` 和 session 裝置僅表示已載入的組態，**不代表所有運算子都在 GPU 執行**。`loadMs` 包含模型快取預取／驗證及 processor／模型載入；`firstTokenMs` 是第一個非空 decoded streamer callback 的時間，不是原始 token 或 GPU kernel 的時間。JavaScript／GPU 記憶體數值無法取得，會以 `null` 回報；不會以估算值替代。Node 和瀏覽器的結果／速度未保證一致。

完成後呼叫 `dispose()` 釋放模型資源並還原 Transformers.js 共用設定。`status()` 回傳引擎快取狀態。除非個別事件契約有明確文件，否則應將進度 callback 的事件內容視為不透明資料。

## 網頁抽取與圖片前處理

```js
import { extractPage, loadImage } from 'neko.js';

const page = await extractPage('https://example.test/article', {
  maxHtmlBytes: 2 * 1024 * 1024,
  maxImages: 20,
  timeoutMs: 10_000,
});
const prepared = await loadImage(page.images[0]);
// prepared.data 是正規化 PNG 位元組；prepared.imageId 對應來源圖片。
```

`extractPage(input, options?)` 接受 HTTP(S) 網址或 HTML 字串。HTML 字串中的相對網址需設定 `baseUrl`。預設 HTML 上限 2 MiB、最多 20 個去重圖片網址、圖片下載上限 10 MiB、每個請求逾時 10 秒。解析器抽取語意文字區塊和圖片來源，不執行頁面腳本或取得頁面連結資源。圖片來源包含 `img`、`picture/source[srcset]`、inline style、`background` 屬性及 `og:image`，並保留來源資訊。

`loadImage(image, options?)` 是瀏覽器 API，需要 `createImageBitmap` 和 canvas。它會載入 HTTP(S) raster 圖片或解碼 raster data URL，檢查 MIME 類型是否符合位元組實際格式，限制輸入大小與解碼尺寸、套用圖片方向，等比例縮放至 1280×1280 範圍內（不放大），輸出 PNG。SVG、類型不符或無法解碼的內容會被拒絕。此函式只負責前處理，**不會**生成圖片描述。網路圖片請求受 CORS 限制。

## 報告工具

`renderMarkdown(report)` 會將 `StructuredReport` 渲染為 Markdown，跳脫不可信的報告文字，且僅為 HTTP(S) 來源網址建立連結。`validateStructuredReport(report, page)` 會依擷取頁面／圖片 ID 驗證引用。這些工具不會呼叫模型，也不會產生完整網頁報告。

## 快取與後端

模型 manifest 固定 Hugging Face revision、必要檔案大小與 SHA-256。快取命中時仍會驗證完整性；檢查失敗會報錯，不會當成快取未命中而靜默略過。建立的 prototype 提供 `prefetch(signal?)`，可明確確保固定模型檔案已在快取。瀏覽器快取由瀏覽器管理，可能因使用者操作或瀏覽器淘汰策略而被清除。

`device: 'cpu'` 和 `device: 'webgpu'` 是明確選擇。瀏覽器 WebGPU 需要可用 adapter 且支援 FP16 vision encoder 所需的 `shader-f16`。瀏覽器 CPU 設定使用 ONNX Runtime WebAssembly，但固定模型的 CPU 端對端推理尚未驗證。Node.js provider 是否可用，取決於安裝的原生 ONNX Runtime 建置。沒有自動後端回退，也不保證跨平台結果或效能一致。

## 安全與隱私

頁面 HTML 和模型生成文字都屬不可信資料。解析器不執行腳本，擷取工具不載入頁面子資源。Markdown 雖會跳脫文字，仍須交由安全的 Markdown renderer 顯示。本機推理不保證模型輸出安全或正確。請閱讀 [SECURITY.md](../../SECURITY.md) 的威脅模型與漏洞回報方式。
