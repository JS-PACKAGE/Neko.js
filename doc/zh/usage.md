# 使用說明

## 目前範圍

Neko.js 是供 Node.js 與受支援 WebGPU 瀏覽器使用的本機多模態 SDK。`createNeko()` 會懶載入固定版本 Qwen 模型，提供文字／圖片推理、網站到結構化報告、後端狀態，以及明確的模型／引擎快取控制。報告會以不執行腳本的方式擷取 HTML，並對找到的圖片執行真實推理；它不是爬蟲。呼叫端應自行保護遠端 URL 請求的網路政策。模型輸出可能不正確，請勿用於安全、授權或其他重大決策。

模型固定為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，embedding／decoder 使用 Q4，vision encoder 使用 FP16。首次使用約需下載 871 MB。每個快取資產在使用前均依固定清單驗證大小與 SHA-256。執行時 bundle 內含 Transformers.js 4.2.0；Node 使用 ONNX Runtime 1.30.0。

## 安裝與建置

- 需要 Node.js 22 以上，套件使用 ESM，且不發佈至 npm。
- Git 倉庫支援方式：clone 專案後執行 `npm ci` 和 `npm run build`。這種明確本機建置可配合 npm 11 的生命週期腳本核准機制；若 Git 相依套件的 `prepare` 尚未核准，請勿直接安裝尚未建置的 Git dependency。
- 請從目前工作樹本機建置後執行 `npm pack`；即使 `dist/` 被 Git 忽略，它仍會包含於套件。其他專案可安裝此本機產生的檔案：`npm install /path/to/neko.js-1.0.0.tgz`。`1.0.0` 檔名／版本取自此 checkout 的 `package.json`，不表示 GitHub 或 npm 已發佈更新版 SDK。Node bundle 已包含 Transformers.js，僅將固定版本 `onnxruntime-node`、`sharp`、`parse5` 保留為直接執行依賴。npm 11 若封鎖原生 postinstall，僅核准 `onnxruntime-node@1.30.0` 和 `sharp@0.34.5`，勿對所有套件一律核准腳本。
- `npm test`、`npm run typecheck`、`npm run lint` 分別執行 Node 測試、型別檢查和 ESLint。`npm run test:package:artifact` 會在隔離的 consumer 中實際安裝 packed artifact，檢查公開匯入與 TypeScript 宣告。`npm run test:package` 會額外以該 consumer 執行文字、圖片和報告推理；冷快取時可能下載約 871 MB。
- `npm run test:browser` 執行瀏覽器契約測試。`npm run smoke:browser` 是明確執行的真實模型瀏覽器 UI smoke，可能下載約 871 MB；靜態 demo 請透過安全的 HTTP(S) 來源提供。

此模型不支援瀏覽器 CPU/WASM，因 ONNX Runtime Web 缺少 `GatherBlockQuantized(1)`；建立時會回報不支援後端錯誤，不會自動 fallback。Node 和瀏覽器後端須分別明確選擇與驗證。

## 實測結果

量測使用 Node.js 22.23.3、Transformers.js 4.2.0、原生 ONNX Runtime 1.30.0 及固定 Qwen revision。Hybrid provider profile 記錄 embedding 128 個 WebGPU 事件、vision 536 WebGPU + 50 CPU、decoder 169,472 WebGPU + 12,288 CPU；這不代表全部運算均在 GPU 執行。

- 已填妥驗證快取：模型載入 6,414.542 ms、第一個非空 decoded 輸出 1,487.161 ms、生成 9,353.701 ms、推理總計 9,486.618 ms；網路請求 0。
- 空快取冷啟動：下載 13 個檔案／871,364,778 位元組；外層載入 37,118.820 ms、推理載入 37,114.206 ms、首個輸出 1,228.452 ms、生成 7,728.604 ms、推理總計 7,796.944 ms。
- 全新離線程序重用相同快取：外層載入 4,655.131 ms、推理載入 4,652.725 ms、首個輸出 2,447.610 ms、生成 9,874.194 ms、推理總計 9,997.882 ms；網路請求 0。

`loadMs` 包含快取驗證與 processor／模型載入。`InferenceResult.timings.firstTokenMs` 是從生成開始計算第一個模型 token callback 的時間；它不是第一段非空 decoded 文字片段或 kernel 延遲。下方 prototype 數據另記錄第一段非空 decoded 輸出。Process RSS 約為載入後 2.147 GB、推理中 1.966 GB、dispose 後 0.731 GB。GPU 記憶體未知；以上是實測而非跨平台效能保證。


## SDK 使用方式

```js
import { createNeko } from 'neko.js';

const neko = await createNeko({ device: 'webgpu' });
try {
  // image 可省略；省略時執行純文字推理。
  const answer = await neko.infer({
    image: './photo.png',
    prompt: 'Describe the visible scene.',
    maxNewTokens: 128,
    onToken: (text) => process.stdout.write(text),
  });
  console.log(answer.usage, answer.finishReason);

  const report = await neko.describe('https://example.test/article', {
    language: 'zh-TW',
    format: 'json',
    imageFailurePolicy: 'omit',
    maxNewTokens: 256,
  });
  console.log(report.page.summary, report.images);
  console.log(await neko.cache.model.status(), neko.cache.engine.status());
} finally {
  await neko.dispose();
}
```

`createNeko(options?)` 會設定所選後端與驗證快取後立即回傳，不會載入模型。`cache.model.prefetch()` 只下載並驗證模型檔案，不會載入推理引擎；第一次 `infer()` 或 `describe()` 才懶載入引擎。預設裝置為 `webgpu`，沒有自動 fallback。`cacheDir` 可覆寫 Node 快取位置；否則 macOS 使用 `~/Library/Caches/neko.js`、Windows 使用 `%LOCALAPPDATA%/neko.js`、Linux 使用 `$XDG_CACHE_HOME/neko.js` 或 `~/.cache/neko.js`。瀏覽器使用 Cache Storage。`localFilesOnly: true` 禁止網路下載，必要且已驗證的資產缺少時會明確失敗。

`infer({ prompt, image?, maxNewTokens?, contextWindowTokens?, signal?, onToken? })` 接受 Node 路徑／file URL、HTTP(S) 圖片 URL、瀏覽器 `Blob`／`File` 或本機 `blob:` object URL、支援的 raster data URL，或結構化解碼圖片。Node 讀取遠端圖片時，應提供 `validateDestination`／網路政策；瀏覽器遠端圖片仍受 CORS 限制。提示不可為空白；`maxNewTokens` 預設 128。結果包含解碼文字、`finishReason`、token 用量、固定模型身分、觀測到的後端／session 組態、耗時及記憶體數值（無法觀測時為 `null`）。`firstTokenMs` 由第一個模型 token callback 起算，不等於第一段 decoded 文字。`onToken` 接收解碼文字片段；callback 拋出的錯誤會使推理失敗。

`describe(urlOrHtml, options?)` 會擷取頁面文字與圖片、執行實際圖片推理，並預設產生結構化報告。每個生成階段的 `maxNewTokens` 預設為 256。`format: 'markdown'` 回傳 Markdown；`format: 'json'`（預設）回傳有型別的報告物件。Markdown 欄位標題預設為英文，`zh-*` 語言標籤使用繁體中文標籤。`language` 使用 BCP 47 語言標籤。`imageFailurePolicy: 'error'` 預設於圖片推理失敗時拒絕；`'omit'` 則保留包含錯誤資訊的具型別失敗圖片項目。報告保留來源圖片 ID／provenance，並要求涵蓋所有擷取段落與圖片，且驗證摘要、段落重點和圖片描述等所有生成欄位是否符合指定語言。English 和繁體中文的明顯文字系統錯誤會被拒絕；這只是啟發式檢查，不能保證流暢度或事實正確，其他語言則盡力遵從。`onToken(text, phase)` 會回報 `image`、`section`、`summary` 或 `conclusion` 階段的片段。取消時 Promise 會拒絕且不回傳部分報告；已送入 `onToken` 的片段仍可由呼叫端保留。生成遭截斷時會以 `NekoError` `INCOMPLETE_GENERATION` 拒絕，檢出的語言不符則為 `LANGUAGE_MISMATCH`；兩者都不會自動重試。

`contextWindowTokens` 會依固定模型設定檢查；預設使用保守的 4096-token 工作視窗，不代表實際可用的最大上下文。輸入與輸出預算須合計落在視窗內。報告依實際 tokenizer 分割段落，保留 Unicode 文字邊界；中間證據無法納入摘要預算時，會執行多階段階層縮減。Schema 檢查保留所有段落／圖片來源 ID，但來源參照與成功縮減均不保證語意忠實：摘要可能遺漏事實，結論也可能與來源矛盾。

`neko.cache.model.prefetch/status/clear` 分別下載、檢查、移除固定模型檔案；三者皆回傳 Promise，呼叫 status 時必須 `await`。`neko.cache.engine.status/release` 回報或釋放可重用的即時引擎。預設 `cache: { engine: true, engineTtlMs: 1_800_000 }`，引擎閒置 30 分鐘後釋放；可用 `cache: { engine: false }` 停用或調整 TTL。`neko.backend.current()` 回報所選後端；`neko.backend.detect(device?)` 檢查模型／runtime 相容性，並在瀏覽器 WebGPU 檢查 adapter 可用性。`supported` 不表示已安裝 native provider／driver 或已成功建立 session。同一時間只能由一個 `Neko` 實例擁有 process-global Transformers runtime hooks（`RUNTIME_BUSY`）。每個實例會序列化呼叫；`dispose()` 會中止排隊工作、等待目前工作完成、釋放引擎並還原共用 hooks。錯誤為 `NekoError`，提供 `stage`、`code`，可用時亦包含 `cause`。

`npm run smoke:browser` 透過靜態 demo 執行圖片、純文字和網頁報告推理。`--offline-reload` 會在同一持久化瀏覽器快取中重跑選定流程，同時封鎖模型網路請求。重用快取時，`test:package` 的 `NEKO_MODEL_CACHE` 必須是 SDK 快取**根目錄**；`smoke:browser` 則必須是特定 **revision 目錄**（直接含 `tokenizer.json`、`onnx/`）。這些耗時的真實模型 smoke 不屬於預設 CI。以下以 macOS 預設快取為例：

```sh
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js" npm run test:package
NEKO_MODEL_CACHE="$HOME/Library/Caches/neko.js/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0" npm run smoke:browser -- webgpu 16 --headed --offline-reload
```

此 headed smoke 需要實際可用的 WebGPU adapter；headed 模式本身不保證裝置可用。runner 仍需要本機靜態伺服器；`--offline-reload` 只封鎖模型下載，不代表所有網站網路流量離線，也不代表可關閉該伺服器。

## 擷取、圖片與報告

```js
import { extractPage, loadImage, renderMarkdown, validateStructuredReport } from 'neko.js';

const page = await extractPage('https://example.test/article', {
  maxHtmlBytes: 2 * 1024 * 1024,
  maxImages: 20,
  timeoutMs: 10_000,
  validateDestination: (url) => { /* 套用應用程式的 outbound network policy */ },
});
const prepared = await loadImage(page.images[0]);
// prepared.data 是正規化 PNG 位元組；prepared.imageId 保留來源資訊。
```

`extractPage(input, options?)` 接受 HTTP(S) 網址或 HTML 字串；HTML 中的相對網址需設定 `baseUrl`。預設 HTML 上限為 2 MiB、最多 20 個去重圖片網址、圖片下載上限 10 MiB、每個請求逾時 10 秒。解析器不執行腳本，僅擷取語意文字及圖片來源，不載入連結資源。它會發現 `img`、`picture/source[srcset]`、inline style、`background` 屬性和 `og:image`，並保留來源 metadata。

呼叫端接收不可信 URL 時，遠端擷取可能產生 SSRF 風險。請用 `validateDestination` 檢查每個目的地／redirect，並套用應用程式層級的 outbound network 控制；此 SDK 無法判斷特定部署中哪些私人或內部目的地安全。

`loadImage(image, options?)` 使用 Node 原生解碼器或瀏覽器 bitmap/canvas 路徑解碼擷取出的圖片。它會檢查 raster 位元組／MIME、限制輸入與解碼尺寸、套用方向、等比例縮放至 1280×1280（不放大），並輸出 PNG。SVG 和格式不符／無效內容會被拒絕。此函式只負責前處理；模型推理由 `Neko.describe()` 執行。瀏覽器遠端圖片請求仍受 CORS 限制。

`renderMarkdown(report)` 將 `StructuredReport` 渲染為 Markdown、跳脫不可信文字，且只為 HTTP(S) 建立來源連結。`validateStructuredReport(report, page)` 驗證欄位型別、指定語言檢查、段落完整涵蓋、圖片數量／身分／狀態、失敗策略及 provenance。`Neko.describe()` 將擷取、圖片推理、摘要與驗證串接為端到端報告流程。

## 快取與後端

模型 manifest 固定 Hugging Face revision、必要檔案大小與 SHA-256。每次使用快取前都會驗證；不符時會失敗，不會靜默當作 cache miss。`neko.cache.model.prefetch/status/clear` 僅操作這些固定檔案。瀏覽器 Cache Storage 仍受使用者操作與瀏覽器淘汰策略影響。

`device: 'cpu'` 和 `device: 'webgpu'` 是明確選擇；沒有自動 provider fallback，也不保證跨平台一致。packed consumer 已實際以 Node 原生 CPU 執行文字、本機路徑圖片與 HTML 報告推理。觀察到的固定 model/runtime 組合曾將飽和紅／藍色描述成粉紅或偏粉紅的紅色。請勿將生成圖片描述視為可靠事實；此觀察不能單獨歸因模型，也不代表 Ollama parity。`backend.detect()` 回報模型／runtime 相容性；瀏覽器 WebGPU 也會檢查支援 `shader-f16` 的 adapter，但 Node 不會在實際推理前探測 native provider／driver 是否可用。瀏覽器 CPU/WASM 因 ONNX Runtime Web 缺少 `GatherBlockQuantized(1)` 而不受支援；Neko 會回報錯誤，不會靜默 fallback。Session provider／組態資料不能證明每個 operator 都在 GPU 執行。

## 安全與隱私

頁面 HTML 和模型生成文字都屬不可信資料。解析器不執行腳本，擷取工具不載入頁面子資源。Markdown 雖會跳脫文字，仍須交由安全的 Markdown renderer 顯示。本機推理不保證模型輸出安全或正確。請閱讀 [SECURITY.md](../../SECURITY.md) 的威脅模型與漏洞回報方式。
