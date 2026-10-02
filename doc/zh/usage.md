# 使用說明

## 目前範圍

Neko.js 是供 Node.js 與受支援 WebGPU 瀏覽器使用的本機多模態 SDK。`createNeko()` 會懶載入固定版本 Qwen 模型，提供文字／圖片推理、網站到結構化報告、後端狀態，以及明確的模型／引擎快取控制。報告會以不執行腳本的方式擷取 HTML，並對找到的圖片執行真實推理；它不是爬蟲。呼叫端應自行保護遠端 URL 請求的網路政策。模型輸出可能不正確，請勿用於安全、授權或其他重大決策。

模型固定為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，embedding／decoder 使用 Q4，vision encoder 使用 FP16。首次使用約需下載 871 MB。每個快取資產在使用前均依固定清單驗證大小與 SHA-256。執行時 bundle 內含 Transformers.js 4.2.0；Node 使用 ONNX Runtime 1.30.0。

## 安裝與建置

- 需要 Node.js 22 以上，套件使用 ESM，且不發佈至 npm。
- Git 倉庫支援方式：clone 專案後執行 `npm ci` 和 `npm run build`。這種明確本機建置可配合 npm 11 的生命週期腳本核准機制；若 Git 相依套件的 `prepare` 尚未核准，請勿直接安裝尚未建置的 Git dependency。
- 請從目前工作樹本機建置後執行 `npm pack`；即使 `dist/` 被 Git 忽略，它仍會包含於套件。其他專案可安裝此本機產生的檔案：`npm install /path/to/neko.js-1.1.0.tgz`。`1.1.0` 檔名／版本取自此 checkout 的 `package.json`；本專案不發佈至 npm。Node bundle 已包含 Transformers.js，僅將固定版本 `onnxruntime-node`、`sharp`、`parse5` 保留為直接執行依賴。npm 11 若封鎖原生 postinstall，僅核准 `onnxruntime-node@1.30.0` 和 `sharp@0.35.4`，勿對所有套件一律核准腳本。
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
import { realpath } from 'node:fs/promises';

const photo = await realpath('./photo.png');
const neko = await createNeko({ device: 'cpu', policy: { localFiles: (path) => path === photo } });
try {
  // image 可省略；省略時執行純文字推理。
  const answer = await neko.infer({
    image: './photo.png',
    prompt: 'Describe the visible scene.',
    maxNewTokens: 128,
    onToken: (text) => process.stdout.write(text),
  });
  console.log(answer.usage, answer.finishReason);

  const report = await neko.describe('<article><p>The sky is blue.</p></article>', {
    language: 'zh-TW',
    format: 'json',
    imageFailurePolicy: 'omit',
    maxNewTokens: 256,
  });
  console.log(report.page.summary, report.images);
  console.log(await neko.cache.model.status(), await neko.cache.engine.status());
} finally {
  await neko.dispose();
}
```

`createNeko(options?)` 只設定後端與驗證快取，不會載入模型。`modelProfile: 'default'` 使用 q4 embedding／decoder 與 fp16 vision；`'all-q4'` 使用另行固定的 q4 vision 資產。Loaded session metadata 記錄實際組態，而非只列出請求值。`cache.model.prefetch()` 驗證檔案但不建立 session；推理會懶載入。`load()` 載入 session，`warmup()` 亦實際執行文字與視覺路徑，`runtimeStatus()` 回報 readiness。預設裝置為 `webgpu`，沒有自動 fallback。Node `cacheDir` 預設為 macOS `~/Library/Caches/neko.js`、Windows `%LOCALAPPDATA%/neko.js`、Linux `$XDG_CACHE_HOME/neko.js` 或 `~/.cache/neko.js`；瀏覽器使用 Cache Storage。`localFilesOnly: true` 禁止缺少快取的模型下載及遠端頁面／圖片請求；仍允許受信任的套件本機 worker／runtime bootstrap。

`infer({ prompt?, messages?, image?, generation?, maxNewTokens?, contextWindowTokens?, signal?, onToken? })` 接受非空白提示，或依序排列的 `system`／`user`／`assistant` 文字與圖片訊息。多張圖片於同一次生成共同推理，保留各自尺寸與內容雜湊。圖片支援 Node 路徑／file URL、HTTP(S) URL、瀏覽器 `Blob`／`File`／本機 `blob:` URL、支援的 raster data URL 或結構化解碼圖片。本機檔案與遠端頁面／圖片須由實例 policy 核准，瀏覽器仍受 CORS 限制。`maxNewTokens` 預設 128。結果包含文字、finish reason、usage、model／profile、觀測 session、execution、timings 和 memory（不可觀測時為 `null`）。`onToken` 回報解碼片段；`firstTokenMs` 由第一個模型 token callback 起算。

`generation` 支援 `sampling`、`temperature`、`topK`、`topP`、`repetitionPenalty`、`noRepeatNgramSize`、`stop`、`stopTokenIds`。Temperature／top-K／top-P 須搭配 `sampling: true`；預設仍為 greedy。字串 stop 可跨解碼片段邊界，且不洩漏 stop 文字。`inferStructured({ ...inferenceOptions, schema })` 將驗證過的 JSON Schema 納入 prompt／context 規劃，回傳解析後的 `value` 與推理 metadata。無效／不支援的 schema 會在載入模型前以 `SCHEMA_INVALID` 拒絕；模型產生無效 JSON 則為 `STRUCTURED_OUTPUT`。這是輸出驗證，不保證模型遵守 schema；不會修補或重試。

`describe(urlOrHtmlOrPage, options?)` 接受 HTTP(S) URL、惰性 HTML 或經驗證的 `Page` 擁有副本。`sources` 可選取段落／圖片 ID，並提供可非同步的 `paragraph`／`image` predicate；ID 須存在且不重複，predicate 接收 readonly 來源記錄。擷取與選取發生於載入模型前。報告須涵蓋所有選取來源，刻意排除的來源不需涵蓋；圖片前處理結果會重用於該階段推理。每個階段產生證據連結的 claims，必要時階層縮減摘要／結論。每階段 `maxNewTokens` 預設 256；`format: 'json'` 回傳具型別報告，`'markdown'` 回傳已跳脫的 Markdown。`language` 為 BCP 47；英文與繁體中文文字系統檢查僅為啟發式，不保證流暢度或事實。`onToken(text, phase)` 區分 `image`、`section`、`summary`、`conclusion`。`imageFailurePolicy: 'error'` 拒絕圖片失敗；`'omit'` 保留具型別的失敗項目，但不壓下 policy 違規、callback 錯誤（含 `throw undefined`）、取消或 budget 錯誤。無效／截斷生成與語言不符不會自動重試。

`contextWindowTokens` 會依固定模型設定檢查；預設使用保守的 4096-token 工作視窗，不代表實際可用的最大上下文。輸入與輸出預算須合計落在視窗內。報告依實際 tokenizer 分割段落，保留 Unicode 文字邊界；中間證據無法納入摘要預算時，會執行多階段階層縮減。Schema 檢查保留所有段落／圖片來源 ID，但來源參照與成功縮減均不保證語意忠實：摘要可能遺漏事實，結論也可能與來源矛盾。

`budget: { maxTotalTokens, maxDurationMs }` 限制累計輸入／輸出 token 與整個請求耗時，包含排隊、載入、擷取、前處理、生成，以及支援的非同步來源 predicate、資源核准、`onEvent`、`onCheckpoint`。Deadline 取消 SDK 對使用者 Promise 的等待，不會強制搶占 native ORT／OS 呼叫，也不終止呼叫端自己的外部副作用。`onEvent` 回報階段轉換；`onCheckpoint` 提供可保存、cloneable 的 state。`resume: checkpoint` 只重用相容的完成階段，保留已花費 budget，並驗證來源／模型／設定／checksum 與圖片內容版本。報告 checkpoint 建立後的失敗提供 `ReportError.checkpoint` 最終 accounting；較早的擷取／選取／載入失敗可能只有 `NekoError`。Checkpoint 保存選取文字與 metadata，不包含圖片像素；持久化資料可能敏感。

所有 cache／backend／status 方法都回傳 Promise。`cache.model.prefetch/status/clear` 僅操作選取的 pinned profile；`cache.engine.status/release` 檢查／釋放 live engine。`cache: { engine: true, engineTtlMs: 1_800_000 }` 於閒置 30 分鐘內重用引擎。`backend.detect()` 檢查相容性，並不證明 native driver／session 成功。預設 `execution: 'inline'` 只允許一個 process-global runtime owner；`'worker'` 使用真正的 Node thread 或 browser module worker，各自獨立持有 runtime。各實例使用有上限的 FIFO（`queue: { maxPending: 8 }`）、回報排隊耗時，滿額以 `QUEUE_FULL` 拒絕；`queueStatus()` 提供即時狀態。Worker 保留 callback 順序與 typed errors／checkpoints。`dispose()` 取消排隊工作、等待目前工作的安全清理、釋放資源並還原 hooks。

`policy.network(url, kind)` 正常回傳（`undefined`／`true`）表示核准；拋錯／回傳 `false` 表示拒絕。`kind` 為 `model`、`runtime`、`worker`、`page`、`image`。`policy.localFiles(canonicalPath)` 明確核准 Node 輸入檔案。預設只允許固定模型傳輸與套件 bootstrap；任意頁面／圖片目的地與本機檔案預設拒絕。每個可觀測 redirect hop 都先核准；每次呼叫的 `validateDestination` 只增加限制，不取代實例 policy。SDK 無法替部署判斷 private／SSRF-safe 邊界，應用程式核准須自行落實。

Factory 會擷取已知 policy hook 參照（含 class prototype 方法），並綁定原始 receiver。已定義卻不是函式的 hook／無效 policy record 在 preflight 拒絕；只凍結 SDK 自有 hook record，不凍結呼叫端 policy 物件。

`npm run smoke:browser` 透過靜態 demo 執行圖片、純文字和網頁報告推理。`--offline-reload` 會在同一持久化瀏覽器快取中重跑選定流程，同時封鎖模型網路請求。設定 `NEKO_MODEL_CACHE` 時，smoke runner 會透過本機 mirror 驗證所選 pinned profile，並先寫入瀏覽器 Cache Storage，再以 local-only 模式執行 UI。重用快取時，`test:package` 的 `NEKO_MODEL_CACHE` 必須是 SDK 快取**根目錄**；`smoke:browser` 則必須是特定 **revision 目錄**（直接含 `tokenizer.json`、`onnx/`）。這些耗時的真實模型 smoke 不屬於預設 CI。以下以 macOS 預設快取為例：

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

`renderMarkdown(report)` 跳脫不可信文字，只為 HTTP(S) provenance 建立連結。`await validateStructuredReport(report, selectedPage?)` 稽核持久化的選取來源 snapshot／SHA-256 版本、精確引用 offsets／內容、claim spans／證據連結、完整選取來源涵蓋、圖片狀態／provenance、語言、loaded model／session 身分及累計 metadata。可另外傳入相符的選取 `Page` 比較外部來源；保存的 JSON 不必重新 fetch 即可稽核。它可檢出參照與 accounting 不一致，但不判斷 claim 是否由來源或圖片像素蘊含；雜湊亦不是來源認證。

## 快取與後端

模型 manifest 固定 Hugging Face revision、必要檔案大小與 SHA-256。每次使用快取前都會驗證；不符時會失敗，不會靜默當作 cache miss。`neko.cache.model.prefetch/status/clear` 僅操作這些固定檔案。瀏覽器 Cache Storage 仍受使用者操作與瀏覽器淘汰策略影響。

### 瀏覽器模型來源

瀏覽器隱藏跨來源 redirect 目的地（`opaqueredirect`）；Hugging Face redirect 因此以 `POLICY_DENIED` fail closed，不會靜默跟隨未核准的 hop。首次線上下載請明確指定 `modelSource: { baseUrl }` mirror／broker，以選取 profile 的固定相對路徑提供資產。所有 body 仍驗證 manifest 大小／SHA-256，並使用原始 canonical Hugging Face Cache Storage key。Base 須為無 credentials／query／fragment 的絕對 HTTP(S) URL；可觀測 redirect 仍逐次核准，CORS 與 secure-context 限制不變。

Factory 在 worker 序列化前，擷取並凍結正規化的 plain `baseUrl` record，亦支援 class getter。跨 origin mirror remap／可觀測 redirect hop 會移除 `Authorization`、`Cookie`、`Proxy-Authorization`；後續回到原 origin 也不復原已移除的 credentials。同 origin 請求與非敏感 headers 保留。

已建置 checkout 且已有驗證 Node 快取時，可使用僅綁定 loopback、不提供任意檔案路徑的 helper：

```sh
npm run build
node scripts/serve-model-mirror.mjs --cache-dir="$HOME/Library/Caches/neko.js" --model-profile=default --port=8787 --allow-origin=http://127.0.0.1:4173
```

於該精確核准 origin 提供的瀏覽器頁面：

```js
const neko = await createNeko({
  device: 'webgpu',
  modelProfile: 'default',
  modelSource: { baseUrl: 'http://127.0.0.1:8787/models/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0/' },
});
await neko.cache.model.prefetch();
```

亦可將該目錄 reverse-proxy 至應用程式同 origin。使用 all-q4 時，helper `--model-profile=all-q4` 與 factory profile 須相符。Helper 啟動時驗證整個選取快取，並在每次請求前驗證該資產；不會 fetch Hugging Face 或跟隨瀏覽器 redirect。完成快取後，新建 `localFilesOnly: true` 實例可重用驗證檔案，不發出模型網路請求。

### 後端相容性

`device: 'cpu'` 和 `device: 'webgpu'` 是明確選擇；沒有自動 provider fallback，也不保證跨平台一致。packed consumer 已實際以 Node 原生 CPU 執行文字、本機路徑圖片與 HTML 報告推理。觀察到的固定 model/runtime 組合曾將飽和紅／藍色描述成粉紅或偏粉紅的紅色。請勿將生成圖片描述視為可靠事實；此觀察不能單獨歸因模型，也不代表 Ollama parity。`backend.detect()` 回報模型／runtime 相容性；瀏覽器 WebGPU 也會檢查支援 `shader-f16` 的 adapter，但 Node 不會在實際推理前探測 native provider／driver 是否可用。瀏覽器 CPU/WASM 因 ONNX Runtime Web 缺少 `GatherBlockQuantized(1)` 而不受支援；Neko 會回報錯誤，不會靜默 fallback。Session provider／組態資料不能證明每個 operator 都在 GPU 執行。

## 安全與隱私

頁面 HTML 和模型生成文字都屬不可信資料。解析器不執行腳本，擷取工具不載入頁面子資源。Markdown 雖會跳脫文字，仍須交由安全的 Markdown renderer 顯示。本機推理不保證模型輸出安全或正確。請閱讀 [SECURITY.md](../../SECURITY.md) 的威脅模型與漏洞回報方式。
