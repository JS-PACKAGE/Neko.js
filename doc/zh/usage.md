# 使用說明

## 目前範圍

Neko.js 是供 Node.js 與受支援 WebGPU 瀏覽器使用的本機多模態 SDK。`createNeko()` 會懶載入固定版本 Qwen 模型，提供文字／圖片推理、網站到結構化報告、後端狀態，以及明確的模型／引擎快取控制。報告會以不執行腳本的方式擷取 HTML，並對找到的圖片執行真實推理；它不是爬蟲。呼叫端應自行保護遠端 URL 請求的網路政策。模型輸出可能不正確，請勿用於安全、授權或其他重大決策。

模型固定為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，embedding／decoder 使用 Q4，vision encoder 使用 FP16。首次使用約需下載 871 MB。每個快取資產在使用前均依固定清單驗證大小與 SHA-256。執行時 bundle 內含 Transformers.js 4.2.0；Node 使用 ONNX Runtime 1.30.0。

## 安裝與建置

- 需要 Node.js 22 以上，套件使用 ESM，且不發佈至 npm。
- Git 倉庫支援方式：clone 專案後執行 `npm ci` 和 `npm run build`。這種明確本機建置可配合 npm 11 的生命週期腳本核准機制；若 Git 相依套件的 `prepare` 尚未核准，請勿直接安裝尚未建置的 Git dependency。
- 請從目前工作樹本機建置後執行 `npm pack`；即使 `dist/` 被 Git 忽略，它仍會包含於套件。使用 `npm pack` 印出的檔名，在其他專案執行 `npm install /absolute/path/to/<printed-filename>.tgz`。檔名／版本取自當前 `package.json`，不是本說明的固定值；本專案不發佈至 npm。Node bundle 已包含 Transformers.js，僅將固定版本 `onnxruntime-node`、`sharp`、`parse5` 保留為直接執行依賴。npm 11 若封鎖原生 postinstall，僅核准 `onnxruntime-node@1.30.0` 和 `sharp@0.35.4`，勿對所有套件一律核准腳本。
- `npm test`、`npm run typecheck`、`npm run lint` 分別執行 Node 測試、型別檢查和 ESLint。`npm run test:package:artifact` 會在隔離的 consumer 中實際安裝 packed artifact，檢查公開匯入與 TypeScript 宣告。`npm run test:package` 會額外以該 consumer 執行文字、圖片和報告推理；冷快取時可能下載約 871 MB。
- `npm run test:browser` 執行瀏覽器契約測試。`npm run smoke:browser` 是明確執行的真實模型瀏覽器 UI smoke，可能下載約 871 MB；靜態 demo 請透過安全的 HTTP(S) 來源提供。

此模型不支援瀏覽器 CPU/WASM，因 ONNX Runtime Web 缺少 `GatherBlockQuantized(1)`；建立時會回報不支援後端錯誤，不會自動 fallback。Node 和瀏覽器後端須分別明確選擇與驗證。

### Node 首次建立快取，再離線推理

在已安裝套件的 consumer 中執行此 ESM 範例（`node first-use.mjs`）。選擇由目前使用者擁有的絕對快取根目錄，不能是 symlink 或檔案系統根目錄。第一個實例需連線下載模型；prefetch 驗證所選 profile，但不建立推理 session。若搬至離線機器，保留完整 model ID／revision 目錄結構。

```js
import { createNeko } from 'neko.js';
import { resolve } from 'node:path';

const cacheDir = resolve('./neko-model-cache');
const setup = await createNeko({ device: 'cpu', cacheDir, modelProfile: 'default' });
try {
  await setup.cache.model.prefetch(); // 只需首次連線：default 約 871 MB。
} finally {
  await setup.dispose(); // 建立下個實例前，先釋放 inline runtime owner。
}

const offline = await createNeko({
  device: 'cpu', cacheDir, modelProfile: 'default', localFilesOnly: true,
});
try {
  const result = await offline.infer({ prompt: 'Write one short greeting.', maxNewTokens: 32 });
  console.log(result.text);
} finally {
  await offline.dispose();
}
```

之後可在全新程序中只執行第二個實例，無須模型網路存取。缺少／損壞資產會 fail closed；離線模式不會安裝 Node 原生依賴，也不讓遠端頁面／圖片可用。本機圖片須如後方範例以 `policy.localFiles` 核准 canonical path。`all-q4` 須另外 prefetch 該 profile。離線執行、有效 JSON 與精確來源引用都不是品質或事實正確性保證。

## 歷史 prototype 實測結果

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

`generation` 支援 `sampling`、`temperature`、`topK`、`topP`、`repetitionPenalty`、`noRepeatNgramSize`、`stop`、`stopTokenIds`。Temperature／top-K／top-P 須搭配 `sampling: true`；預設仍為 greedy。字串 stop 可跨解碼片段邊界，且不洩漏 stop 文字。`inferStructured({ ...inferenceOptions, schema })` 將驗證過的 JSON Schema 納入 prompt／context 規劃，回傳解析後的 `value` 與推理 metadata。低成本的 prompt／messages／generation／budget 驗證及 schema 編譯會先於取得模型執行，包含 worker 模式；拒絕無效請求不需已有模型快取。

結構化生成在第一個完整 JSON 值的邊界確定性停止，不是在生成後擷取看似有效的子字串。邊界偵測使用原始生成 token IDs；若完成 JSON 的同一個 token 內還有多餘內容，仍會驗證並拒絕，不切除尾端。數字根值須等到空白或 EOS 才確立邊界。無效前綴、不完整 JSON 與 schema 違規會被拒絕，不修補、不重試。完整的受支援 Draft-07 runtime 驗證仍 fail closed（無效／不支援 schema 為 `SCHEMA_INVALID`；無效生成 JSON／schema 輸出為 `STRUCTURED_OUTPUT`）。結果的 `structured.mode` 為 `'json-boundary-runtime-validation'`，`structured.dialect` 為 `'draft-07'`。這**不是** schema grammar constrained decoding、語意驗證或生成成功保證。

### 精確推理規劃

```ts
import type { InferencePlan } from 'neko.js';

const request = { prompt: 'Write one short greeting.', maxNewTokens: 32 };
const plan: InferencePlan = await neko.planInference(request);
console.log(plan.inputTokens, plan.maxNewTokens, plan.contextLimit,
  plan.availableOutputTokens, plan.fits);
if (plan.fits) console.log((await neko.infer(request)).text);
```

`planInference(options)` 使用與推理相同的實際 chat template／tokenizer／圖片前處理，包含圖片 token 展開。選填 `schema` 會把結構化輸出指令納入計數。結果含模型／圖片觀測及 execution metadata，但不生成 token。`availableOutputTokens` 為 `max(0, contextLimit - inputTokens)`；`fits` 比較請求輸出預算與該容量。有效的冷啟動規劃請求**會載入模型／processor，也可能下載所選資產**，不是輕量的 tokenizer-only API。規劃超出上下文時回傳 `fits: false`，實際推理則拒絕。規劃不預留佇列容量，也不保證稍後生成成功。

```ts
const schema = {
  type: 'object',
  properties: { greeting: { type: 'string' } },
  required: ['greeting'],
  additionalProperties: false,
};
const structuredRequest = {
  prompt: 'Return a short greeting in the requested JSON shape.',
  maxNewTokens: 64, schema,
};
const structuredPlan = await neko.planInference(structuredRequest);
if (structuredPlan.fits) {
  const result = await neko.inferStructured(structuredRequest);
  console.log(result.value, result.structured.mode, result.structured.dialect);
}
```

`describe(urlOrHtmlOrPage, options?)` 接受 HTTP(S) URL、惰性 HTML 或經驗證的 `Page` 擁有副本。`sources` 可選取段落／圖片 ID，並提供可非同步的 `paragraph`／`image` predicate；ID 須存在且不重複，predicate 接收 readonly 來源記錄。擷取與選取發生於載入模型前。報告須涵蓋所有選取來源，刻意排除的來源不需涵蓋；圖片前處理結果會重用於該階段推理。每個階段產生證據連結的 claims，必要時階層縮減摘要／結論。每階段 `maxNewTokens` 預設 256；`format: 'json'` 回傳具型別報告，`'markdown'` 回傳已跳脫的 Markdown。`language` 為 BCP 47；英文與繁體中文文字系統檢查僅為啟發式，不保證流暢度或事實。`onToken(text, phase)` 區分 `image`、`section`、`summary`、`conclusion`。`imageFailurePolicy: 'error'` 拒絕圖片失敗；`'omit'` 保留具型別的失敗項目，但不壓下 policy 違規、callback 錯誤（含 `throw undefined`）、取消或 budget 錯誤。無效／截斷生成與語言不符不會自動重試。

`contextWindowTokens` 會依固定模型設定檢查；預設使用保守的 4096-token 工作視窗，不代表實際可用的最大上下文。輸入與輸出預算須合計落在視窗內。報告依實際 tokenizer 分割段落並保留 Unicode 邊界；每個 section 請求最多處理四段 quote span，產生一至四個證據連結 claims，不再把任意大型來源壓成單一 claim。`sourceFacts` 獨立於生成摘要，以精確、連續引用及 UTF-16 offsets 保留所有選取段落文字。Ledger 是保留的來源文字，**不是**擷取或查核過的真實世界事實。容量足夠時，摘要／結論使用保留來源證據；不足時才階層縮減生成 claims。來源參照與成功縮減均不保證語意忠實。

生成 schema 依階段區分：sections 與中間 reduction 請求一至四個 `{ text, evidenceIds }` claims；圖片描述、`page.summary` 與 `conclusion` 請求一段精簡、證據連結的文字，可包含多個受來源支持的事實。此設計限制輸出形狀，不保證事實或完整性。Checkpoint 的 request hashes 綁定實際階段 schema。

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

### 版本化報告與 checkpoints

報告使用 `schemaVersion: 2`，包含 `sourceFacts` 與 `integrity: { algorithm: 'sha256', checksum }`。`metadata.coverage` 記錄選取段落 IDs／字元數、保留引用數／字元數、模型與摘要引用的 fact IDs、`conclusionBasis`（`'retained-source'` 或 `'reduced-generated-claims'`），以及 `semanticRetention: 'not-measured'`。來源涵蓋是結構 accounting，不是語意 recall 分數；生成 claims 沒有事實查核。

`renderMarkdown(report)` 除生成 sections 外，也包含保留引用 ledger 與 coverage；即使模型摘要遺漏內容，仍可檢視精確來源文字。跳脫不讓引用文字變成可信或私密資料。

```ts
import {
  serializeStructuredReport, parseStructuredReport,
  serializeReportCheckpoint, parseReportCheckpoint,
} from 'neko.js/report';
import type { StructuredReport, ReportCheckpoint } from 'neko.js';

const savedReport = await serializeStructuredReport(report);
const restored: StructuredReport = await parseStructuredReport(savedReport);
console.log(restored.schemaVersion, restored.metadata.coverage);

// 在 describe 的 onCheckpoint 中，安全保存回傳字串。
async function saveCheckpoint(checkpoint: ReportCheckpoint): Promise<string> {
  return serializeReportCheckpoint(checkpoint);
}
async function resumeSaved(saved: string): Promise<StructuredReport> {
  const checkpoint = await parseReportCheckpoint(saved);
  return neko.describe('<article><p>The sky is blue.</p></article>', {
    language: 'zh-TW', format: 'json', imageFailurePolicy: 'omit',
    maxNewTokens: 256, resume: checkpoint,
  });
}
```

序列化與解析都先驗證才接受資料；report helper 可用第二個參數傳入相符的選取 `Page`。`validateReportCheckpoint(value)` 也可驗證記憶體中的 checkpoint。Checkpoint 使用 `version: 2`、`plan: 'evidence-first-v2'`，保存引用 ledger，只有輸入／模型／設定相符才可 resume。**持久化相容性的 breaking change：**沒有版本、舊版或未來版本的報告／checkpoint 都會拒絕，不自動遷移、不透過 alias 解讀。報告拒絕為 `TypeError`；無效 checkpoint 版本為 `CHECKPOINT_INVALID`。請以原始輸入在目前契約下重新產生。Checksum 可偵測保存內容不一致／遭修改，但不認證作者；來源引用、metadata 和生成文字可能敏感。

Checkpoint 的 `sectionPlan` 記錄依序排列的 source-fact ID 群組，每組一至四個，且須精確涵蓋 ledger。即使重算 checksum，驗證仍拒絕 section 引用其他群組；resume 亦核對確定性 plan 及 request hashes。請以 helper 保存 SDK 提供的 checkpoint，不手動構造保存 state。

## 快取與後端

模型 manifest 固定 Hugging Face revision、必要檔案大小與 SHA-256。每次使用快取前都會驗證；不符時會失敗，不會靜默當作 cache miss。`neko.cache.model.prefetch/status/clear` 僅操作這些固定檔案。瀏覽器 Cache Storage 仍受使用者操作與瀏覽器淘汰策略影響。

### 瀏覽器模型來源

#### 瀏覽器首次使用流程

1. 先在 checkout 或已安裝套件的 consumer 完成前述 Node 線上 prefetch，產生 mirror 所需的相同驗證檔案，再於 checkout 執行 `npm run build`。
2. 現有 demo 可先以 `npx playwright install chromium` 安裝瀏覽器 harness，再執行現有 runner；它會自建 loopback 靜態伺服器及精確 origin CORS mirror、驗證並寫入瀏覽器 Cache Storage，再執行 UI：

   ```sh
   NEKO_MODEL_CACHE="/absolute/path/to/neko-model-cache/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0" npm run smoke:browser -- webgpu 256 --headed --offline-reload
   ```

   環境變數是 **revision 目錄**，不是快取根目錄。`256` 是明確的每階段輸出預算，不保證所有報告都能完成。Runner 結束會關閉暫存 server／profile；它是真實推理檢查，不是持續運作的開發伺服器。瀏覽器／driver 須支援 WebGPU，`--headed` 不會產生 adapter。隨附 demo 本身沒有 mirror URL 欄位，故空快取時單純開啟 demo 並不會設定首次線上 mirror。
3. 自己的應用程式須透過 HTTPS 或可信任 loopback HTTP 提供 ESM bundle 與整個 `dist/browser/assets/`，**不可**使用 `file:`。保留輸出的 `neko.js`、`worker.js`、`assets/` 相對佈局；`.js`／`.mjs` 使用 JavaScript MIME、`.wasm` 使用 `application/wasm`。WebGPU 仍需要 ONNX WASM runtime 資產，這不是 CPU fallback。若使用 SDK 的 fetched/blob runtime modules 與 module workers，CSP 須允許它們。
4. 建議以應用程式同 origin 的 `/models/` reverse proxy 連至 loopback mirror，完整保留固定路徑，upstream 設定 `Host: 127.0.0.1:8787`（helper 拒絕任意 Host）。若轉送 `Origin`，`--allow-origin` 應為精確應用程式 origin。勿公開任意快取檔案或通用 URL proxy。例如既有 nginx server 監聽 `http://127.0.0.1:4173` 時：

   ```nginx
   location /dist/browser/ {
     root /absolute/path/to/Neko.js;
     types { application/javascript js mjs; application/wasm wasm; }
   }
   location /models/ {
     proxy_pass http://127.0.0.1:8787;
     proxy_set_header Host 127.0.0.1:8787;
   }
   ```

   用後方指令啟動 mirror，替換成實際快取**根目錄**。頁面匯入 `/dist/browser/neko.js`，把 `modelSource.baseUrl` 設為 `new URL('/models/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0/', location.href).href`。同 origin 的模型／runtime 不需跨 origin CORS 核准。若直接存取另一個 `8787` mirror，頁面 origin 的 scheme／host／port 必須精確符合 `--allow-origin`，勿使用 wildcard CORS。遠端圖片／頁面的 CORS 與模型 CORS 是不同條件。

   ```js
   import { createNeko } from '/dist/browser/neko.js';

   const neko = await createNeko({
     device: 'webgpu',
     modelSource: {
       baseUrl: new URL('/models/onnx-community/Qwen3.5-0.8B-ONNX-OPT/fafab72d87a9e6be3925b38caf48286d2838f2d0/', location.href).href,
     },
   });
   try {
     await neko.cache.model.prefetch();
     document.querySelector('#output').textContent =
       (await neko.infer({ prompt: 'Write one short greeting.', maxNewTokens: 32 })).text;
   } finally {
     await neko.dispose();
   }
   ```

完成快取後，dispose 線上實例，再以相同 profile 建立 `localFilesOnly: true` 實例。封鎖模型流量後仍須提供套件／runtime 資產；模型快取離線不代表整個網站離線。Cache Storage 可能淘汰，不同 origins／profiles 不會隱含共用。瀏覽器 CPU/WASM 拒絕且不自動 fallback；此流程不承諾輸出正確。

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

`device: 'cpu'` 和 `device: 'webgpu'` 是明確選擇；沒有自動 provider fallback，也不保證跨平台一致。`backend.detect()` 回報模型／runtime 相容性；瀏覽器 WebGPU 也會檢查支援 `shader-f16` 的 adapter，但 Node 不會在實際推理前探測 native provider／driver 是否可用。瀏覽器 CPU/WASM 因 ONNX Runtime Web 缺少 `GatherBlockQuantized(1)` 而不受支援。請求／session provider 組態既不證明推理成功，也不證明所有 operator 都在 GPU 執行；前述歷史 hybrid profiling 實際觀測到 CPU 與 WebGPU operator 並存。

證據快照（2026-10-03）：macOS（Darwin 27）、arm64，於 Node 22.23.3 與 Node 26.7.0 完成全新 CPU smoke。下方結果只驗證所記錄 runtime／backend／profile／輸入，不涵蓋其他列或所有 Node `>=22` 版本。

| Runtime／平台 | API／契約狀態 | 歷史真實推理證據 | 目前改動的全新驗證 |
| --- | --- | --- | --- |
| Node，macOS／arm64，CPU | 原生 CPU 路徑；套件要求 Node `>=22`、build target 為 Node 22 | Node 22.23.3 冷下載／離線 prototype 與 packed consumer 文字／路徑圖片／HTML 報告實測 | Node 22.23.3 與 26.7.0 離線 CPU／default worker 推理、規劃、inline／worker preflight、報告持久化及 4-stage checkpoint resume 通過 |
| Node，macOS／arm64，WebGPU | 明確選擇原生 provider；driver／session 須可用 | Hybrid profiling 記錄 GPU **與 CPU** operator | 本次改動尚未建立證據 |
| Node，Windows 或 Linux，CPU／WebGPU | 依平台的原生依賴／provider 路徑；已實作快取位置 | 本處沒有特定平台真實推理證據 | 未驗證；不主張 Windows／Linux 或 GPU parity |
| Chromium，WebGPU | 需 secure context、可用 adapter／`shader-f16`、模型快取／mirror 及 runtime assets | 過去版本記錄 demo 推理與封鎖模型網路後 reload | macOS／arm64 的完整 Chromium 153.0.0.0 headless、搭配 `--enable-unsafe-webgpu` 通過精確文字／圖片／schema 規劃、結構化生成、圖片／報告推理、持久化與實際 demo worker 文字；不涵蓋所有 Chromium／OS 組合 |
| Firefox／Safari／其他瀏覽器 | 須各自符合相同 runtime 條件 | 本處沒有特定瀏覽器真實推理證據 | 未驗證；API 存在不足以證明可用 |
| 任意瀏覽器，CPU／WASM | 不支援；拒絕且不 fallback | 實際觀測缺少 `GatherBlockQuantized(1)` | 仍為不支援組態 |

全新 Node 22.23.3 與 26.7.0 CPU／default worker runs 的文字／chat／schema 計數符合規劃、structured 串流符合完整回應、超額規劃回傳 `fits: false`，並驗證 report／checkpoint 序列化／解析、worker execution identity 與四個 resume 階段。報告 sections 包含五個測試事實（1987、一棵 18 公尺高的橡樹、沒有噴泉、Plot A 12–19、Plot B 24–31），5 段精確引用保留 134/134 UTF-16 字元。**生成摘要遺漏開園年份，且用 `q1` 而非「沒有噴泉」的 `q3` 引用；紅色圖片 fixture 也被描述為粉紅色。**選取文字的引用 ledger 不失真，不代表生成摘要與引用已通過語意查核（`semanticRetention: 'not-measured'`）。

最終 packed consumer 亦於 Node 22.23.3 通過：安裝本機產生的 tarball、檢查公開套件 exports 型別，並實際執行離線原生 CPU 文字、路徑圖片、精確推理規劃、結構化生成、HTML／圖片報告、持久化驗證、竄改來源引文的拒絕與 checkpoint resume。瀏覽器 inline 與 worker API smoke 分別通過，包含記錄 worker execution identity 後的報告 checksum 驗證；亦再次視覺確認 demo worker 文字流程。

全新瀏覽器 run 將 13 個本機 SHA-256 驗證的固定資產寫入新的 persistent profile；**沒有**驗證首次網際網路下載。規劃計數符合實際推理（文字 25、圖片 90，另含 schema 請求）；無效冷請求在取得模型前以 preprocess 拒絕。結構化輸出為 `{ answer: 7 }`；報告 key points／overview 保留三個測試事實（1987、一棵 18 公尺高的橡樹、沒有噴泉），3 段精確引用涵蓋 75/75 UTF-16 字元，並通過驗證／持久化 round-trip。實際 demo worker 文字回傳 `7`，亦完成視覺檢查。

這是使用 unsafe-WebGPU enablement flag 的 headless 測試組態，不代表一般無旗標瀏覽器支援或硬體 GPU／operator 證據；software／headless adapter 可能不同。預設 headless-shell adapter 缺少 `shader-f16`，正確於取得模型前拒絕。Chromium／Firefox／WebKit 契約測試亦通過（15/15），但後兩者**不是**真實模型推理證據。Fixture 輸出成功不代表一般報告品質 gate 通過。

契約測試、typecheck 和 adapter preflight 不是實際推理證據。固定 pipeline 的歷史觀察包括把飽和紅／藍色描述為粉紅／偏粉紅的紅色、報告遺漏事實與矛盾結論；這不能單獨歸因模型，也不是 Ollama 比較。矩陣中沒有任何平台保證顏色辨識可靠、語意保留、事實正確或效能。

## 安全與隱私

頁面 HTML 和模型生成文字都屬不可信資料。解析器不執行腳本，擷取工具不載入頁面子資源。Markdown 雖會跳脫文字，仍須交由安全的 Markdown renderer 顯示。本機推理不保證模型輸出安全或正確。請閱讀 [SECURITY.md](../../SECURITY.md) 的威脅模型與漏洞回報方式。
