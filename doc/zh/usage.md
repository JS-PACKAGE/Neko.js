# 使用說明

## 目前範圍

Neko.js 是供 Node.js 與受支援 WebGPU 瀏覽器使用的本機多模態 SDK。`createNeko()` 會懶載入固定版本 Qwen 模型，提供文字／圖片推理、網站到結構化報告、後端狀態，以及明確的模型／引擎快取控制。生成報告以不執行腳本的方式擷取 HTML，並對選取圖片執行真實推理；extractive 模式不做模型或圖片推理。它不是爬蟲。呼叫端應自行保護遠端 URL 請求的網路政策。模型輸出可能不正確，請勿用於安全、授權或其他重大決策。

預設模型為 `onnx-community/Qwen3.5-0.8B-ONNX-OPT` revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`，另提供已註冊的 `onnx-community/Qwen3.5-2B-ONNX-OPT` revision `2ea7886f48b926aca97de8b0e041ffca7e3ebaa9`。以 `createNeko({ model: 'onnx-community/Qwen3.5-2B-ONNX-OPT', device: 'cpu' })` 選取；公開的 `MODEL_REGISTRY` 列出固定身分。兩者的 `default` profile 均為 q4 embedding／decoder、fp16 vision，`all-q4` 則全部 q4。不可指定任意模型或 revision；`modelSource` 只改變已核准固定資產的傳輸 mirror。0.8B/default 首次完整下載約 871 MB，其他模型／profile 不適用此大小。每個快取資產在使用前均依固定清單驗證大小與 SHA-256。執行時 bundle 內含 Transformers.js 4.2.0；Node 使用 ONNX Runtime 1.30.0。

## 安裝與建置

- 需要 Node.js 22.13 以上，套件使用 ESM，且不發佈至 npm。
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

預設 `structuredMode: 'constrained'` 使用真正的 tokenizer-aware JSON grammar，逐 token 限制合法候選，再完整執行 runtime schema 驗證；結果 `structured.mode` 為 `'tokenizer-constrained-runtime-validation'`。支援 primitive type、primitive `enum`／`const`、字串 `minLength`／`maxLength`、以 `properties`／`required`／`additionalProperties: false` 定義的封閉物件，以及同質 `items` 陣列的 `minItems`／`maxItems`／`uniqueItems`。數值上下界、schema composition 等不受 grammar 支援的契約在取得引擎前以 `SCHEMA_UNSUPPORTED` 拒絕，不會暗中 fallback。只有明確指定 `structuredMode: 'validation-only'` 才使用既有完整受支援 Draft-07 runtime validator 與 JSON 邊界停止，結果模式為 `'json-boundary-runtime-validation'`；無效 schema 為 `SCHEMA_INVALID`。兩種模式皆驗證整個 JSON，不修補生成、不擷取子字串，不保證完成或事實正確。完成 JSON 的同一 token 若含尾端多餘內容仍會拒絕；數字根值需空白或 EOS 確立邊界。Schema／JSON 不符為 `STRUCTURED_OUTPUT`。

### 精確推理規劃

```ts
import type { InferencePlan } from 'neko.js';

const request = { prompt: 'Write one short greeting.', maxNewTokens: 32 };
const plan: InferencePlan = await neko.planInference(request);
console.log(plan.inputTokens, plan.maxNewTokens, plan.contextLimit,
  plan.availableOutputTokens, plan.fits);
if (plan.fits) console.log((await neko.infer(request)).text);
```

`planInference(options)` 使用與推理相同的實際 chat template／tokenizer，選填 `schema`／`structuredMode` 會納入結構化指令。**純文字規劃僅下載／讀取已驗證 tokenizer、config 與 template，不建立 ONNX sessions**；離線仍須已有這些固定資產。圖片規劃會做真實前處理與圖片 token 展開，可能取得引擎。`availableOutputTokens` 為 `max(0, contextLimit - inputTokens)`；`fits` 比較輸出預算與容量。超出上下文回傳 `fits: false`，實際推理則拒絕；規劃不預留佇列，也不保證生成成功。`preprocessing` metadata 只表示完全相同、SDK 擁有的 rendered prompt tokenization 重用，`kvReuse: false`；不是成長中的對話 prefix 或模型 KV cache 重用。

```ts
const schema = {
  type: 'object',
  properties: { greeting: { type: 'string' } },
  required: ['greeting'],
  additionalProperties: false,
} as const;
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

`describe(urlOrHtmlOrPage, options?)` 接受 HTTP(S) URL、惰性 HTML 或經驗證的 `Page` 擁有副本。`sources` 可選取段落／圖片 ID，並提供可非同步的 `paragraph`／`image` predicate；ID 須存在且不重複，predicate 接收 readonly 來源記錄。擷取與選取發生於載入模型前。預設 `mode: 'generated'` 須涵蓋所有選取來源；圖片前處理結果會重用於該階段推理。每個階段產生證據連結 claims，必要時階層縮減摘要／結論。每階段 `maxNewTokens` 預設 256；`format: 'json'` 回傳具型別報告，`'markdown'` 回傳已跳脫 Markdown。`language` 為 BCP 47；文字系統檢查只是啟發式。`onToken(text, phase)` 區分 `image`、`section`、`summary`、`conclusion`。`imageFailurePolicy: 'omit'` 保留失敗項目，但不壓下 policy、callback、取消或 budget 錯誤；預設 `'error'` 拒絕圖片失敗。

`contextWindowTokens` 會依固定模型設定檢查；預設使用保守的 4096-token 工作視窗，不代表實際可用的最大上下文。輸入與輸出預算須合計落在視窗內。報告依實際 tokenizer 分割段落並保留 Unicode 邊界；每個 section 請求最多處理四段 quote span，產生一至四個證據連結 claims，不再把任意大型來源壓成單一 claim。`sourceFacts` 獨立於生成摘要，以精確、連續引用及 UTF-16 offsets 保留所有選取段落文字。Ledger 是保留的來源文字，**不是**擷取或查核過的真實世界事實。容量足夠時，摘要／結論使用保留來源證據；不足時才階層縮減生成 claims。來源參照與成功縮減均不保證語意忠實。

生成 schema 依階段區分：sections 與中間 reduction 請求一至四個 `{ text, evidenceIds }` claims；圖片描述、`page.summary` 與 `conclusion` 請求一段精簡、證據連結的文字，可包含多個受來源支持的事實。此設計限制輸出形狀，不保證事實或完整性。Checkpoint 的 request hashes 綁定實際階段 schema。

`budget: { maxTotalTokens, maxDurationMs }` 限制累計輸入／輸出 token 與整個請求耗時，包含排隊、載入、擷取、前處理、生成，以及支援的非同步來源 predicate、資源核准、`onEvent`、`onCheckpoint`。Deadline 取消 SDK 對使用者 Promise 的等待，不會強制搶占 native ORT／OS 呼叫，也不終止呼叫端自己的外部副作用。`onEvent` 回報階段轉換；`onCheckpoint` 提供可保存、cloneable 的 state。`resume: checkpoint` 只重用相容的完成階段，保留已花費 budget，並驗證來源／模型／設定／checksum 與圖片內容版本。報告 checkpoint 建立後的失敗提供 `ReportError.checkpoint` 最終 accounting；較早的擷取／選取／載入失敗可能只有 `NekoError`。Checkpoint 保存選取文字與 metadata，不包含圖片像素；持久化資料可能敏感。

### 結構化型別、串流與對話

保留上例的 `as const`，`result.value.greeting` 才會由 `SchemaValue<typeof schema>` 推導成 `string`；required 欄位為必要屬性，其他 properties 為選填。不以呼叫端任意指定的泛型冒充驗證結果。若確實需要數值上下界等 validator 功能，須明確選擇 fallback：

```ts
const bounded = await neko.inferStructured({
  prompt: 'Return a JSON integer from 1 to 5.',
  schema: { type: 'integer', minimum: 1, maximum: 5 } as const,
  structuredMode: 'validation-only',
  maxNewTokens: 32,
});
console.log(bounded.value, bounded.structured.mode);
```

SDK 保留呼叫端提示內容；沒有公開 `_budget`、`_prepared` 或 `_structured` 選項。

```js
for await (const event of neko.inferStream({
  prompt: 'Write one short greeting.',
  maxNewTokens: 32, maxBufferedEvents: 64, maxBufferedCharacters: 1_048_576,
})) {
  if (event.type === 'token') process.stdout.write(event.text);
  else console.log(event.result.usage, event.result.finishReason);
}

const session = neko.session({
  system: 'Answer briefly.', contextPolicy: 'drop-oldest', maxHistoryMessages: 32,
  maxNewTokens: 64,
});
try {
  console.log(await session.plan('Hello.'));
  console.log((await session.send('Hello.')).text);
  const branch = await session.branch();
  try {
    const saved = await session.export();
    await branch.import(saved);
    await branch.reset();
  } finally { await branch.dispose(); }
} finally { await session.dispose(); }
```

`inferStream` 回傳有界 `AsyncIterable`，事件為 `token`／最後的 `result`；完整 usage 在 result。消費者過慢溢位為 `STREAM_OVERFLOW` 並取消生成；提前 `break`／iterator return 亦取消，不默默丟棄片段。串流在 EOF／取消前保留佇列所有權。

Session 的 `send(content, options?)`／`plan(content, options?)` 接受文字或 chat content array。操作序列化；只有成功完成的 user／assistant turn 才交易式寫入 history，規劃不提交。`contextPolicy: 'error'` 是預設；`'drop-oldest'` 只移除完整最舊 turn，保留 system。`maxHistoryMessages` 包含 system 且至多 128。同步 `history()` 提供擁有副本；branch 有獨立 history，但共用 host。`reset()` 保留 system；`export()`／`import(snapshot)` 使用 model-bound `version: 1` snapshot。圖片匯出只接受可攜 raster data URL／擁有 raw pixels；外部 URL／path 被拒絕，匯出不為此讀檔或 fetch。Snapshot 可能含敏感對話／像素，須安全保存。Dispose session 不 dispose host。每次生成仍處理整段對話；只可能重用 plan/send 的完全相同 tokenization，並非 KV／跨 turn prefix 重用。

### 結構化、報告與問答串流

`inferStructuredStream(options)`、`describeStream(input, options?)`、`askStream(input, question, options?)` 和 `askDocumentsStream(index, question, options?)` 使用對應非串流 API 的選項，另加 `StreamBufferOptions`。它們先提供尚未驗證的 `{ type: 'provisional', text }` 片段，再提供最後的 `{ type: 'result', result, usage }`。只有最後結果完成 schema／引用驗證；provisional JSON 不是答案或證據。報告片段另有 `phase`，並以 `{ type: 'stage', event }` 回報進度。`format: 'markdown'` 的最後結果是渲染完成的 Markdown，不是逐片 Markdown。

```ts
for await (const event of neko.inferStructuredStream({
  prompt: 'Return a greeting.', schema, maxNewTokens: 64,
  maxBufferedEvents: 64, maxBufferedCharacters: 1_048_576,
})) {
  if (event.type === 'result') console.log(event.result.value);
}
for await (const event of session.sendStream('Hello.')) {
  if (event.type === 'token') process.stdout.write(event.text);
}
```

Buffer 預設為 64 個事件及 1,048,576 個 UTF-16 字元；可設範圍分別為 1–10,000 和 1–16,777,216。佇列中的最後結果也計入限制。溢位以 `STREAM_OVERFLOW` 失敗並取消操作；提前 `break`、iterator return 或取消不會提供成功的最後結果。

`session.sendStream(content, options?)` 提供 `token`／`result`，不是 `provisional`。History **只有在讀取 result 後繼續迭代至正常結束時才提交**。收到 result 就 break 仍會回滾，請完整使用 `for await`。失敗、溢位或取消均不寫入 user turn 或部分 assistant 輸出；session 操作所有權保留至正常結束或取消。

### 工具選取與明確核准執行

```ts
import { defineTool, executeToolCalls, runToolLoop } from 'neko.js';

const tools = [defineTool({
  name: 'openingHours',
  description: 'Read the application-owned opening hours.',
  parameters: { type: 'object', properties: {}, additionalProperties: false } as const,
  result: { type: 'string' } as const,
})] as const;
const messages = [{ role: 'user' as const, content: 'What are the opening hours?' }];
const execution = {
  approve: (call: { name: string }) => call.name === 'openingHours',
  handlers: { openingHours: () => 'Monday 09:00–17:00' },
};
const selected = await neko.inferTools({ tools, messages, maxNewTokens: 256 });
const results = await executeToolCalls(tools, selected.toolCalls, execution);
const completed = await runToolLoop(neko, tools, {
  messages, ...execution, maxRounds: 4, maxNewTokens: 256,
});
console.log(results, completed.stopReason, completed.message);
```

`defineTool({ name, description?, parameters, result?, structuredMode? })` 驗證 schema 並持有定義。`neko.inferTools({ tools, messages, maxToolCalls?, ...inferenceOptions })` 透過結構化模型推理選取，回傳已驗證的 `toolCalls` 與 assistant `message`，不執行 handler。`maxToolCalls` 預設 8（0–64）。工具對話使用獨立的 `ToolConversationMessage` 契約，包含 `role: 'tool'` 結果，不是一般推理 chat messages。

`executeToolCalls(tools, calls, { approve, handlers, signal? })` 驗證 calls，依序核准及執行 handler。只有應用程式核准函式回傳字面值 `true` 才可執行；**模型永遠不能授權**。結果 `status` 為 `'ok' | 'denied' | 'error' | 'cancelled'`；成功輸出必須可表示為 JSON，並符合選填 result schema。Handler 自行負責副作用，且須觀察取消 signal。

`runToolLoop(neko, tools, options)` 將工具結果當作**不可信資料**送回模型。`maxRounds` 限制已執行回合（1–16，預設 4），不是推理次數。除非較早選取已無 calls，最後一個執行回合後仍會做一次終止選取。`stopReason` 為 `'no-calls'` 或 `'max-rounds'`；`rounds`、`roundResults`、`messages` 和合計 `usage` 記錄工作。最後 `message.toolCalls` 的 calls 會回傳，但**絕不執行**；不能在沒有新的應用程式明確決策下執行它們。

### 獨立 worker pool

```ts
import { createNekoPool } from 'neko.js';

const reservation = 3 * 1024 ** 3; // 應用程式估算，不是量測記憶體。
const pool = await createNekoPool({
  workers: [
    { options: { device: 'cpu' }, memoryBytes: reservation },
    { options: { device: 'cpu' }, memoryBytes: reservation },
  ],
  budget: { memoryBytes: 2 * reservation }, maxPending: 8,
});
try {
  const results = await pool.inferBatch([
    { prompt: 'Write a greeting.', maxNewTokens: 32 },
    { prompt: 'Write a farewell.', maxNewTokens: 32 },
  ]);
  console.log(results, pool.status());
} finally { await pool.dispose(); }
```

每個設定擁有獨立 worker／runtime，各 owner 可能各自載入模型。FIFO 等候請求分派給可用 owner；這**不是 tensor batching**，也不共用模型記憶體。`maxWorkers` 預設 4（1–32），非空 `workers` 陣列不能超過它。`maxPending` 預設 8（0–10,000），超額 admission 以 `QUEUE_FULL` 失敗。正整數 `memoryBytes` 預留合計須符合 `budget.memoryBytes`；這是應用程式宣告的 accounting，不是實測或 OS 強制的記憶體上限。

`pool.infer(request)` 回傳結果 promise；`pool.submit(request)` 回傳 `{ id, result, cancel, dispose }`。Item 取消／dispose 只取消該請求，不 dispose owner 或 pool。`inferBatch(requests, { signal? })` 按輸入順序回傳含 ID 的 fulfilled／rejected records，保留個別錯誤。Pool disposal 取消未完成工作並 dispose 全部 owner，務必 await。

### Decoder state 與 vision 重用

`infer`／`inferStructured` 接受 `reuse: { retainState?, state?, vision? }`。`retainState: true` 的成功結果可提供 `result.reuse.state`；將不透明 handle 作為 `state` 傳入相容且 token prefix 精確延伸的請求。`vision: true` 啟用處理後的 vision encoder features 重用。這和 normalized pixels／rendered prompt tokenization 快取不同，也不是 session 自動 KV 重用。

```ts
const retained = await neko.infer({
  prompt: 'Write a greeting.', maxNewTokens: 32,
  reuse: { retainState: true, vision: true },
});
try {
  console.log(retained.reuse, await neko.reuseCacheInfo());
  // 後續請求須符合精確 token prefix 與 compatibility key。
} finally {
  if (retained.reuse?.state) await neko.releaseGenerationState(retained.reuse.state);
}
await neko.clearReuseCaches();
```

Handle 屬於 engine，不是可攜 snapshot；已釋放、淘汰或其他 engine 的 handle 不能延續生成。相容性綁定 model／profile、schema／constraint mode、instructions、處理後圖片身分及 stop settings；文字相似不夠。結果提供 `reusedDecoderTokens`、`visionEncoderHits` 和 `visionEncoderMisses`，不保證加速。

以 `createNeko({ reuseCache: { stateEntries, stateBytes, visionEntries, visionBytes } })` 限制 entries／bytes（正 safe integers）。預設 4 個 state／512 MiB、8 個 vision entries／64 MiB。過大的 state／features 會失敗，已保留 entries 可被淘汰。`reuseCacheInfo()` 提供 accounting，engine 尚未取得時為 `null`；`releaseGenerationState(handle)` 釋放單一 state，`clearReuseCaches()` 清除兩種快取。釋放／dispose engine 會使 handle 失效。`scripts/smoke-reuse.mjs` 是明確執行的真實模型重用 exercise；檔案存在不代表已在你的後端通過。

結構化呼叫（`inferStructured` 與報告階段）在固定 system 指令至少 32 個 token、請求僅含文字且未傳入 `reuse` 時，另外保留最多四個 engine 私有的 decoder state。它們不是 handle、不改變結果形狀、與 handle 分開計入 `stateBytes` 限制（過大者略過），並以 `prefixEntries`、`prefixBytes`、`prefixHits`、`prefixMisses` 出現在 `reuseCacheInfo()`；`clearReuseCaches()` 會清除它們。

### 生成診斷與生命週期事件

```ts
import { getGenerationDiagnostic } from 'neko.js';

try {
  await neko.inferStructured({
    prompt: 'Return a greeting.', schema, maxNewTokens: 64,
    diagnostics: { capture: { maxCharacters: 2048 } },
  });
} catch (error) {
  const diagnostic = getGenerationDiagnostic(error);
  console.log(diagnostic?.code, diagnostic?.usage);
  throw error;
}
```

`infer`、`inferStructured` 和 `describe` 的 `diagnostics` 啟用錯誤 metadata：stage／attempt、finish reason、usage、輸出長度，以及可用的 JSON／schema 資訊。`true` 不擷取原始文字；`{ capture: { maxCharacters } }` 明確擷取最多 1–65,536 個字元。`getGenerationDiagnostic(error)` 回傳 metadata 或 `undefined`，不替換原錯誤。擷取的模型輸出不可信且可能含使用者／來源內容，不要無差別記錄、上傳或保存。

`createNeko({ onEvent: (event) => { /* 記錄不含內容的 metrics */ } })` 觀察 `NekoEvent`：request start／end ID、`operation`、queue／duration timing、outcome／error code、token usage、model／execution 身分，或 engine loaded timing。`NekoOperation` 包含 `infer`、`inferStructured`、`planInference`、`ask`、`describe`、`load`、`warmup`；不保證每個組合 helper 都有獨立外層事件。事件不含 prompt、生成文字或來源文件。Observer throw／reject 不影響結果，SDK 不 await observer，worker 轉送為 best-effort。Factory observer 與逐報告的 `DescribeOptions.onEvent` 不同；後者回報報告 stages。

### 報告模式、規劃、稽核與續跑

```js
const html = '<article><p>The garden opened in 1987.</p></article>';
const extractive = await neko.describe(html, {
  mode: 'extractive', sourceLanguage: 'en', format: 'json',
});
const reportPlan = await neko.planReport(html, {
  mode: 'generated', language: 'en', maxNewTokens: 256,
  retries: { section: 1, summary: 1 },
});
console.log(reportPlan.stages, reportPlan.reduction);
```

`extractive` 不載入 tokenizer／ONNX、不 fetch 圖片、不做 vision；以整段精確引用建立報告，usage 為零、backend 為 `null`。`sourceLanguage` 預設 `'und'`，不是翻譯要求；生成模式與來源語言分開。Generated／extractive 都提供 claim audit：`supported`／`contradicted`／`unknown` 是保守的 lexical quote check，**不是**語意蘊含、真實世界事實、信心或相關性判定；圖片內容不由文字 audit 證明。

`planReport(input, options?)` 回傳 `mode`、`snapshotId`、`sectionCount`、`imageCount`、`stages`、`knownInputTokens`、`maxKnownOutputTokens`。Stage 含 `id`、`phase`、`inputTokens: number | null`、`maxOutputTokens`、`maxAttempts`；未確定輸入為 `null`。`reduction.required` 為 `boolean | null`、`reduction.stageCount` 為 `null`，`estimatedDurationMs`／`totalTokensUpperBound` 也為 `null`，不能作精確總成本承諾。Generated 規劃可能取得引擎，extractive 不會。

每 phase 的 `retries` 是 0–3 次**額外**嘗試，只對 `STRUCTURED_OUTPUT`／`MODEL_OUTPUT` 授權；不是任意錯誤的自動重試。累計 budget 包含失敗生成的已花費 tokens；budget 造成不完整 JSON 為 `BUDGET_EXCEEDED`，不因 retries 重試。Resume identity 不綁定可增加的 budget／retry 授權，其他來源／模型／生成設定仍不可更換。即使 retries 為 0，只增加 budget 也可續跑尚未完成的階段：

```js
import { ReportError } from 'neko.js';

try {
  await neko.describe(html, { budget: { maxTotalTokens: 100 }, retries: { section: 0 } });
} catch (error) {
  if (!(error instanceof ReportError)) throw error;
  console.log(error.partial); // 具型別的 snapshot、sourceFacts、completedStages、usage。
  const resumed = await neko.describe(html, {
    resume: error.checkpoint,
    budget: { maxTotalTokens: 10_000 }, retries: { section: 0 },
  });
  console.log(resumed.metadata);
}
```

`ReportError.partial`／checkpoint 亦跨 worker 傳輸；partial 不是完整可驗證報告。失敗在 checkpoint 建立前可能只有 `NekoError`；續跑保留所有已花費 accounting，不重置預算。

除 `inferStream` 與 `cache.model.exportBundle` 同步回傳串流，以及同步建立 session 外，cache／backend／status 方法回傳 Promise。`cache.model.prefetch/status/clear` 僅操作選取的 pinned profile；`cache.engine.status/release` 檢查／釋放 live engine。`cache: { engine: true, engineTtlMs: 1_800_000 }` 於閒置 30 分鐘內重用引擎。`backend.detect()` 不證明 native driver／session 成功。預設 `execution: 'inline'` 只允許一個 process-global runtime owner；`'worker'` 使用真正 Node thread 或 browser module worker。各實例使用有上限 FIFO（`queue: { maxPending: 8 }`），滿額為 `QUEUE_FULL`；`queueStatus()` 提供狀態。Worker 保留 callback 順序與 typed errors／checkpoints／partial。`dispose()` 取消排隊工作、等待目前工作的安全清理、釋放資源並還原 hooks。

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

### 文件結構與有來源引用的問答

```js
const document = await extractPage(
  '<main><p>The garden opened in 1987.</p></main>', { content: 'main' },
);
console.log(document.extraction, document.containers, document.tables);
const answer = await neko.ask(document, 'When did the garden open?', { maxNewTokens: 256 });
console.log(answer.status, answer.claims);
```

`content: 'main'` 是 opt-in：只在唯一可見 main／article 可保守識別且非空時採用，否則完整 body fallback；預設 `'full'`。請查看 `page.extraction` 的 `mode`／`root`／`fallback`，不是通用 readability 演算法。`containers` 保存 parent／paragraph 關係，paragraph 可含 `containerId`／`sectionId`；`tables` 保存 caption、cells、row／column／span、header references 與 paragraph IDs。選取後的 partial table 保留空 cell 幾何，不保留被排除文字。SDK 擷取／選取持有經驗證的 Page snapshot，不信任呼叫端可變物件。

`ask(input, question, options?)` 的模型受 grammar 限制選取 paragraph IDs 與 extractive claims；SDK 建立整段精確引用及 UTF-16 offsets。未知／重複 ID 或不受引用支持的 claims 被拒絕，或結果為 `insufficient-evidence`。選取文字不會為了塞進 context 而暗中截斷；超過容量須由呼叫端明確縮小 sources。引用精確不代表答案相關、語意正確或世界事實已查核。

### 本機文件索引與精確子字串問答

```ts
import {
  createDocumentIndex, importDocumentIndex, documentFromPage, askDocuments,
} from 'neko.js';

const page = await extractPage('<main><p>The park opened in 1987.</p></main>');
const index = await createDocumentIndex([
  documentFromPage(page, 'park'),
  { id: 'hours', text: 'Monday opening hours are 09:00–17:00.' },
], { chunkSize: 1200, maxDocuments: 1000 });
const found = index.search('opening hours', { topK: 4 });
const answer = await neko.askDocuments(index, 'When did the park open?');
// 明確提供推理／規劃 host 的等價 helper：
const other = await askDocuments(neko, index, 'What are Monday opening hours?');
const restored = await importDocumentIndex(JSON.parse(JSON.stringify(index.exportSnapshot())));
console.log(found.coverage, answer.claims, other.retrieval, restored.id);
```

`createDocumentIndex(documents?, options?)` 非同步持有 `{ id, text, title?, url?, blocks? }` 文件並建立有界 BM25／CJK chunks。預設為 `chunkSize: 1200`、`maxDocuments: 1000`、`maxCharacters: 8_000_000`、`maxChunks: 100_000`。`documentFromPage(page, id?)` 串接段落文字並保留 paragraph provenance，預設 ID 為 `page.url`。`search(question, { topK?, documentIds?, maxScoredChunks? })` 回傳 hits、quotes 和明確 coverage，永遠為 `exhaustive: false`。檢索不證明整份文件缺少某資訊、相關性或完整性。

`updateDocument(document)`、`replaceDocuments(documents)` 和 `removeDocument(id)` 原子更新索引；成功 mutation 改變索引身分，舊答案因而過時。`exportSnapshot()` 回傳 version-1 immutable snapshot，可由呼叫端以 JSON 保存；`importDocumentIndex(snapshot)` 驗證身分並重建檢索結構。Snapshot 含來源文字，不含 embedding vectors，也不是身分驗證；敏感內容須安全保存。

`askDocuments(host, indexOrSnapshot, question, options?)`／`neko.askDocuments(...)` 在結構化生成前，以實際 tokenizer／context budget 規劃完整檢索 chunks。選項包含 `search`、`maxNewTokens`（預設 512，1–2048）、`contextWindowTokens`、`generation`、`signal`、`hardDeadlineMs` 和 provisional `onToken`。`retrieval` 記錄選取／context 排除 chunk IDs 和規劃 coverage，不會 fallback 提交完整文件。每個 claim 必須是**每個**引用 chunk 的精確連續子字串。SDK 引用提供 document／chunk version IDs、來源 provenance，及 canonical document text 的 UTF-16 offsets，不是原始 PDF／HTML bytes 的位置。`status` 為 `'answered'` 或 `'insufficient-evidence'`。精確引用不是事實查核、OCR 驗證或相關性保證。

### 呼叫端提供的 hybrid retrieval

Neko.js **不內建 embedding model**；固定 Qwen 生成模型不是 embedder。應用程式自行負責 embedding model 的來源、授權與完整性驗證。請提供真正由應用程式持有的 `DocumentEmbedder`；下例接收該相依物件，不捏造向量：

```ts
import type { DocumentEmbedder, DocumentIndex, Neko } from 'neko.js';

async function queryWithEmbeddings(
  neko: Neko, index: DocumentIndex, embedder: DocumentEmbedder,
) {
  const found = await index.searchHybrid('opening hours', {
    embedder, topK: 4, batchSize: 32, maxEmbeddedChunks: 4096,
  });
  const answer = await neko.askDocuments(index, 'What are the opening hours?', {
    embedder, embedding: { batchSize: 32, maxEmbeddedChunks: 4096 },
    search: { topK: 4 },
  });
  return { found, answer };
}
```

`DocumentEmbedder` 有 readonly `id`、`dimensions`（1–8192），及 `embed(texts, { kind: 'query' | 'document', signal? }): Promise<readonly ArrayLike<number>[]>`。每個輸入須依原順序回傳一個符合宣告維度、有限且非零的向量；asymmetric model 可自行套用 query／passage prefix。模型、prompt format 或任何可能改變向量的設定變更時，**必須更換 `embedder.id`**。

`DocumentIndex.searchHybrid(question, { embedder, ...searchOptions, batchSize?, maxEmbeddedChunks?, minSimilarity?, signal? })` 以 reciprocal-rank fusion（RRF）融合 lexical BM25 與 cosine 排序候選。回傳 `score` 是**融合排名分數，不是相似度**。`batchSize` 為 1–256（預設 32）；`maxEmbeddedChunks` 為 1–100,000（預設 4096），限制本次新嵌入 chunks，超額會拒絕而非暗中省略。`minSimilarity` 是 [-1, 1] 的選填 cosine 門檻，只影響 semantic candidates。Coverage 包含 semantic embedded／cached／scored counts，仍為 `exhaustive: false`。

`AskDocumentsOptions.embedder` 啟用相同 hybrid 路徑；`embedding` 提供上述 semantic settings，須同時提供 embedder。向量依 embedder ID 與內容衍生的 chunk version，快取於各 `DocumentIndex` 實例的有界 128 MiB cache；mutation 後未變 chunks 可重用。每次傳 snapshot 都會重新 import 新索引；要重用快取，須傳 **DocumentIndex 物件**而非 snapshot。每次 search 都重新嵌入 query。Hybrid retrieval 仍不保證證據完整性或 semantic recall。

### PDF 擷取與圖片 OCR

```ts
import { extractPdf, documentForIndex, ocrImage } from 'neko.js';
import { readFile } from 'node:fs/promises';

const pdfBytes = new Uint8Array(await readFile('./article.pdf'));
const native = await extractPdf(pdfBytes, { id: 'article', ocr: 'none', maxPages: 20 });
const withOcr = await neko.extractPdf(pdfBytes, { id: 'article-ocr', ocr: 'scanned' });
const imageBytes = new Uint8Array(await readFile('./scan.png'));
const scan = await neko.ocr(imageBytes, { id: 'scan', maxNewTokens: 2048 });
const sameApi = await ocrImage(imageBytes, (request) => neko.inferStructured(request));
const documents = await createDocumentIndex([
  documentForIndex(native), documentForIndex(scan),
]);
console.log(withOcr.pages, sameApi.provenance, documents.id);
```

這些 helpers 接受擁有的 `Uint8Array`、`ArrayBuffer` 或 `Blob` 資料，不接受路徑／URL；`ocrImage`／`neko.ocr` 另接受 decoded raster pixels。範例的讀檔屬應用程式 IO。`extractPdf(bytes, options?)` 惰性解析 PDF，不啟動 scripts、links、attachments 或 XFA。預設 `ocr: 'none'` 擷取原生文字，推理 usage 為零。`'scanned'` 只對沒有 native layout blocks 的頁面 OCR；`'all'` 以 OCR 結果取代每頁 native blocks。獨立 PDF OCR 須提供 `infer: (request) => neko.inferStructured(request)`；`neko.extractPdf` 自動提供 host。

PDF 預設（括號為上限）：`maxBytes` 32 MiB（256 MiB）、`maxPages` 100（1000）、`maxPagePixels` 800 萬（4000 萬）、`maxTotalPixels` 4000 萬（4 億）、`maxItems` 100,000（100 萬）、`maxTextCharacters` 200 萬（1000 萬）、`maxCells` 4096（16,384）。超額會拒絕，不暗中截斷。`renderScale` 預設 1.5（0.25–4）；可提供 `password`、`id`、`title`、`signal`。OCR 另接受 `maxBlocks`、`maxNewTokens`、`contextWindowTokens`、`hardDeadlineMs`；推理 host 的 worker hard-deadline 限制仍適用。

圖片 OCR 預設：`maxBytes` 64 MiB（最多 256 MiB）、`maxPixels` 4000 萬（硬上限）、`maxBlocks` 128（2048）、`maxCells` 512（4096）、`maxTextCharacters` 100,000（200 萬）、`maxNewTokens` 2048（最多 2048）。Raster decoding 拒絕 SVG。轉錄以實際傳入 pixels 和結構化推理執行；**不宣稱真實模型 OCR 準確度**。文字、幾何、閱讀順序與表格分組仍不可信。Native PDF blocks 為 `provenance: 'native-text'`；OCR 為 `'model-ocr-untrusted'` 和 `accuracy: 'not-verified'`。Native provenance 不保證真實性。PDF 幾何使用左上原點的 PDF points，圖片 OCR 使用 pixels。`documentForIndex(pdfOrOcr)` 建立擁有的 index payload，保留 canonical text spans 與來源 provenance。

Node 懶載入本機 `pdfjs-dist` assets 和 native canvas；自訂 `assetBase` 須為本機 `file:` 目錄 URL。Browser `assetBase` 是應用程式控制、以斜線結尾的目錄，包含 `pdf.mjs`、`pdf.worker.mjs`、`cmaps/`、`standard_fonts/`、`wasm/`、`iccs/`（預設相對 module 的 `./assets/pdf/`）。Browser raster rendering 需要**主執行緒 document／2D canvas**：在主執行緒組合 `extractPdf(bytes, { ocr: 'scanned', infer: request => worker.inferStructured(request) })`，不要在 worker 內渲染。Browser CPU/WASM 推理仍不支援；browser OCR 須有受支援的 WebGPU 推理 host。

### ROI、切片與前處理快取

```js
const cropped = await neko.infer({
  image: './photo.png', prompt: 'Describe this region.', maxNewTokens: 64,
  region: { unit: 'normalized', x: 0, y: 0, width: 0.5, height: 1 },
  tiling: { tileWidth: 640, tileHeight: 640, overlap: 0.15, maxTiles: 16 },
  maxDimension: 1280,
});
console.log(cropped.images);
```

此範例沿用前述核准 canonical path 的 host。`region` 支援整數 `pixels` 或 `[0,1]` 範圍的 `normalized`，均在 EXIF 方向校正後座標裁切。`tiling` 於 ROI 內切片，`overlap` 為 0–0.5（預設 0.15），`maxTiles` 預設 16、至多 64；整次請求最多 16 個來源圖片、64 個處理區域，超額拒絕而非漏圖。每個 decoded image 上限 40 MP，`maxDimension` 至多 1280，不放大。公開 `prepareImageRegions(source, options?, cache?)` 可取得切片；單圖 `prepareImage`／`readImage`／`loadImage` 不接受 tiling。Raw pixels 須給正確 `width`／`height`／`channels: 1 | 2 | 3 | 4`，瀏覽器亦處理灰階與 alpha，不假定所有輸入為 RGBA。

引擎重用有 byte／count 上限、SDK 擁有的 normalized pixels；每次仍重新讀取、核准來源與 digest 後才命中，不是 vision embeddings cache。`images` provenance 包含 `sourceVersionId`、`sourceWidth`／`sourceHeight`、實際整數 `region`／`normalizedRegion`、處理後 `versionId`／尺寸，以及 `preprocessing.pipeline`／`cache`／`reused`。`cache` 為 `hit`／`miss`／`disabled`，`reused` 為 `normalized-pixels`／`none`。內容 hash 不認證來源，切片也不保證辨識品質。

呼叫端接收不可信 URL 時，遠端擷取可能產生 SSRF 風險。請用 `validateDestination` 檢查每個目的地／redirect，並套用應用程式層級的 outbound network 控制；此 SDK 無法判斷特定部署中哪些私人或內部目的地安全。

`loadImage(image, options?)` 使用 Node 原生解碼器或瀏覽器 bitmap/canvas 路徑解碼擷取出的圖片。它會檢查 raster 位元組／MIME、限制輸入與解碼尺寸、套用方向、等比例縮放至 1280×1280（不放大），並輸出 PNG。SVG 和格式不符／無效內容會被拒絕。此函式只負責前處理；模型推理由 `Neko.describe()` 執行。瀏覽器遠端圖片請求仍受 CORS 限制。

`renderMarkdown(report)` 跳脫不可信文字，只為 HTTP(S) provenance 建立連結。`await validateStructuredReport(report, selectedPage?)` 稽核持久化的選取來源 snapshot／SHA-256 版本、精確引用 offsets／內容、claim spans／證據連結、完整選取來源涵蓋、圖片狀態／provenance、語言、loaded model／session 身分及累計 metadata。可另外傳入相符的選取 `Page` 比較外部來源；保存的 JSON 不必重新 fetch 即可稽核。它可檢出參照與 accounting 不一致，但不判斷 claim 是否由來源或圖片像素蘊含；雜湊亦不是來源認證。

### 版本化報告與 checkpoints

報告使用 `schemaVersion: 3`，包含 `sourceFacts` 與 `integrity: { algorithm: 'sha256', checksum }`。`metadata.coverage` 記錄選取段落 IDs／字元數、保留引用數／字元數、模型與摘要引用的 fact IDs、`conclusionBasis`（`'retained-source'` 或 `'reduced-generated-claims'`），以及 `semanticRetention: 'not-measured'`。來源涵蓋是結構 accounting，不是語意 recall 分數；claim 的啟發式 audit 也不是事實查核。

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

序列化與解析都先驗證才接受資料；report helper 可用第二個參數傳入相符選取 `Page`。`validateReportCheckpoint(value)` 亦驗證記憶體 checkpoint。Checkpoint 使用 `version: 3`、`plan: 'evidence-first-v3'`，保存引用 ledger、mode、attempts 與 authorization；只有來源／模型／不可變設定相符才可 resume。沒有版本、舊版（含 v2）或未來版本均拒絕，不自動遷移、不透過 alias 解讀。報告拒絕為 `TypeError`；無效 checkpoint 版本為 `CHECKPOINT_INVALID`。請以原始輸入重新產生。Checksum 不認證作者；來源引用、metadata 和生成文字可能敏感。

Checkpoint 的 `sectionPlan` 記錄依序排列的 source-fact ID 群組，每組一至四個，且須精確涵蓋 ledger。即使重算 checksum，驗證仍拒絕 section 引用其他群組；resume 亦核對確定性 plan 及 request hashes。請以 helper 保存 SDK 提供的 checkpoint，不手動構造保存 state。

## 快取與後端

模型 manifest 固定 Hugging Face revision、必要檔案大小與 SHA-256。每次使用快取前都會驗證；不符時會失敗，不會靜默當作 cache miss。`neko.cache.model.prefetch/status/clear` 僅操作這些固定檔案。瀏覽器 Cache Storage 仍受使用者操作與瀏覽器淘汰策略影響。

在 Node 上，每個 ONNX payload 檔案每個已安裝的 runtime 只雜湊一次：Transformers.js 每次載入會多次要求這些檔案，因此未變動的檔案（device、inode、大小、mtime 與 ctime 皆相同；ctime 無法由呼叫端設定）在該 installation 內不會重複雜湊。新行程、檔案身分改變，以及所有非 ONNX 檔案仍完整驗證。Node session 使用 `os.availableParallelism()` 個 intra-op 執行緒。在一台 Apple M 系列機器（4 效能核心加 6 效率核心）上，416 token 的 prefill 由約 213–238 提升到約 290 tokens/s，decode 由約 28 降到約 25 tokens/s；這只是單機觀察、不是保證，最佳執行緒數取決於硬體。

### 並行下載與 validator-bound 續傳

```ts
const downloading = await createNeko({
  device: 'cpu', downloadConcurrency: 3, resumeDownloads: true,
  onCacheProgress: (event) => {
    console.log(event.file, event.phase, event.loaded, event.total, event.resetReason);
  },
});
try { await downloading.cache.model.prefetch(); }
finally { await downloading.dispose(); }
```

`NekoOptions.downloadConcurrency` 限制同時下載的固定檔案數（1–16，預設 3）。`resumeDownloads` 預設 `true`，保存中斷 staging 供下次明確安裝續傳，不自動 retry 失敗請求。`onCacheProgress` 接收 `CacheProgress`：`file`、byte `loaded`／`total`、phase `'download' | 'verify' | 'resume'`、選填 `resumedFrom`／`resetReason`。Reset 原因可為 `'source-changed'`、`'resume-disabled'`、`'validator-unavailable'`、`'invalid-partial'`、`'range-rejected'`、`'validator-changed'` 或 `'integrity'`。

續傳將 partial bytes 綁定固定 source／size／hash，以及 strong ETag 或可用 Last-Modified validator，送出 `Range` 和 `If-Range`。Validator 無效、object 改變或 range 拒絕時重設 staging；完整內容仍須重新驗證大小和 SHA-256 才可正式安裝。Signed CDN query strings 只在以 object location（origin + path）匹配 partial destination 時忽略，不會略過內容驗證或授權任意目的地。同源 Hub `/api/resolve-cache/` redirect 只接受精確已註冊的固定 model／revision／file。

Node install locks 跨 process 協調；browser 安裝使用 Web Locks API，須有該 API。Locks 協調快取安裝，不協調推理 owner 或取代應用程式 network policy。Local-only 缺少／損壞資產仍會失敗，不使用未驗證 partial data。

### 離線 bundle 與診斷

```js
// source／target 是同 model／profile 的獨立 worker；先完成 source prefetch。
await source.cache.model.prefetch();
const bytes = source.cache.model.exportBundle(); // 同步 ReadableStream，不是 Promise。
await target.cache.model.importBundle(bytes);   // 亦接受 Blob。
console.log(await target.cache.model.diagnostics());
console.log(await target.diagnostics());
```

`exportBundle(signal?: AbortSignal)` 不是 async，也不接受 `{ signal }`；`importBundle(blobOrStream, signal?)` 回傳 Promise。串流匯出在 EOF／cancel 前持有 queue slot。Bundle 使用嚴格版本化 manifest、模型／profile pins、檔案大小與 digest；import 先 staging／驗證才提交，取消與損壞拒絕，保留既有已驗證及無關檔案。已損壞 cache fail closed，不會被當作一般 miss 靜默替換。匯入後可於相同設定建立 `localFilesOnly: true` host；bundle 不包含原生依賴、runtime bootstrap 或遠端頁面／圖片。

`cache.model.diagnostics()` 提供必要位元組與 storage；Node filesystem quota 無法觀測，為 `null`，不是「空間足夠」。瀏覽器數值來自 `navigator.storage.estimate()`，是 origin storage 估計而非模型記憶體保證；Cache Storage 可用性、quota、persisted 與淘汰皆由 host 決定。`diagnostics()` 僅回報允許的 readiness／backend／engine／queue／worker transport metadata，不含 prompt、圖片 bytes 或模型輸出；cache diagnostics 另包含快取 path，分享時仍須審查。

Browser bundle 只有模型資產，**不含 runtime `.mjs`／WASM**。使用 `localFilesOnly: true` 前，另行部署相符的隨附 runtime，並將請求的 WASM URL 寫入對應快取；缺少 runtime 仍會回報 offline cache miss，不會偷偷連線。現有 browser runner 會一併 seed；靜態 app／worker module 仍須可取得，這不是完整網站離線。相同 origin 的 browser hosts 共用 CacheStorage。匯出 chunks 僅持有可見 bytes，避免 transferable stream 複製上游超大的 backing buffer。

### Worker 健康與硬期限

```js
const worker = await createNeko({ device: 'cpu', execution: 'worker' });
try {
  console.log(await worker.health({ timeoutMs: 1000 }));
  try {
    console.log((await worker.infer({
      prompt: 'Write one short greeting.', maxNewTokens: 32, hardDeadlineMs: 60_000,
    })).text);
  } catch (error) {
    console.error(error);
    await worker.restart(); // 明確重建 worker；自行決定是否重新提交原工作。
  }
} finally { await worker.dispose(); }
```

`health({ signal?, timeoutMs? })` 是輕量 liveness／round-trip 檢查，不載入模型、不測量品質或證明 provider 可推理。`restart()` 僅支援 worker，需明確呼叫，沒有自動 replay。`hardDeadlineMs` 僅 worker 支援：到期終止該 worker realm，所有 pending／queued 呼叫一併失敗；不只中止單一生成。之後須 restart 才繼續。Inline 的 hard deadline／restart 為不支援；一般 signal／report budget 仍是合作式取消，不能強制中斷 native 工作。

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

**本輪 SDK 實測**，macOS／arm64：

- Node 22.23.3 CPU／default：冷文字規劃不建立 ONNX、Unicode／指數整數／unique-enum 約束生成、有界 iterator、session 淘汰／分支／重設／匯入、兩個 ROI tiles（input 152／output 24）、報告 audit 與累計預算續跑。完整 871,364,778-byte 串流 bundle 跨 worker 匯入新 filesystem cache 後，完成 local-only 真實推理；已回傳串流 abort 後仍健康。
- Registry 的 2B 兩種 profile 均完成 Node 真實文字／圖片／報告；不代表 2B browser 或品質 parity。
- 完整 Chromium 153.0.8010.12 headless，搭配 `--enable-unsafe-webgpu`：0.8B 真實文字、Unicode grammar、iterator、session、ROI 與段落引用 QA。另部署／快取 runtime，將完整匯出 Blob 匯入**空** CacheStorage；新 local-only worker 拒絕模型網路後實際生成 8 tokens。指數整數／unique-enum 結果 17 tokens；1-token 預算失敗後，以零 output retries 續跑，累計 1,212 tokens。硬期限同時使推理與排隊規劃回報 `DEADLINE_EXCEEDED`，明確 restart 恢復 health／輕量規劃；最終 browser 證據畫面已視覺確認。
- Typecheck／lint／build、130 個 Node 契約、27 個品質工具、21 個 browser 契約與 packed consumer 真實推理通過；獨立[完整四樣本品質 gate 仍未通過](quality.md#目前實測結果)。Provider 證據仍只是 loaded-session configuration，不是逐 operator／硬體 GPU 認證。


以下為前一輪平台驗證的**歷史證據快照**（2026-10-03）：macOS（Darwin 27）、arm64，於 Node 22.23.3 與 Node 26.7.0 完成全新 CPU smoke。「全新／目前改動」均指該歷史輪次，不代表上述新 API 已逐項在所有平台重驗。結果只驗證所記錄 runtime／backend／profile／輸入，不涵蓋其他列或所有 Node `>=22` 版本。

| Runtime／平台 | API／契約狀態 | 歷史真實推理證據 | 該歷史輪次的驗證 |
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
