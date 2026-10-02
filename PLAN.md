# Neko.js 專案企劃書

> **現行產品契約與驗證狀態以第 17 節為準（2026-10-02）。** 第 1–3 節說明目前產品；第 4–16 節保留原始需求、架構決策與第一階段原型歷程，均不得當作目前公開 API 或驗收狀態。Neko.js 是提供文字／圖片推理及 URL／HTML 網頁報告的可打包本機多模態 SDK，支援 Node.js 與相容 WebGPU 瀏覽器。舊 `createPrototype`、Node CPU 未驗證與 full-report 待完成等狀態已由第 17 節取代。兩種執行環境須分別選擇與驗證後端，不自動 fallback。

版本：v1.0　｜　日期：2026-10-02

---

## 1. 專案基本資料

| 項目 | 內容 |
| --- | --- |
| 專案名稱 | Neko.js |
| 遠端倉庫 | https://github.com/YueyuHoshizora/Neko.js |
| 專案類型 | TypeScript 本機推理 SDK（Node.js／瀏覽器；來源為 Git 倉庫，不發佈 npm） |
| 執行環境 | Node.js ≥ 22（ESM）及具備相容 WebGPU adapter 的安全瀏覽器 |
| 核心模型 | Qwen3.5-0.8B-ONNX-OPT，固定 revision，Q4 embeddings／decoder、FP16 vision encoder |
| 專案授權 | Apache License 2.0 |
| 文件語言 | 繁體中文＋英文雙語 |
| 倉庫現況 | 已初始化：LICENSE（Apache 2.0）、CNAME、.nojekyll（另有他用；`/doc` 不部署為 GitHub Pages） |

---

## 2. 專案目標

Neko.js 是供 Node.js 與支援 WebGPU 瀏覽器使用的本機多模態 SDK：以固定 Qwen3.5 ONNX 模型提供純文字／圖片推理，並能從 URL 或 HTML 產生具來源 provenance 的結構化網頁報告；它不會執行頁面腳本，也不是爬蟲。

具體目標：

1. 以單一 API 輸入網址或 HTML，產出涵蓋頁面文字與圖片內容的文字報告。
2. 圖片理解是核心賣點：報告須對頁面中的圖片給出可讀的描述，而非只列檔名。
3. 模型推理在本機執行並重用驗證快取；URL 擷取是明確的 outbound request，須由應用程式的網路政策控制。
4. WebGPU 與 CPU 是明確指定的後端；不承諾自動 fallback、平台間 parity 或 GPU 全算子執行。

---

## 3. 範圍

**納入**：Node.js 原生 CPU 推理、Node／瀏覽器 WebGPU backend 選擇、瀏覽器 bundle、固定模型快取與引擎快取 API、文字／圖片推理、URL／HTML 網頁報告、資料完整性與語言驗證、雙語文件。

**不納入**：TTS 應用、npm 發佈、具產品承諾的 CLI、HTTP 遠端推理服務、批次爬蟲佇列、GUI、模型訓練／微調。此 SDK 只提供單次頁面擷取與本機生成，不會執行頁面腳本。

---

## 4. 核心功能需求

### 4.1 內容理解

- **F1**　必須支援以網址或 HTML 字串為輸入，擷取頁面文字內容與內嵌圖片（`<img>`、`picture`、背景圖、og:image 等）。
- **F2**　必須支援圖片理解：對每張圖片產生文字描述，並整合進整體報告；不可只輸出圖片清單或 alt 文字。
- **F3**　必須輸出結構化報告：至少包含頁面摘要、分段重點、圖片逐張描述、整體結論；格式須可程式化取用（JSON）亦可直接閱讀（Markdown）。輸出語言由 API 參數指定（如 `language: "zh-TW" | "en" | ...`），不寫死。
- **F4**　強烈建議報告中標註描述來源（哪張圖、哪個段落），避免幻覺內容無法追溯。

### 4.2 快取機制（皆須為公開 API）

- **F5**　模型快取：GGUF 模型檔下載後須落盤快取、以 SHA-256 校驗完整性、支援預先下載（prefetch）、查詢狀態、手動清除；不得每次執行都重新下載。預設下載來源：`unsloth/Qwen3.5-0.8B-GGUF`（Hugging Face，Q4_K_M）。
- **F6**　推理引擎快取：已載入的模型實例／推理 session 須可重用，支援取得快取狀態、釋放資源、設定生命週期；重複請求不得重複載入模型。
- **F7**　快取目錄須可自訂，預設位於使用者快取目錄（如 `~/.cache/neko.js` 或對應平台等價路徑）。

### 4.3 執行後端與回退

- **F8**　後端選擇須做成 API：支援 `auto`（自動偵測）、`webgpu`（強制）、`cpu`（強制回退）等模式。`webgpu` 強制模式下若 WebGPU 不可用，允許回退到 CPU，但**必須發出警告事件**（與 F10 同一觀察管道，等級高於一般回退通知）。
- **F9**　WebGPU 不可用時必須自動回退到 CPU／原生推理路徑，回退過程不得中斷請求。
- **F10**　回退事件必須可被外部觀察（事件或 callback：偵測到什麼後端、為何回退、最終使用哪個後端）。
- **F11**　後端能力偵測結果須可單獨查詢（是否 WebGPU 可用、目前後端、記憶體狀況）。

### 4.4 應用範例（首版不納入）

- **F12**　（後續版本）提供範例程式：讀取網頁內容 → 產生描述報告 → 以 TTS 技術朗讀報告文字。TTS 實作不限定技術。**首版不做。**

---

## 5. 技術架構

### 5.1 模組分層

```
neko.js
├── core/        模型載入、推理引擎抽象、後端管理
├── cache/       模型快取（F5）、推理引擎快取（F6）
├── backend/     WebGPU 偵測、後端選擇、回退策略（F8-F11）
├── web/         網頁擷取、HTML 解析、圖片抽取與前處理
├── report/      報告產生（JSON + Markdown）
└── index.ts     公開 API 匯出
```

### 5.2 資料流

```
輸入（URL / HTML）
  → web/ 擷取頁面（文字＋圖片）
  → cache/ 取得模型與推理引擎實例（未命中則載入並快取）
  → core/ 推理（WebGPU，失敗自動回退 CPU）
  → report/ 產生結構化報告
  → 輸出（JSON / Markdown）
```

### 5.3 技術選型（候選，實作前須確認）

| 項目 | 候選方案 | 說明 |
| --- | --- | --- |
| GGUF 推理 | **已定案：node-llama-cpp（程序內建綁定）** | 須確認對 Qwen3.5-0.8B 架構與圖片輸入的支援度 |
| HTML 解析 | 輕量 DOM 解析器 | 僅解析不執行任何頁面腳本 |
| 圖片前處理 | sharp 或同級影像庫 | 縮放／正規化後再進模型 |
| 網路擷取 | Node 22 內建 fetch | 逾時、大小上限、內容類型檢查 |
| 型別建置 | tsc + ESM 輸出 | 須附 `.d.ts` |

---

## 6. 公開 API 規劃（草案）

```ts
// 建立實例
const neko = await Neko.create({
  model: "Qwen3.5-0.8B-GGUF:Q4_K_M",
  backend: "auto",              // "auto" | "webgpu" | "cpu"
  cache: {
    modelDir: "/path/to/cache", // 可自訂
    engine: true,               // 開啟推理引擎快取
    engineTtlMs: 30 * 60_000,
  },
  onFallback: (info) => { /* { from, to, reason } */ },
});

// 內容理解 → 報告
const report = await neko.describe("https://example.com", {
  includeImages: true,
  format: "json",               // "json" | "markdown"
  language: "zh-TW",            // 報告輸出語言，由 API 指定
});

// 模型快取 API
await neko.cache.model.prefetch();
neko.cache.model.status();      // { downloaded, verified, size, path }
neko.cache.model.clear();

// 推理引擎快取 API
neko.cache.engine.status();     // { loaded, sessions, memory }
neko.cache.engine.release();

// 後端狀態 API
neko.backend.detect();          // { webgpu: boolean, active: BackendName }
neko.backend.current();
```

---

## 7. 應用範例：網頁描述＋TTS 朗讀（後續版本）

> 首版不實作。以下為後續版本規劃方向：

```ts
import { Neko } from "neko.js";
import { speak } from "./tts";   // 任一 TTS 實作（系統內建或第三方 API）

const neko = await Neko.create({ backend: "auto" });

const report = await neko.describe("https://example.com", {
  includeImages: true,
  format: "markdown",
});

await speak(report.text);
```

---

## 8. 文件規劃

| 檔案 | 語言 | 內容重點 |
| --- | --- | --- |
| README.md | 中英雙語（同一檔分區塊） | 專案用途、安裝、快速上手、功能列表、API 概覽、授權 |
| /doc/zh/usage.md | 繁體中文 | 使用說明：安裝、設定、API 詳解、快取管理、回退設定、疑難排解 |
| /doc/en/usage.md | 英文 | 同上之英文版 |
| AGENTS.md | 英文為主＋繁中重點摘錄 | 執行 Agent 協作規範：程式碼約定、變更流程、**資訊安全要求（核心）** |
| CLAUDE.md | 英文為主＋繁中重點摘錄 | 引用 AGENTS.md（不重複內容），補充資訊安全重點提示 |
| SECURITY.md | **英文** | 資安細節：威脅模型、漏洞回報流程、依賴政策、模型檔完整性策略 |
| LICENSE | 英文 | Apache License 2.0（已存在） |

文件規則：

- **D1**　README.md 必須中英雙語，且明確敘述「本專案是用來做什麼的」。
- **D2**　CLAUDE.md 必須引用 AGENTS.md，不得複製貼上其內容；兩者皆著重資訊安全。語言：英文為主＋繁體中文重點摘錄（D3 同）。
- **D3**　SECURITY.md 必須以英文撰寫資安細節。
- **D4**　使用說明放 `/doc` 下，英文版與中文版各一份。
- **D5**　文件不得綁定特定 AI 工具名稱，統稱「執行 Agent」。

---

## 9. 資訊安全要求

- **S1**　網頁內容一律視為不可信任輸入：解析出的文字／圖片不得被當作指令執行（prompt injection 防護）。
- **S2**　模型檔須以 SHA-256 校驗；校驗失敗即拒絕載入並報錯，不得靜默沿用損壞檔案。
- **S3**　快取路徑須防路徑穿越；快取目錄權限限制為使用者可讀寫。
- **S4**　擷取網頁時限制通訊協定（僅 http/https）、逾時與回應大小上限；不執行頁面中的任何腳本或外部指令。
- **S5**　依賴最小化並鎖定版本（lockfile 入庫）；定期審查依賴漏洞。
- **S6**　推論過程資料不出本機；不內建任何遙測／上傳機制。

---

## 10. 里程碑

| 階段 | 內容 | 產出 |
| --- | --- | --- |
| M0 | 規格定案 | 本企劃書確認 |
| M1 | 專案骨架 | TypeScript 專案結構、建置流程、單元測試框架、GitHub Actions CI（lint＋tsc＋測試；矩陣：Node 22/24/26 × Chromium/Firefox/WebKit） |
| M2 | 推理核心 | 模型載入、模型快取（F5）、推理引擎快取（F6） |
| M3 | 網頁理解 | 網頁擷取、圖片理解、報告產生（F1-F4） |
| M4 | 後端與回退 | WebGPU 偵測、自動回退、狀態 API（F8-F11） |
| M5 | 文件 | README 雙語、/doc 雙語說明、AGENTS／CLAUDE／SECURITY |
| M6 | 交付 | 倉庫版本標記（TTS 範例列為後續版本項目） |

---

## 11. 風險與假設

| # | 項目 | 說明 | 因應 |
| --- | --- | --- | --- |
| R1 | 模型圖片輸入支援 | **已確認**：Qwen3.5-0.8B-GGUF:Q4_K_M 支援圖片輸入 | 無風險，F2 按計畫實作 |
| R2 | WebGPU 於 Node 22 的成熟度 | **已驗證**：成熟度足以跑動 WebGPU | 後端抽象層保留；回退機制照常實作（F9-F11 不變） |
| R3 | 0.8B 小模型描述品質 | 圖片描述可能較簡略或有幻覺 | 報告模板約束輸出格式；標註描述來源（F4）；提示詞工程優化 |
| R5 | 模型授權相容性 | **已確認**：Hugging Face 上 Qwen3.5-0.8B-GGUF 各版本皆標 Apache 2.0，與本專案相容 | 無風險；文件中標註模型授權來源 |

---

## 12. 驗收標準

- [ ] F1-F4：輸入任一網頁網址，可產出含圖片逐張描述的 JSON 與 Markdown 報告。
- [ ] F5-F7：模型快取的 prefetch／status／clear 皆可用；重複執行不重新下載模型。
- [ ] F6：重複請求重用推理引擎實例，status 可觀察到快取命中。
- [ ] F8-F11：`backend: "cpu"` 可強制回退；WebGPU 不可用時自動回退且事件可觀察。
- [ ] D1-D5：文件齊備且符合語言與資訊安全要求。
- [ ] S1-S6：資訊安全要求逐條落實並於 SECURITY.md 對應說明。
- [ ] 全專案 TypeScript 編譯零錯誤，附型別宣告檔（`.d.ts`）。
- [ ] CI：GitHub Actions 通過 lint＋tsc＋單元測試。
- [ ] CI 測試矩陣：
  - Node.js **22 / 24 / 26** 三版本全數通過單元測試。
  - 瀏覽器測試 **3 種**（Chromium、Firefox、WebKit，以 Playwright 執行）全數通過；內容為 **WebGPU 相容性測試**——在瀏覽器環境驗證 WebGPU 偵測與回退邏輯（F8-F11）。

---

開始執行。

---

## 13. 實作追蹤（新增；不改寫原企劃）

以上第 1–12 節完整保留原企劃內容，包括待重新核實的 R1、R2；模型自身支援圖片不代表指定綁定提供圖片推理 API，偵測到 `navigator.gpu` 不代表 GGUF 推理使用 WebGPU。

| 項目 | 狀態 | 驗證／限制 |
| --- | --- | --- |
| 原企劃轉為根目錄 PLAN.md | 已建立 | 保留所有 F1–F12、D1–D5、S1–S6、里程碑、排除項與原驗收清單 |
| 既定 node-llama-cpp 綁定能力核實 | 歷史查核，路線已被取代 | 正式版 3.22.1 的圖片／WebGPU 限制仍為歷史證據；不再作為瀏覽器實作路線 |
| M0 現行規格與整合前提 | Transformers.js ONNX 原型切換已授權，實作中 | 固定 Transformers.js 4.2.0、指定 ONNX 模型 revision 與 dtype；先實測 Node 22 同圖／同 prompt、多模態推理、後端／耗時／記憶體、快取／離線重載，再驗證瀏覽器內推理 |
| M1–M6 | 未完成 | 所有原驗收項仍未勾選；不自動 commit／push／建立版本標記 |

使用者已授權 Transformers.js 4.2.0 + ONNX 第一原型，取代 Node-only／N-API／wllama／GGUF 方案；瀏覽器原生執行仍是硬性要求。Node 與瀏覽器能力分開列示；GPU 不可用時明確報錯，只有實測成功的 CPU 路徑才可描述為可用。完整網頁報告系統與原 F1–F12 驗收仍未完成。

## 14. 已查核證據與整合前提

查核日期：2026-10-02。倉庫原先沒有 node_modules 或已安裝推理套件；查核 npm 正式發佈的 3.22.1 原始套件（含 TypeScript 宣告與原生來源），不臆測未安裝 API。

- [npm 最新正式版資料](https://registry.npmjs.org/node-llama-cpp/latest)：3.22.1，gitHead `0ad4a867b05f35def42ffc84d501e9a55077b531`。[版本標籤](https://registry.npmjs.org/-/package/node-llama-cpp/dist-tags)為 latest=3.22.1、beta=3.0.0-beta.47；不能假設已有 v4 多模態版本。
- [公開 LlamaGpuType](https://node-llama-cpp.withcat.ai/api/type-aliases/LlamaGpuType) 與[同版本原始碼](https://github.com/withcatai/node-llama-cpp/blob/0ad4a867b05f35def42ffc84d501e9a55077b531/src/bindings/types.ts#L5-L10)：`"metal" | "cuda" | "vulkan" | false`，建置選項也只有這四種；WebGPU 不在公開推理 API 中。瀏覽器的 `navigator.gpu` 與 adapter 偵測僅能表示瀏覽器能力，不得標為 node-llama-cpp 正在使用 WebGPU。
- [正式版套件](https://registry.npmjs.org/node-llama-cpp/-/node-llama-cpp-3.22.1.tgz) `dist/index.d.ts`、`dist/bindings/Llama.d.ts`、`dist/evaluator/LlamaChatSession/LlamaChatSession.d.ts`：模型載入與文字 `prompt(prompt: string, ...)` 可用；沒有影像或 mmproj 載入／推理公開 API。[維護者 2026-04-05 說明](https://github.com/withcatai/node-llama-cpp/issues/585#issuecomment-4189166187)：完整多模態支援仍在開發，將於 v4 beta 發佈。
- [llama.cpp 多模態設計](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md)：影像編碼在獨立 `libmtmd`，需語言模型 GGUF 加相對應 mmproj GGUF；只有語言 GGUF 或輸入 alt 無法完成圖片理解。可用替代方案是批准更換為支援 `libmtmd` 的 llama.cpp 本機程序（例如 llama-server），但這改變既定程序內建 node-llama-cpp 且 HTTP 伺服器排除項，不能自行採用；另一方案是等待或實作並驗證 node-llama-cpp 的真正多模態／WebGPU 綁定。原生 Metal/CUDA/Vulkan 可加速文字，不等同 WebGPU，必須另行批准規格變更。

### 14.1 模型固定版本與完整性

[Hugging Face blob 中繼資料](https://huggingface.co/api/models/unsloth/Qwen3.5-0.8B-GGUF?blobs=true) 確認模型 Apache-2.0、架構 qwen35、image-text-to-text；固定 revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`：

| 用途 | 檔名 | bytes | SHA-256 |
| --- | --- | --- | --- |
| Q4_K_M 語言模型 | Qwen3.5-0.8B-Q4_K_M.gguf | 532517120 | bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 |
| 相對應 F16 影像 projector | mmproj-F16.gguf | 204987232 | 56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453 |

必須對下載內容與載入前的快取內容實際計算 SHA-256，不以 metadata 宣稱本機檔已驗證。mmproj 可獨立完整性快取，但目前不能用指定綁定消費它。

### 14.2 共享模組契約

`Page {url:string,title?:string,paragraphs:Paragraph[],images:Image[]}`；`Paragraph {id:string,heading?:string,text:string,source:{kind:'html',startOffset?:number,endOffset?:number}}`；`Image {id:string,url:string,alt?:string,caption?:string,discoveredBy:Array<'img'|'picture'|'background'|'og:image'>}`。

`StructuredReport {language:string,page:{url:string,title?:string,summary:string},sections:Array<{heading?:string,keyPoints:string[],paragraphIds:string[]}>,images:Array<{imageId:string,url:string,description:string,source:{kind:'image',imageId:string},alt?:string}>,conclusion:string}`。

網頁解析／圖片預處理／Markdown 格式化／模型完整性快取是可獨立完成的真實模組；圖片或輸入超過安全上限須報錯，不得靜默省略。不提供假的圖片描述或空殼推理；文字推理須經原生 Qwen3.5 實測才可標為完成。F2、多模態整體報告及真正 WebGPU 推理仍未完成。

### 14.3 歷史：llama.cpp 原生 N-API 替代方案（已被瀏覽器需求取代）

本節記錄先前可行性研究，**不是現行建議或待批准切換項**。使用者後續已明確要求並授權瀏覽器內直接執行，因此不採用 N-API/addon 路線。llama.cpp 本身支援 WebGPU 與多模態；先前阻塞的是 node-llama-cpp 綁定。

- [上游 WebGPU 建置說明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#webgpu)：原生 Dawn、`GGML_WEBGPU=ON`；瀏覽器 Emscripten 路線另為選項，不必引入瀏覽器 bundle 或 HTTP 伺服器。直接 llama.cpp 的嚴格 CPU 路線必須排除 GPU devices（CLI 為 `--device none`），不能只設 GPU layers=0。現有 node-llama-cpp 路線採其真實 `getLlama({gpu:false})`。
- [固定上游來源](https://github.com/ggml-org/llama.cpp/tree/a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd)：[mtmd.h](https://github.com/ggml-org/llama.cpp/blob/a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd/tools/mtmd/mtmd.h) 實際提供 `mtmd_init_from_file`、`mtmd_support_vision`、`mtmd_bitmap_init`、`mtmd_tokenize`；`mtmd_context_params` 有 `use_gpu` 與 `ggml_backend_dev_t device`。[mtmd-helper.h](https://github.com/ggml-org/llama.cpp/blob/a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd/tools/mtmd/mtmd-helper.h) 提供編碼影像 buffer 載入與 `mtmd_helper_eval_chunks`，後者明確不是 thread-safe，綁定須序列化同一 context 並在非事件迴圈執行緒運算。
- [mtmd CMake](https://github.com/ggml-org/llama.cpp/blob/a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd/tools/mtmd/CMakeLists.txt)：public library 連結 ggml、llama；可設定 `LLAMA_BUILD_MTMD=ON`、`LLAMA_BUILD_TOOLS=OFF` 建置純函式庫，`LLAMA_SUBPROCESS=OFF` 可停用 ffmpeg 子程序影像以外功能。
- [WebGPU 原始碼](https://github.com/ggml-org/llama.cpp/blob/a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd/ggml/src/ggml-webgpu/ggml-webgpu.cpp) 含 hybrid 模型相關 SSM_CONV、SSM_SCAN、GATED_DELTA_NET，與影像相關 CONV_2D、IM2COL、ROPE／量化矩陣運算；原生 adapter 檢查 ShaderF16、ImplicitDeviceSynchronization。這是實作能力證據，**尚不是指定 Qwen3.5-0.8B Q4_K_M + mmproj-F16 在實際 WebGPU 的通過證據**。
- 查核候選 [llama-web-bridge](https://github.com/leehack/llama-web-bridge) 是瀏覽器 JS/WASM/Emscripten，與 [yzma](https://github.com/hybridgroup/yzma) 是 Go/purego FFI 及 browser WASM，不是可直接替換的 Node N-API 原生綁定。尚未查得可驗證且同時提供 mtmd＋原生 WebGPU 的現成 Node 綁定，不宣稱不存在任何其他套件。

先前曾研究自有非阻塞 N-API addon，但該方案現已被瀏覽器原生需求取代，不再實作或推薦。既有 Node cache 資料須安全保留，不再追加 Node-only 模型預下載。

## 15. 歷史：wllama 瀏覽器架構研究（已被 Transformers.js ONNX 原型取代）

- 必須可直接在瀏覽器頁面執行，不依賴 Node 原生 addon、HTTP 推理伺服器或雲端服務。靜態 JS/WASM/worker 資產可由同源靜態網站提供；這不構成外部推理服務。
- 候選穩定套件 [@wllama/wllama 3.6.1](https://registry.npmjs.org/@wllama/wllama/latest) 與 [@wllama/wllama-compat 3.6.1](https://registry.npmjs.org/@wllama/wllama-compat/latest)，同一來源提交 `c35450cf9597eaf901293b12458cae204aea0b65`，無執行期套件依賴。已查核 npm 實際包內 ESM、8.1 MB WASM 與公開 TypeScript API；底層 llama.cpp 提交為 `83d855c5a6d70487121edbf4020b25c96b7a04e7`。正式選用仍須指定模型與圖片實測。
- [同版本多模態範例](https://github.com/ngxson/wllama/blob/c35450cf9597eaf901293b12458cae204aea0b65/examples/multimodal/index.html)：`new Wllama({default:wasmUrl})`，`loadModel(File[])` 載入語言 GGUF 與 mmproj；`createChatCompletion` 使用 `content:[{type:'image',data:ArrayBuffer},{type:'text',text:prompt}]`。JSON schema、取消 signal 與 `exit()` 皆為實際公開 API。
- 本地資產契約：`dist/index.js` 瀏覽器 ESM；`dist/wasm/wllama.wasm`；`dist/wasm/compat/wllama.wasm` 與 `dist/wasm/compat/wllama.js`。立即以明確本地 URL 呼叫 `setCompat({wasm,worker},'firefox_safari')`，覆蓋建構子預設 CDN 設定，不隱式載入第三方 runtime。
- 嚴格 CPU 路線已由 [同版來源](https://github.com/ngxson/wllama/blob/c35450cf9597eaf901293b12458cae204aea0b65/src/wllama.ts) 核實：`n_gpu_layers:0` 在 module 初始化前設定 `workerResources.noWebGPU=true`；並以 `mmproj_offload:false`、`no_kv_offload:true`、`offload_kqv:false` 避免 projector/KV GPU offload。adapter 可用性不得直接標成推理已使用 WebGPU。
- 模型快取改為瀏覽器 OPFS，以 WebCrypto SHA-256 與固定大小核實後才交付 File；下載串流寫暫存，避免累積整份 chunk 陣列。WebCrypto 雜湊須取得該單一檔案的 ArrayBuffer，因此峰值仍受固定最大模型大小約束；不能聲稱零記憶體或增量 WebCrypto。
- session/context 由頁面／worker 活體快取與序列化請求管理；DOMParser、fetch、瀏覽器 raster decode/Canvas 前處理保持無 Node 依賴。跨來源網址與圖片仍受瀏覽器 CORS 限制，不以伺服器繞過。
- 待驗證：指定 Qwen3.5-0.8B Q4_K_M + mmproj-F16 真實圖片理解、CPU WASM、WebGPU 真實執行／回退、Chromium／Firefox／WebKit 表面、重複請求無頁面殘留、模型快取 reuse／清理、live session TTL／釋放。未通過前保留原驗收清單未勾選。

## 16. 歷史原型驗證紀錄（已由第 17 節的 SDK 契約取代）

本節保留從原生 GGUF／wllama 設計轉向 Transformers.js／ONNX 的決策與第一階段原型證據；`createPrototype`、CPU 未驗證及 full-web-report 未完成等描述只記錄當時狀態。不可用作目前公開 API、功能範圍或驗證狀態。

- 精確依賴 `@huggingface/transformers@4.2.0`；[npm 實際版本資料](https://registry.npmjs.org/@huggingface/transformers/4.2.0) 顯示 Node／瀏覽器條件匯出，依賴 `onnxruntime-node@1.24.3` 與 `onnxruntime-web@1.26.0-dev.20260416-b7804b056c`。不混淆 npm 最新版與使用者固定版本。
- Node 原生 runtime 精確 override 為 `onnxruntime-node@1.30.0`（正式版，非 dev）；使用者已批准此最小相容性前提。實際 1.24.3 在 GPU 與 CPU 載入同一固定 decoder 均缺少 `com.microsoft:CausalConvWithState(-1)`，不能生成。Transformers.js 4.2.0、模型 revision／dtype 與 browser `onnxruntime-web@1.26.0-dev.20260416-b7804b056c` 保持不變；新 runtime 必須重新實測，不因升版即聲稱成功。
- 固定模型 `onnx-community/Qwen3.5-0.8B-ONNX-OPT`、revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`；[Hugging Face blobs metadata](https://huggingface.co/api/models/onnx-community/Qwen3.5-0.8B-ONNX-OPT?blobs=true) 提供圖與外部權重／tokenizer 的大小與 SHA-256。小型 processor／config／chat template／tokenizer config 檔案須下載固定 revision 並計算 SHA-256，不以 Git SHA-1 blobId 冒充。
- 精確 dtype：`embed_tokens:'q4'`、`decoder_model_merged:'q4'`、`vision_encoder:'fp16'`。包含對應 ONNX graphs／external data、processor、preprocessor、tokenizer、chat template、generation config；不可只快取單一 ONNX。
- 已讀範例實際使用 `AutoProcessor`、`Qwen3_5ForConditionalGeneration`、`RawImage`、`TextStreamer`；chat template 設定 `enable_thinking:false`，真實影像經 processor 後交予 `model.generate`，不得以 alt、metadata 或 mock 文字取代。
- 固定 4.2.0 的 tokenizer discovery 會忽略 revision，以 `{}` 查詢 `main` 的 `tokenizer_config.json`。嚴格快取已實際拒絕此未固定 URL；改用同套件公開 `TokenizersBackend`、`Qwen3VLProcessor` 與固定 revision 的 `AutoImageProcessor` 組裝 processor，資源仍全部經大小／SHA-256 核實，不放寬 guard 或重導向 `main`。
- API 為明確原型 `createPrototype(...)`、`infer({image,prompt,...})`、`dispose()`，支援實際使用者圖片／prompt；不聲稱原網頁 StructuredReport 系統已完成。Node 22 runner 接受同 Ollama 測試的 `--image`／`--prompt`（或 prompt file），記錄真實 runtime／provider、首載、前處理、首 token、生成／總耗時、可觀察記憶體與 live session reuse。
- GPU 設定明確選擇，不提供無條件 `auto -> cpu` 承諾。Node 的實際 ONNX Runtime native WebGPU 與瀏覽器 WebGPU 分開；`navigator.gpu` 或獨立 Dawn adapter 存在不能證明 Node ONNX session 使用 GPU。記憶體不可觀察的部分標為 unknown/null，不能以檔案大小偽裝 GPU RAM。
- 全資源固定 revision、大小與 SHA-256 核實；實際離線重載使用 `local_files_only:true`，沒有網路時必須讀到完整有效快取，缺少／損壞則明確失敗；browser runtime 的 WASM／JS 也須本地提供及快取，不能隱藏 CDN 依賴。
- Ollama 使用的同一圖片與 prompt 尚未提供於目前對話／倉庫；runner 必須可接受它們，但不得聲稱已完成同 fixture 對比。可先用另一張實際圖片 smoke，必須標明此限制。

下方各子節是截至相應日期的 prototype 測試紀錄；其中對 CPU、WebGPU 與網頁報告仍待完成的舊狀態，已由第 17 節最新 SDK／consumer 驗證取代。未提供同一 fixture 的 Ollama output，因此 parity 仍未知。

### 16.1 已執行的後端診斷（不等同 Qwen 圖片驗收）

Node **22.23.3**、darwin arm64 的實際 `onnxruntime-node@1.24.3` WebGPU session 已執行 [Microsoft 同版 mul_1.onnx 130-byte fixture](https://github.com/microsoft/onnxruntime/blob/v1.24.3/onnxruntime/test/testdata/mul_1.onnx)，關閉 CPU EP fallback，輸出 `[2,4,6,8,10,12]` 正確，session 已釋放。實際 profiling trace `/tmp/neko-native-probe/verified-gpu_2026-10-02_18-31-42.json` 包含 `mul_1_kernel_time` 的 `provider:"WebGpuExecutionProvider"`，不是單純 `session.config` 或 navigator capability。

此證據僅證明該環境能執行原生 ONNX WebGPU 小型運算，不證明完整 Qwen vision／decoder 相容、圖片理解成功或 GPU 記憶體量。先前 cache wrapper 的 Node-only `ReferenceError: name is not defined` 與 tokenizer 未固定 revision 問題已修復並乾淨重建；隨後實際 GPU／CPU 均到達 decoder 載入而因 1.24.3 缺少上述 operator 失敗。不得將這些診斷視為模型通過。

### 16.2 已執行的完整快取與影像前處理（不等同離線生成）

- 首次含下載的真實 Qwen 執行在 36.74 秒後因未固定 tokenizer probe 失敗，但 13 個固定 ONNX／外部權重／tokenizer／processor 資源均已下載及驗證；資料安全保留於測試 cache，不追加 GGUF 預下載。
- Node 22.23.3 新程序、`local_files_only:true` 且 fetch 主動禁用：完整 13 檔離線 prefetch 再驗證成功，0 次網路呼叫，0.350 秒。另驗證重複安裝會在任何 WASM 環境變更前拒絕。
- 真實另一張足球照片（800×533，**不是尚未提供的 Ollama fixture**）經上述 pinned processor 離線前處理成功：`input_ids=[1,450]`、`pixel_values=[1700,1536]`、image grid `[1,34,50]`，0 次網路呼叫，前處理 66.96 ms。
以上快取／前處理證據本身不等同完整模型離線生成；後續原生 runtime 1.30.0 的完整生成證據如下，瀏覽器不得據此推定通過。

### 16.3 實際 Node 22 圖片生成／provider／離線／live reuse

- 使用 **Node 22.23.3／Transformers.js 4.2.0／onnxruntime-node 與 common 1.30.0**，固定同一 ONNX revision、q4／q4／fp16，真實足球照片與 prompt「Describe the visible scene and the people in this photograph.」生成成功，描述足球場景、PSG 球衣及前景球員等；不是 alt／metadata／mock 描述，亦不是未提供的 Ollama 同 fixture 對比。
- metadata 修復後、權重已在驗證 cache 的真實執行（**不是包含首次下載的冷啟動**）：外部載入 **6416.68 ms**、首個非空解碼串流片段 **1487.16 ms**、生成 **9353.70 ms**、單次 infer **9486.62 ms**。後兩者不包含載入；預設 128 新 token 截斷仍可觀察到真實文字輸出，0 次遠端網路呼叫且無 metadata warnings。先前 2.424 秒是較早已快取資源的引擎初始化，不能當作冷啟動。
- 實際 profiling provider events：embedding **WebGPU 128**；vision **WebGPU 536／CPU 50**；decoder **WebGPU 169472／CPU 12288**。這是 **hybrid GPU／CPU 真實執行**，不是只有 session 設定，也不代表所有算子在 GPU。當前 source 修復後 trace 為 `/private/tmp/neko-qwen-metadata-fixed*.json`。
- 該 cached 執行實際 `process.memoryUsage().rss`：載入後 **875,462,656 bytes**、生成後 **1,394,081,792 bytes**、dispose 後 **833,683,456 bytes**。RSS 是整個 Node 程序的 resident memory，不能當作 GPU RAM；GPU 記憶體維持 `null`／unknown，dispose 不宣稱 RSS 歸零。
- **最終目前 source 的新程序完整離線生成成功**：同時將 global fetch 與 Transformers env.fetch 改為計數後拋錯，使用本地 image Blob、`local_files_only:true`，從下節新 cold cache 完整重載；外部載入 **4655.13 ms**、首可見片段 **2447.61 ms**、生成 **9874.19 ms**、infer **9997.88 ms**，生成同一照片描述，**0 次嘗試網路呼叫**，無 metadata warnings。trace 為 `/private/tmp/neko-qwen-final-offline-profile*.json`；RSS 載入／生成／dispose 後分別 **1,248,968,704／1,115,258,880／811,778,048 bytes**。這才是完整模型離線重載／生成證據，不是單獨 prefetch。
- 另一個完全禁用網路的真實新程序，以同一活體 prototype 連續詢問球衣主色與運動類型，分別回應 **「Blue」／「Football」**；status **loads=1／hits=2／sessions=1**（一個引擎 context，並非一個 ONNX graph），0 次網路呼叫，最後已 dispose。觀察到第二題採用新 prompt，未延續第一題回答。

### 16.4 真正空模型 cache 的首次下載＋載入

- metadata Range／HEAD 修復後再建立 **新的空暫存 cache `/private/tmp/neko-qwen-final-cold-1Ue70B`**，保留前述完整 cache、browser bytes 與既有使用者資料。Node 22／固定 1.30.0 GPU 原型成功下載及核實 **13 檔、871,364,778 bytes**，並完成同一真實圖片生成。
- 外部計時器在 `createPrototype` **之前**開始：完整下載＋SHA／大小核實＋模型／session 初始載入為 **37,118.82 ms（37.12 秒）**；不是先前已快取資源的 2.424 秒。之後首可見文字片段 **1228.45 ms**、生成 **7728.60 ms**、infer **7796.94 ms**。這是目前 source 的成功實測；舊的 36.376 秒含 metadata warnings，不作為最終版本的首次載入數據。
- 實際 RSS：載入後 **2,147,352,576 bytes**、生成後 **1,966,080,000 bytes**、dispose 後 **731,152,384 bytes**；GPU RAM 仍不可觀察。實際 provider trace 為 `/private/tmp/neko-qwen-final-cold-profile*.json`，仍為 embedding GPU128、vision GPU536／CPU50、decoder GPU169472／CPU12288，證明 hybrid 推理。
- 此次觀察到 **13 次遠端 fetch 呼叫**，0 個 ModelIntegrity metadata warnings；不壓制 logger。4.2.0 的 `fetch_file_head` 實際使用 `GET Range: bytes=0-0`，現在 metadata Range 與 HEAD 僅在完整 cache 檔大小／SHA 驗證後回覆正確 metadata，部分 response 不標記為完整驗證、不能 PUT 提升為完整資源。七個固定 cache 回歸測試及實際六個 ONNX graph/data metadata 呼叫已覆蓋正確大小、0 網路、損壞／缺少拒絕與 partial PUT 拒絕。

### 16.5 真實 Browser UI 推理與同 profile 離線重載（managed Browser tool）

- 使用持久化 headed Chromium 153、專案 demo UI、實際上傳足球照片與真實推理。WebGPU 輸出為：`A football player in a Paris Saint-Germain uniform is dribbling the ball while being pursued by a defender in a dark blue kit.` 瀏覽器回報 Apple／Metal 3、`fallback:false`、`shader-f16:true`；所有已載入 ONNX session 設定為 WebGPU。這是實際模型推理證據，但不宣稱每個運算子都在 GPU，也未量得 GPU RAM。
- UI 首次推理 load／preprocess／第一個非空 decoded 輸出／generation／total：**5238.6／271.6／3963.9／4717.6／4989.2 ms**。同一 profile reload 後，先勾選 `localFilesOnly` 再重新上傳並推理，輸出相同；觀察到 Hugging Face 網路請求 **0**，Cache Storage **894,993,408 bytes**。離線 run timings：**6192.4／483.9／4748.3／5778.5／6262.4 ms**。Screenshot：`/tmp/neko-browser-tool-offline-verified.png`（在 repo 外）。
- 獨立 headed Chromium smoke 的 WebGPU 計數器 delta：`dispatchWorkgroups` **+2,656**、`dispatchWorkgroupsIndirect` **+0**、`queue.submit` **+1,104**；同次 offline reload model request attempts **0**，Cache.put errors **0**。這些計數器來自 smoke instrumentation，不與 managed Browser UI run 混稱。
- 同一 demo UI 明確選 CPU 的試跑在建立 session 時失敗：`Could not find an implementation for GatherBlockQuantized(1) node with name '/model/embed_tokens/Gather_Quant'`，此固定模型的 browser CPU/WASM 路徑已確認不支援，沒有 CPU 推理輸出且不會 fallback。Node CPU 路徑與此分開，除非另有真實端對端證據，仍為未測。Headless adapter preflight 不替代上述模型推理證據。
- 修改 demo 選單為 `CPU/WASM — unsupported by this pinned model`，`smoke:browser -- cpu` helper 也會先明確警告 GatherBlockQuantized(1) 失敗且不會 fallback。以本機 HTTP 實際載入修改後頁面、選取 CPU，確認選單值及畫面可見文字一致；managed Browser screenshot：`/var/folders/ds/95v_ts_d1sq_lrjt8_m7lpfc0000gn/T/omp-sshots-15966a8a6370830f.webp`（repo 外）。
本節所記錄的是先前 prototype 的 browser inference/cache 證據，不等同 Ollama parity；目前 SDK 的 WebGPU demo smoke 與實際輸出限制見第 17 節。

## 17. 現行工作樹 SDK 契約與驗證（非發佈紀錄）

本節描述目前工作樹及 `Unreleased` 的 SDK 行為與實測；`package.json` 仍為 `1.0.0`，本機 `npm pack` 產生同名版本 tarball 不代表 GitHub 已發佈此更新版，也不更改版本、tag 或 release。

### 17.1 現行範圍與 API

- 公開入口 `createNeko()` 提供 Node.js 22+ 原生 CPU，以及明確選擇 WebGPU 的瀏覽器路徑；Node 與瀏覽器透過條件匯出及獨立 bundle／runtime 資產支援。瀏覽器 CPU/WASM 不支援固定模型且不會 fallback。
- SDK 提供文字／圖片推理、URL／HTML 擷取與結構化報告、模型／後端狀態、顯式模型／引擎快取控制、逐階段 token callback、取消與 disposal。Node 端把 `onnxruntime-node@1.30.0`、`sharp@0.34.5`、`parse5@8.0.1` 列為固定直接執行依賴；Transformers.js 4.2.0 隨 bundle 打包。
- 結構化報告保留所有段落／圖片來源 ID，驗證完整產生與生成欄位語言；已觀察的英文／繁中異常字元系統回報 `LANGUAGE_MISMATCH`，截斷回報 `INCOMPLETE_GENERATION`，不自動 retry。`imageFailurePolicy:'omit'` 僅處理圖片 inference failure，不吞掉呼叫方 callback 例外或 abort（包括 callback `throw undefined`），並保留正確的來源錯誤。
- 這些語言檢查是啟發式，不保證任何語言的流暢度或事實正確性；報告圖片描述也不可當作顏色等視覺 ground truth。

### 17.2 工作樹實際驗證

- 最終工作樹 `npm run lint`、`npm run typecheck` 通過。`npm test` 在 Node **22.23.3**（亦於 Node 26.7.0 執行）會重建 bundle 並執行 **22/22** Node tests。
- `npm run test:browser` 對 Chromium、Firefox、WebKit 跑過 **15/15** browser contract tests，覆蓋後端明確性、圖片前處理限制及不可信報告 Markdown 安全性。
- `npm run test:package` 在 Node 22.23.3 與 26.7.0 將實際 `npm pack` tarball 安裝至隔離 consumer 並呼叫 Node native CPU inference 成功：文字答案 `4`、本機圖片幾何描述為 square，HTML report 保留頁面中的 red／blue source fact 及圖片 provenance。
- Node 22.23.3／原生 CPU 的實際階層報告驗證通過：**300 段落、15,811 bytes、4,391 source tokens**，`contextWindowTokens:1536`／`maxNewTokens:512`。生成前最小來源 ID 證據已需 **1506 input tokens**；實際中間摘要 prompt 為 **1848 input tokens**，加上輸出預算均超過視窗，因此實際執行多階段縮減。報告產生 **8 sections**、摘要及結論，保留所有 **300 paragraph IDs**。較小的 1024／256 設定曾在 summary 以 `INCOMPLETE_GENERATION` 明確拒絕。
- 上述階層案例的摘要遺漏部分 garden plot 範圍，結論更否定來源中實際存在的 oak／pond 資訊。這是分塊／縮減流程完成與來源 ID 完整性的證據，不是語意忠實、摘要完整性或模型單獨責任的證明。
- Node packed report 圖片描述將飽和紅／藍影像錯述為淡紫／淡粉色。先前同一 headed 持久化 Chromium WebGPU demo 跑過 image、text、report 與同 profile 離線 reload；離線時模型及 Hugging Face 請求 0、Cache Storage 約 **895 MB**、Cache.put errors 0、WebGPU dispatch／queue submit 計數非零。觀察到的色彩差異描述只限這個固定模型／runtime／fixture；未隔離原因且沒有同 fixture Ollama 輸出，因此不宣稱是模型單獨責任或 Ollama parity。
- `npm pack` archive 的 consumer 成功證據是從目前工作樹產生的本機 artifact，不代表發佈已完成。
