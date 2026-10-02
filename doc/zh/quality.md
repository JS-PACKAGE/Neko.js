# 品質基準測試

本儲存庫中的基準測試會以固定且可重現的輸入，對固定版本的 SDK 模型執行三類測試：五項事實的文字樣本、點陣化圖片樣本，以及含 300 個段落的階層樣本。測試會實際呼叫 Node 或瀏覽器 SDK，並保留各案例的原始 SDK 結果與錯誤。這是依據有限 oracle 設計的回歸基準測試，不是通用事實性測試，也不保證模型輸出正確。

## 執行

```sh
MODEL_CACHE=/absolute/path/to/verified-node-cache

# Node、CPU、default profile；除非提供下方旗標，否則採離線模式
npm run quality:benchmark -- --runtime node --device cpu --model-profile default \
  --cache-dir "$MODEL_CACHE" --output artifacts/quality-node.json

# 瀏覽器、WebGPU；使用專用且持久的 Chromium 使用者資料目錄
npm run quality:benchmark -- --runtime browser --device webgpu \
  --cache-dir "$MODEL_CACHE" --browser-profile .cache/neko-quality-chromium \
  --browser-port 4173 --output artifacts/quality-browser.json

# 評估器本身的確定性測試（不會執行模型）
npm run test:quality
```

預設不會啟用 `--allow-network`。Node 執行會使用所選的已驗證檔案系統快取。離線瀏覽器執行必須提供 `--cache-dir PATH`：runner 會驗證該 Node 快取，只透過暫時的 loopback mirror 提供所選 profile，將完整 Response 寫入 SDK 實際使用的 `env.cacheKey` CacheStorage，並寫入固定版本 runtime WASM，然後啟用 `localFilesOnly`、讓 Chromium 進入離線狀態並封鎖外部請求。快取缺少檔案或未通過驗證時不會推理；不應有任何 HF 模型請求。產物會記錄快取與網路證據。只有在明確同意 SDK 使用網路模型來源時，才傳入 `--allow-network`。不會自動重試。SDK 會依其固定 manifest 驗證快取檔案。

`--cache-dir PATH` 用來選擇 Node 檔案系統快取，也是離線瀏覽器執行時已驗證的 seed 來源；模型推理仍使用瀏覽器 Cache API。瀏覽器快取依來源 origin 區隔：請使用專用的 `--browser-profile` 目錄，並在不同執行間維持相同的 `--browser-port` 才能重用快取。預設連接埠為 `4173`，必須可供使用；若已被占用，請用 `--browser-port` 指定另一個固定連接埠。未指定 `--browser-profile` 時會使用短暫的瀏覽器 context，快取不會保留供下次執行使用。請勿將個人瀏覽器 profile 當作 `--browser-profile`。

其他選項包括 `--case text|image|hierarchy`（只執行一種樣本）、`--model-profile default|all-q4`、`--context-window-tokens N`、`--max-new-tokens N` 及 `--headed`。可用 `--profile-prefix PATH` 要求原生 ONNX profiling（僅 Node）。`--output PATH` 會寫入完整 JSON 產物並在標準輸出顯示簡短摘要；未使用時則將完整產物印至標準輸出。`--help` 會列出 CLI 選項。

## 閱讀產物

每次執行都會記錄 `stimulusKey`，其內容雜湊涵蓋樣本雜湊、實際輸入／提示雜湊、預期紀錄 ID，以及生效中的 token／context 限制；它代表模型輸入，而非評分規則。`oracleSha256` 與 `evaluatorVersion` 會另外標示評分輸入與語意。`caseInputs` 會列出每個提交輸入的 SHA-256 與生成預算。執行環境設定則記錄 runtime、device、profile 及網路許可狀態；因此 stimulus key 相同不代表 runtime 或 backend 相同。離線重新評分的產物也會以 `rescoreSource` 記錄先前的評估器版本與 stimulus key；原始模型輸出、錯誤及 `caseInputs` 均保持不變。

產物包含：

- 模型 ID／revision、要求與觀察到的模型 profile、要求的 runtime／device，以及觀察到的 backend。推理結果直接回報 profile／backend；結構化報告則從 `metadata` 回報。若未回報 profile 或與要求不符，該案例會標記為 `profile-unverified`、整體結果不可比較，並停止後續案例，不會悄悄改用其他 profile。
- 樣本來源／點陣圖雜湊、擷取的 source ID 數量、快取狀態、有界的逐檔／階段進度摘要、適用時的瀏覽器快取 seed 與外部請求證據、環境版本，以及（若有要求）原生 ONNX trace 摘要。
- 每項已嘗試推理／報告的原始 SDK 輸出，以及失敗或封鎖案例的結構化錯誤資訊。請搭配評估指標檢視原始輸出；指標不能取代模型實際產生的文字或報告。

只有所有選取案例均完成且樣本完整性有效時，`status` 才會是 `completed`；推理失敗、profile 未驗證或離線網路政策遭違反時為 `incomplete`；案例未執行時（例如離線但缺少快取）為 `blocked`；樣本完整性驗證失敗時為 `fixture-invalid`。只有完整完成且符合網路政策的執行結果才會令 `comparable` 為 true。要進行有意義的跨執行比較，必須確認 `stimulusKey`、模型 revision/profile、觀察到的 backend 身分一致，並一併考量產物記錄的 runtime／瀏覽器版本。

圖片案例會將相同、經 manifest 固定的 PNG bytes 傳給 Node（`Blob`）與 Browser（data URL）。`imageRenderer.rasterSha256` 記錄點陣圖 bytes；`caseInputs.inputSha256` 則對實際提交的 data-URL 字串計算雜湊。不過 SDK 圖片 `versionId` 是對解碼後的像素、尺寸及 channel 數量計算雜湊：Node 使用 Sharp 產生不含 alpha 的 sRGB raw pixels；Browser 則使用 `createImageBitmap` 與 Canvas `ImageData`（RGBA）。因此，編碼後的 PNG bytes 相同，不代表解碼／前處理表示或 `versionId` 相同。跨 runtime／profile 的分數差異不可單獨歸因於量化；本基準沒有隔離 decoder 或 preprocessing 的影響。

## 指標定義

### 文字與圖片案例

評估器會依有限的 regex oracle 評分含有明確 ID 的宣稱行。`FACT` 前綴不影響事實評分；省略前綴會記錄在 `formatCompliance`，不會把符合預期的事實算成錯誤宣稱。未包含可辨識明確 ID 的非空行屬於格式錯誤，而非錯誤事實。true positive 必須符合該 ID 的預期特徵，且不得重複、遭矛盾規則否定或命中 unsupported 規則。其他受支援事實可作為上下文，不會單獨使符合預期的宣稱失分；只有指定 ID 未符合自身預期事實、但符合另一個已知事實時，才列為跨 ID false claim。未知 ID、重複 ID、內容不符、矛盾及 unsupported pattern 命中仍列為 false-positive claims；未獲 true-positive 計分的預期 ID 則列為 false negative。

`claimPrecision = TP / (TP + FP)`，`factRecall = TP / (TP + FN)`；分母為零時，指標值為 `null`。`formatCompliance` 會將缺少前綴及無法解析的行與事實指標分開回報。產物會列出宣稱行、格式錯誤、遺漏事實、矛盾及 unsupported-pattern 命中。負向的 peppers 事實本身是預期事實；反之，對正向事實加上否定語氣時不應獲得計分。圖片位置 oracle 接受「圓形在正方形左側」或等價的「正方形在圓形右側」；相反關係與明確否定的關係不符合預期。

這些是刻意限制範圍的詞彙比對，不是語意蘊涵判斷，也不會抽取自由格式宣稱。Regex 僅涵蓋文件所述的樣本詞彙。改寫、否定範圍、指代消解、混合陳述及否定詞都可能被誤判；矛盾／unsupported 偵測與否定範圍皆為 heuristic，無法涵蓋所有語言表達。高分不能證明任意陳述為真；低分也無法單獨判定模型失敗原因。

### 階層案例

`sourceIds.recall` 與 `sourceIds.precision` 衡量實際保留的來源段落 ID（唯一 ID）；產物另行列出遺漏、多餘及重複的 ID。有限範圍的紀錄評估器只有在符合以下條件時才會將一筆紀錄算作命中：某個 section 引用了預期 source ID，而且該 section 的 `keyPoints` 同時符合紀錄 ID、地點、欄位、數值及單位的 regex。`recordSignatureMetrics.factRecall` 是命中的預期紀錄數除以全部預期紀錄數。`recordSignatureMetrics.claimPrecision` 是通過完整特徵及引用檢查的紀錄 ID 宣稱，占所有輸出紀錄 ID 宣稱的比例；重複、未知、未連結或內容不符的紀錄 ID 不會通過。

這些紀錄指標不會解析或認證任意報告文字、oracle 以外省略的事實，也不判定未以預期紀錄特徵表示之宣稱的真偽。保留 source ID 只衡量引用涵蓋率，不代表事實正確。
