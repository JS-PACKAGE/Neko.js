# 品質基準測試與閘門

本基準以四種固定輸入執行固定版本的 SDK 模型：五項花園事實、圖片幾何與顏色、八筆邊界紀錄，以及 300 筆有日期的測站觀測。會保留原始 SDK 輸出與錯誤。這是有限 oracle 回歸測試，不是通用語意蘊涵、事實正確性證明或 prompt injection 防護。

## 執行

請先執行 `npm ci`（其 `prepare` 會建置 SDK）；來源修改後執行 `npm run build`。基準與閘門使用編譯後的固定模型 manifest。確定性品質測試不會下載或執行模型。

```sh
MODEL_CACHE=/absolute/path/to/verified-node-cache
npm run quality:benchmark -- --runtime node --device cpu --model-profile default \
  --cache-dir "$MODEL_CACHE" --output artifacts/quality-node.json
npm run quality:gate -- --input artifacts/quality-node.json \
  --output artifacts/quality-node-diagnostics.json

# 評估器、程式化閘門、比較與 CLI 的確定性回歸測試，不執行模型
npm run test:quality

# 瀏覽器 WebGPU，使用專用 Chromium 持久目錄
npm run quality:benchmark -- --runtime browser --device webgpu \
  --cache-dir "$MODEL_CACHE" --browser-profile .cache/neko-quality-chromium \
  --browser-port 4173 --output artifacts/quality-browser.json
```

基準完成或吞吐量高，**不代表品質合格**。閘門輸出結構化 JSON，可透過 `--output` 保存；格式錯誤、過期、不完整、不可比較或未達門檻時會以非零狀態結束。會重新評估原始 SDK 輸出，不信任已保存的分數。預設必須包含**全部四種**案例。`--cases text,image` 僅限縮診斷範圍，不代表完整發行驗收。基準可用 `--case text|image|boundaries|hierarchy` 選擇單一案例，預設為 `all`。

Node 預設 CPU，Browser 必須使用 WebGPU。能啟動瀏覽器不等於 provider 正確或模型品質通過；此處沒有宣稱新增平台驗證。預設禁用網路，只有 `--allow-network` 明確允許模型下載。不會自動重試，缺少已驗證離線資產時不推理。離線瀏覽器必須提供 `--cache-dir`：暫時 loopback mirror 驗證所選固定模型檔案，寫入 SDK CacheStorage 與固定 WASM 資產，再將 Chromium 設為離線並封鎖外部請求。產物保存快取與網路證據；模型／快取完整性政策不變。

瀏覽器快取依 origin 區隔，重用時維持相同 `--browser-port` 與專用 `--browser-profile`；預設連接埠 `4173`。請勿使用個人瀏覽器 profile。未指定 profile 時使用短暫 context。其他基準選項包括 `--model-profile default|all-q4`、`--context-window-tokens N`、`--max-new-tokens N`、`--headed`，以及僅限 Node 的 `--profile-prefix PATH`。`--help` 列出全部選項。文字預設預算 512 tokens，邊界 512、圖片 256；文字提示明確要求五行各自對應的事實，不是只要求單一摘要。

上傳產物只應包含公開合成樣本。原始提示、模型文字與報告均不可信。不得上傳私人輸入、憑證、圖片或快取內容；詳見 [Security](../../SECURITY.md)。

## 明確版本與驗收政策

產物 `schemaVersion: 3`、樣本 `quality-fixtures-v2`、評估器 `quality-claims-v5`、政策 `quality-thresholds-v1` 都是明確契約。階層產物目前包含版本 3 報告；benchmark／gate 針對預設固定 0.8B 模型，不代表所有 registry 模型。閘門會核對目前 oracle SHA-256、樣本雜湊、生效案例輸入／預算與 stimulus key。版本缺少或錯誤、案例遺漏／重複、未嘗試／失敗及 profile 不符均 fail closed。整體與原始結果的模型 ID／revision 必須符合固定 manifest；原始結果與案例 backend 證據須一致，且符合宣告 runtime／device。舊產物須重新生成，改標版本不是遷移。

`scripts/quality/policy.mjs` 宣告門檻：

| 範圍 | 最低宣稱精確率 | 最低事實召回率 | 矛盾／無依據／格式錯誤上限 |
| --- | --- | --- | --- |
| 文字、圖片、邊界宣稱 | 1.0 | 1.0 | 0／0／0 |
| 階層保留來源引文 | 1.0 | 1.0 | 0／0／0 |
| 階層生成 section 紀錄宣稱 | 1.0 | 0.9 | 0／0／0 |
| 階層 overview／conclusion 紀錄提及 | 不要求 | 不要求 | 0／0／0 |

需要精確率／召回率的範圍，分母為零的 `null` 不算滿分，而是失敗。這些是驗收目標，**不是目前模型實測能力**。執行完成但召回不足、紅色說成粉紅色、數字矛盾、遵循注入文字或 section 遺漏事實，都可能失敗。不得為了讓目前模型通過而調低門檻。

## 評分邊界

文字／圖片使用 `FACT <ID>: <statement>`。缺少 `FACT` 另計格式錯誤，不直接改變事實評分。無法解析的非空行是格式錯誤；未知／重複 ID 或錯誤特徵是 false claim。true positive 必須符合所指定有限特徵，且沒有矛盾／unsupported 命中。精確率為 `TP/(TP+FP)`，召回率為 `TP/(TP+FN)`。其他已知受支援事實可作為上下文，但不是任意宣稱抽取。花園否定與圖片相對位置規則仍是限定詞彙的 heuristic。

邊界紀錄刻意更嚴格：每筆紀錄必須符合明列的原語言句子（NFC 正規化；溫度範圍允許 en dash 或 ASCII hyphen）。涵蓋否定事實、不同地塊分別綁定三／五種番茄、18–22 °C 兩端點、繁體中文／日文實體顏色數量、帶重音的 José，以及只能視為資料的惡意引文。添加子句、交換實體、錯誤數量或輸出注入密碼都會失敗。精確比對提供有意義的有限 Unicode 邊界，不代表多語理解或一般注入抵抗力。

階層的**保留與生成分開評估**。`sourceFacts[].citation.quote` 必須等於所引用來源段落的切片，且只有完整預期紀錄 tuple 才有內容涵蓋分數。生成 section `keyPoints` 依指定語法掃描，例如 `Observation H001: station S001 at Alder plot measured soil moisture 10 percent on 2025-01-01.`；`Observation` 可省略。紀錄 code、station、site、field、value、unit、date 必須在同一 tuple 全部吻合。所有 section 的紀錄聯集須滿足 0.9 召回與滿分精確率。

`page.summary` 與 `conclusion` 另查矛盾／unsupported tuple 與不完整紀錄提及，**不要求完整 300 筆召回**：壓縮 overview 不需重述全部目錄，也不能補足 section 遺漏事實。提到 H-ID 卻沒有完整 tuple 是格式錯誤；已知 tuple 內容錯誤是矛盾，未知 tuple 是 unsupported。重複正確紀錄另行記錄，不會增加召回率，也不因重述正確內容失敗。來源／引用 ID **永遠不提供語意特徵**。來源 ledger 全部保留但生成 section 為空或大量遺漏，仍會因生成召回率失敗。

非紀錄形式的階層文字不會獲得語意認證。語法以外的正確改寫可能失分，沒有紀錄 ID 的文字可能未被評分。此閘門仍是有限回歸 oracle，不證明任意報告文字有來源支持。版本 3 報告 audit 是保守詞彙證據檢查；tokenizer-aware JSON 約束、provenance／schema 驗證與精確引文保留，都不會替摘要查核事實，也不能取代獨立品質閘門。

## 比較與診斷

`stimulusKey` 表示目前編碼後樣本輸入與生效預算，不代表模型品質或解碼像素。產物包含模型／revision／profile、要求與觀察 backend、runtime／environment、樣本雜湊、原始輸出／錯誤及案例輸入。`status: completed`、`comparable: true` 只描述執行結果。閘門將 `completion` 與 `quality` 分開，並列出精確門檻失敗與命中診斷。

```sh
npm run quality:gate -- --input artifacts/new.json --baseline artifacts/prior.json \
  --comparison-mode strict --output artifacts/comparison.json
```

strict 比較要求輸入、版本、模型／revision、profile、runtime／device、觀察 backend、environment 及 browser 證據一致。`--comparison-mode paired-inputs` 明確允許 runtime／profile／backend 差異，但仍要求同模型／revision、樣本與編碼刺激，以及完整結構有效產物。任一品質閘門不通過，兩種比較模式都以非零狀態結束。程式化 API 位於 `scripts/quality/gate.mjs`：`evaluateQualityArtifact(artifact, { requiredCases? })`、`compareQualityArtifacts(left, right, { mode? })`。

paired-inputs **不是**獨立量化／平台實驗：Node 使用 Sharp 無 alpha 的 sRGB raw pixels，Browser 使用瀏覽器解碼與 RGBA Canvas pixels。manifest PNG bytes 相同，不保證解碼／前處理表示或 SDK 圖片 `versionId` 相同。不可將差異單獨歸因於量化。

## 目前實測結果

最新 macOS／arm64、Node 22.23.3 CPU／default 離線 run 使用 `quality-claims-v5`，嘗試全部四種樣本。文字（5 項事實）、圖片（3 項事實）、邊界（8 筆紀錄）的有限 oracle 精確率／召回率皆為 1.0，沒有命中矛盾或 unsupported，但全部遺漏指定的 `FACT` 前綴。300 段階層報告回傳 `STRUCTURED_OUTPUT`（`Generated text is not one complete JSON value`）。完整閘門以狀態 1 結束，回報執行不完整與品質失敗。契約測試、packed consumer、約束生成 smoke 與較小報告通過，不代表嚴格發行品質目標達成。

此次 artifact 保存階層錯誤，沒有失敗階段的原始 token stream／checkpoint，因此不能確認精確原因或階段。約束不保證在有限輸出預算內完成；不完整輸出仍 fail closed。沒有調低 gate 門檻。

**歷史 validation-only 診斷：**先前版本 2 run 定位到 `section:67` 的結尾引號錯誤跳脫，512-token 預算生成 149 tokens 即失敗。該舊 checkpoint 保留全部 300 段精確引用及 67 個已完成階段，記錄包含失敗嘗試的 input 33,490／output 11,485 tokens；持久化驗證通過。這是歷史失敗／來源保留證據，不是目前約束 run 的診斷，也不代表 300 段推理完成或事實正確性。

## CI

每次 push／PR 都在現有 verification job 執行確定性 `test:quality`。模型較重的 `real-model-quality` 只在 `workflow_dispatch` 執行，使用既有 macOS 15／Node 22 CPU runtime、明確網路許可及暫存模型快取。同時執行 benchmark **與 gate**，不良模型結果會讓 job 失敗。即使品質失敗，仍透過 `if: always()` 上傳原始輸出及結構化閘門診斷。沒有發布動作，也沒有新增平台認證宣稱。
