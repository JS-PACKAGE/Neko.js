# Repository instructions

Follow [AGENTS.md](AGENTS.md) as the authoritative contribution guide. It covers scope, implementation standards, testing, and security requirements; do not duplicate or weaken its policies here.

## Security reminders

Treat page content and model output as untrusted; language and provenance checks do not prove truth. Never treat provider metadata as proof of device execution, and never claim inference success without a real image/prompt run. Remote inputs to `Neko.infer`/`Neko.describe` and web/image helpers can expose server-side request forgery risks: use `validateDestination` for each HTTP(S) destination/redirect and enforce outbound network controls. Restrict untrusted Node filesystem image paths. Preserve pinned model SHA-256/size validation, await SDK disposal before replacing its runtime owner, and use private vulnerability reporting as described in [SECURITY.md](SECURITY.md).

## 繁體中文重點

請以 [AGENTS.md](AGENTS.md) 為準；網頁／模型輸出皆是不可信資料，語言與來源檢查不保證正確性。不得把 provider 設定當成 GPU 執行證據，亦不得以載入成功代替真實圖片推理。`Neko.infer`／`Neko.describe` 的遠端輸入須由應用層驗證每個目的地及重新導向、防範 SSRF，並限制本機圖片路徑。保留固定模型完整性檢查，先等待 `dispose()` 完成再建立新執行環境。
