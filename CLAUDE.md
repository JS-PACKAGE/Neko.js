# Repository instructions

Follow [AGENTS.md](AGENTS.md) as the authoritative contribution guide. It covers scope, implementation standards, testing, and security requirements; do not duplicate or weaken its policies here.

## Security reminders

Treat page content and model output as untrusted. Never treat provider metadata as proof of device execution, and never claim inference success without a real image/prompt run. Node URL extraction can expose server-side request forgery risks; enforce destination allowlists and outbound network controls at application boundaries. Preserve pinned model SHA-256/size validation and use private vulnerability reporting as described in [SECURITY.md](SECURITY.md).

## 繁體中文重點

請以 [AGENTS.md](AGENTS.md) 為準；網頁／模型輸出皆是不可信資料。不得把 provider 設定當成 GPU 執行證據，亦不得以載入成功代替真實圖片推理。Node 網址抓取須由應用層防範 SSRF，並保留固定模型完整性檢查。
