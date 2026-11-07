# Neko.js 頁面解析 — Privacy policy / 隱私權政策

## English

This extension reads a web page only when you click its "AI 解析頁面" button on that page.

- **What is processed:** a copy of the page's HTML with scripts, styles and frames removed, the page URL, the page language, and images shown on the page.
- **Where:** entirely in your browser. The AI model is included in the extension and runs on your device with WebGPU. Page content, images and the generated report are not sent to the developer or to any other server.
- **Network requests:** to describe images, the extension downloads the page's own images from the addresses the page uses for them. Image addresses on another site that point at local or private network hosts are refused. No other requests are made; the model is not downloaded.
- **Storage:** nothing is stored. The report exists only in the open panel; the loaded model is released from memory after 10 idle minutes.
- **No tracking:** no analytics, advertising, accounts, cookies or sale or transfer of data.

Generated reports can be wrong. Do not rely on them for decisions that matter.

## 繁體中文

只有在你於某個頁面按下「AI 解析頁面」按鈕時，此擴充功能才會讀取該頁面。

- **處理的內容：** 移除腳本、樣式與 frame 後的頁面 HTML 複本、頁面網址、頁面語言，以及頁面上顯示的圖片。
- **處理位置：** 完全在你的瀏覽器內。AI 模型隨擴充功能提供，以 WebGPU 在你的裝置上執行。頁面內容、圖片與產生的報告不會傳送給開發者或任何其他伺服器。
- **網路請求：** 為了描述圖片，擴充功能會從頁面本身使用的位址下載頁面圖片。指向其他網站、且為本機或私有網路主機的圖片位址會被拒絕。除此之外不發出其他請求，也不下載模型。
- **儲存：** 不儲存任何資料。報告只存在於開啟中的面板；載入的模型閒置 10 分鐘後會從記憶體釋放。
- **不追蹤：** 沒有分析、廣告、帳號或 cookie，也不販售或轉移資料。

產生的報告可能有誤，請勿用於重要決策。
