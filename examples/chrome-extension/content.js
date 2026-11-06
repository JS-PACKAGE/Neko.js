// Floating "AI 解析頁面" button on every http(s) page. The page snapshot is sent to the extension's offscreen
// document, which runs Neko.js; results are shown as plain text in a closed shadow root the page cannot read.
(() => {
  if (window.top !== window) return;

  const host = document.createElement('div');
  host.dataset.nekoReader = '';
  const shadow = host.attachShadow({ mode: 'closed' });
  shadow.innerHTML = `
<style>
  :host { all: initial; }
  * { box-sizing: border-box; font: 15px/1.6 system-ui, sans-serif; }
  .fab { position: fixed; right: 24px; bottom: 24px; z-index: 2147483646; display: flex; align-items: center; gap: 8px;
    padding: 12px 18px; border: 0; border-radius: 999px; background: #5b4bc4; color: #fff; font-weight: 600; cursor: pointer;
    box-shadow: 0 6px 20px rgb(0 0 0 / .25); transition: transform .15s; }
  .fab:hover { transform: translateY(-2px); }
  .fab svg { width: 22px; height: 22px; }
  .panel { position: fixed; top: 0; right: 0; bottom: 0; z-index: 2147483647; width: min(440px, 100vw); display: flex; flex-direction: column;
    background: #fff; color: #222; box-shadow: -8px 0 30px rgb(0 0 0 / .2); transform: translateX(100%); transition: transform .2s; }
  .panel.open { transform: none; }
  header { display: flex; align-items: center; gap: 8px; padding: 14px 16px; border-bottom: 1px solid #e5e5e5; }
  header h2 { flex: 1; margin: 0; font-size: 17px; font-weight: 700; }
  button.small { padding: 4px 10px; border: 1px solid #ccc; border-radius: 6px; background: #f6f6f6; color: inherit; cursor: pointer; }
  button.small:disabled { opacity: .5; cursor: default; }
  .status { padding: 10px 16px 0; color: #555; font-size: 13px; min-height: 1.6em; }
  .status.error { color: #c62828; }
  progress { width: calc(100% - 32px); margin: 6px 16px 0; height: 6px; }
  .body { flex: 1; overflow-y: auto; padding: 4px 16px 16px; }
  h3 { margin: 18px 0 6px; font-size: 15px; font-weight: 700; color: #5b4bc4; }
  h4 { margin: 12px 0 4px; font-size: 14px; font-weight: 600; }
  p, li { margin: 4px 0; font-size: 14px; }
  ul { margin: 0; padding-left: 20px; }
  .note { margin-top: 18px; padding: 8px 10px; border-radius: 6px; background: #fff6e0; color: #6b5300; font-size: 12px; }
</style>
<button class="fab" type="button" aria-label="使用 AI 解析這個頁面">
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M4 3l4 4.2A9 9 0 0 1 12 6.5a9 9 0 0 1 4 .7L20 3v8.5A8 8 0 0 1 12 21a8 8 0 0 1-8-9.5zM9 12.5a1.2 1.2 0 1 0 0 .01zm6 0a1.2 1.2 0 1 0 0 .01z"/></svg>
  AI 解析頁面
</button>
<aside class="panel" aria-label="Neko.js 頁面解析" aria-hidden="true">
  <header><h2>Neko.js 頁面解析</h2><button class="small" data-action="stop" type="button" disabled>停止</button><button class="small" data-action="close" type="button">關閉</button></header>
  <div class="status" role="status" aria-live="polite"></div>
  <progress hidden></progress>
  <div class="body"></div>
</aside>`;
  document.documentElement.append(host);

  const $ = (selector) => shadow.querySelector(selector);
  const panel = $('.panel'); const status = $('.status'); const progress = $('progress'); const body = $('.body'); const stopButton = $('[data-action="stop"]');
  const element = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; };
  const setStatus = (text, error = false) => { status.textContent = text; status.classList.toggle('error', error); };
  let port;

  /**
   * Drops page chrome before the model sees it: every report stage costs seconds, and navigation menus, "related" and "most read"
   * link lists can outnumber an article's paragraphs several times over. Removes landmarks outside the article and, innermost first,
   * blocks whose text is mostly link text. Falls back to the unpruned document when too little text would remain.
   */
  function pruneChrome(doc) {
    const length = (node) => node.textContent.replace(/\s+/g, '').length;
    const before = length(doc.body);
    const pruned = doc.body.cloneNode(true);
    for (const node of pruned.querySelectorAll('nav, aside, [role="navigation"], [role="complementary"], [role="banner"], [role="contentinfo"], header, footer')) {
      if ((node.localName === 'header' || node.localName === 'footer') && node.closest('article, main')) continue;
      node.remove();
    }
    for (const node of [...pruned.querySelectorAll('li, ul, ol, div, section, table, p')].reverse()) {
      const total = length(node);
      if (!total) continue;
      let linked = 0;
      for (const link of node.querySelectorAll('a')) linked += length(link);
      if (linked / total > 0.6) node.remove();
    }
    const after = length(pruned);
    if (after >= 200 || after >= before / 2) doc.body.replaceWith(pruned);
  }

  /** Rendered DOM without this button, page chrome, scripts, styles and other non-content markup. */
  function snapshot() {
    // Re-parse the serialized DOM into an inert document: cloning live nodes would make images load again.
    const inert = new DOMParser().parseFromString(document.documentElement.outerHTML, 'text/html');
    for (const node of inert.querySelectorAll('[data-neko-reader], script, style, noscript, template, iframe, link')) node.remove();
    pruneChrome(inert);
    return `<!doctype html>${inert.documentElement.outerHTML}`;
  }

  function render(message) {
    const { report } = message;
    body.replaceChildren(element('h3', '頁面摘要'), element('p', report.page.summary));
    if (report.sections.length) body.append(element('h3', '分段重點'));
    for (const section of report.sections) {
      if (section.heading) body.append(element('h4', section.heading));
      const list = element('ul');
      for (const point of section.keyPoints) list.append(element('li', point));
      body.append(list);
    }
    const described = report.images.filter((image) => image.status === 'described');
    if (described.length) body.append(element('h3', '圖片內容'));
    for (const image of described) body.append(element('p', `${image.alt ? `「${image.alt}」` : '圖片'}：${image.description}`));
    const failed = report.images.filter((image) => image.status === 'failed');
    if (failed.length) body.append(element('p', `有 ${failed.length} 張圖片無法讀取（${failed.map((image) => image.error.message).join('；')}）。`, 'note'));
    body.append(element('h3', '結論'), element('p', report.conclusion));
    const { inputTokens, outputTokens } = report.metadata.usage;
    body.append(element('p', `AI 生成內容可能有誤，請以原文為準。耗時 ${message.seconds.toFixed(1)} 秒，${inputTokens} 輸入／${outputTokens} 輸出 token。`, 'note'));
    setStatus(`完成：${message.paragraphs} 段文字、${message.images} 張圖片${message.images > message.describedImages ? `（描述前 ${message.describedImages} 張）` : ''}。`);
  }

  function finish() { port = undefined; stopButton.disabled = true; progress.hidden = true; }

  function connect(request, attempt = 0) {
    let answered = false;
    port = chrome.runtime.connect({ name: 'neko:read' });
    port.onMessage.addListener((message) => {
      answered = true;
      if (message.type === 'status') { progress.hidden = true; setStatus(message.text); }
      else if (message.type === 'progress') {
        progress.hidden = false; progress.max = message.total || 1; progress.value = message.loaded;
        setStatus(`${message.phase === 'verify' ? '驗證' : '下載'}模型檔案：${(message.loaded / 2 ** 20).toFixed(0)} / ${(message.total / 2 ** 20).toFixed(0)} MiB`);
      } else if (message.type === 'result') { render(message); port.disconnect(); finish(); }
      else if (message.type === 'error') { setStatus(`解析失敗：${message.message}`, true); port.disconnect(); finish(); }
    });
    port.onDisconnect.addListener(() => {
      // A just-created offscreen document may not be listening yet; retry a few times before giving up.
      if (!answered && attempt < 5 && port) { setTimeout(() => connect(request, attempt + 1), 300); return; }
      if (port) { setStatus(answered ? '連線中斷。' : `無法連線到擴充功能：${chrome.runtime.lastError?.message ?? '未知錯誤'}`, true); finish(); }
    });
    port.postMessage(request);
  }

  async function read() {
    panel.classList.add('open'); panel.setAttribute('aria-hidden', 'false');
    if (port) return;
    body.replaceChildren(); setStatus('準備中…'); stopButton.disabled = false;
    try {
      const ready = await chrome.runtime.sendMessage({ type: 'neko:ensure-offscreen' });
      if (!ready?.ok) throw new Error(ready?.message ?? 'offscreen document unavailable');
      const { modelBase } = await chrome.storage.local.get({ modelBase: '' });
      connect({ type: 'read', html: snapshot(), url: location.href, language: document.documentElement.lang || navigator.language, modelBase });
    } catch (error) {
      setStatus(`解析失敗：${error instanceof Error ? error.message : String(error)}`, true); finish();
    }
  }

  $('.fab').addEventListener('click', read);
  stopButton.addEventListener('click', () => { const open = port; finish(); open?.disconnect(); setStatus('已停止。'); });
  $('[data-action="close"]').addEventListener('click', () => { panel.classList.remove('open'); panel.setAttribute('aria-hidden', 'true'); });
})();
