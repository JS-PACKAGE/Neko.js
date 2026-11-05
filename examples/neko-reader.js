// Floating "read this page" button: Neko.js reads the current page in the browser with WebGPU
// and shows a structured report (summary, section key points, image descriptions, conclusion).
//
//   <script type="module" src="/examples/neko-reader.js"
//           data-bundle="/dist/browser/neko.js" data-model-base="/models/<id>/<revision>/"></script>
//
// All attributes are optional. The model base defaults to the pinned same-origin /models/ mirror path,
// which `node scripts/serve-examples.mjs --cache-dir=...` provides. Nothing is loaded until the button is pressed.
const script = [...document.scripts].find((element) => element.src === import.meta.url);
const settings = {
  bundle: new URL(script?.dataset.bundle ?? '../dist/browser/neko.js', import.meta.url).href,
  modelBase: script?.dataset.modelBase,
  language: script?.dataset.language || document.documentElement.lang || navigator.language,
  maxNewTokens: Number(script?.dataset.maxNewTokens ?? 512),
};

const host = document.createElement('div');
host.dataset.nekoReader = '';
const shadow = host.attachShadow({ mode: 'open' });
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
  figure { margin: 10px 0; display: grid; grid-template-columns: 72px 1fr; gap: 10px; align-items: start; }
  figure img { width: 72px; height: 72px; object-fit: cover; border-radius: 6px; border: 1px solid #ddd; }
  figcaption { font-size: 13px; }
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
document.body.append(host);

const $ = (selector) => shadow.querySelector(selector);
const panel = $('.panel'); const status = $('.status'); const progress = $('progress'); const body = $('.body'); const stopButton = $('[data-action="stop"]');
const element = (tag, text, attributes = {}) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; Object.assign(node, attributes); return node; };
const setStatus = (text, error = false) => { status.textContent = text; status.classList.toggle('error', error); };
const phaseNames = { image: '描述圖片', section: '整理段落', summary: '撰寫摘要', conclusion: '撰寫結論' };

let neko; let running; let controller;

async function instance(signal) {
  if (neko) return neko;
  const { createNeko, MODEL_ID, MODEL_REVISION } = await import(settings.bundle);
  const modelBase = new URL(settings.modelBase ?? `/models/${MODEL_ID}/${MODEL_REVISION}/`, location.href);
  const bundleDirectory = new URL('./', settings.bundle).href;
  const files = new Map();
  let loading = true;
  const created = await createNeko({
    device: 'webgpu',
    modelSource: { baseUrl: modelBase.href },
    // A custom policy replaces the default one: allow the model mirror, the runtime files and worker shipped with
    // the bundle, and only this page's own images.
    policy: { network(url, kind) {
      if (kind === 'model') return url.origin === modelBase.origin;
      if (kind === 'runtime' || kind === 'worker') return url.href.startsWith(bundleDirectory);
      return url.origin === location.origin;
    } },
    // Files are reported once per phase (download, then verify), and the first image request verifies the deferred
    // vision files again; show progress per phase and only while load() runs.
    onCacheProgress(event) {
      if (!loading) return;
      files.set(`${event.phase}:${event.file}`, event);
      let loaded = 0; let total = 0;
      for (const file of files.values()) if (file.phase === event.phase) { loaded += file.loaded; total += file.total; }
      progress.hidden = false; progress.max = total || 1; progress.value = loaded;
      setStatus(`${event.phase === 'verify' ? '驗證' : '下載'}模型檔案：${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MiB`);
    },
  });
  try {
    setStatus('載入模型中…');
    await created.load(signal);
  } catch (error) {
    await created.dispose().catch(() => undefined);
    throw error;
  } finally {
    loading = false;
  }
  neko = created;
  return neko;
}

/** Snapshot of the rendered page without this widget; the SDK parses it inertly (no scripts run). */
function pageHtml() {
  // Re-parse the serialized DOM into an inert document: cloning live nodes would make images load again.
  const inert = new DOMParser().parseFromString(document.documentElement.outerHTML, 'text/html');
  for (const node of inert.querySelectorAll('[data-neko-reader]')) node.remove();
  return `<!doctype html>${inert.documentElement.outerHTML}`;
}

function render(report, seconds) {
  body.replaceChildren();
  body.append(element('h3', '頁面摘要'), element('p', report.page.summary));
  if (report.sections.length) {
    body.append(element('h3', '分段重點'));
    for (const section of report.sections) {
      if (section.heading) body.append(element('h4', section.heading));
      const list = element('ul');
      for (const point of section.keyPoints) list.append(element('li', point));
      body.append(list);
    }
  }
  const described = report.images.filter((image) => image.status === 'described');
  if (described.length) {
    body.append(element('h3', '圖片內容'));
    for (const image of described) {
      const figure = element('figure');
      figure.append(element('img', undefined, { src: image.url, alt: image.alt ?? '' }), element('figcaption', image.description));
      body.append(figure);
    }
  }
  body.append(element('h3', '結論'), element('p', report.conclusion));
  const { inputTokens, outputTokens } = report.metadata.usage;
  body.append(element('p', `AI 生成內容可能有誤，請以原文為準。耗時 ${seconds.toFixed(1)} 秒，${inputTokens} 輸入／${outputTokens} 輸出 token。`, { className: 'note' }));
}

async function read() {
  if (running) return;
  panel.classList.add('open'); panel.setAttribute('aria-hidden', 'false');
  if (!navigator.gpu) { setStatus('這個瀏覽器不支援 WebGPU，無法執行 Neko.js 的模型。', true); return; }
  running = true; controller = new AbortController(); stopButton.disabled = false;
  body.replaceChildren(); setStatus('準備中…');
  const startedAt = performance.now();
  try {
    const reader = await instance(controller.signal);
    const { extractPage } = await import(settings.bundle);
    progress.hidden = true;
    setStatus('擷取頁面內容…');
    const page = await extractPage(pageHtml(), { baseUrl: location.href, content: 'main' });
    const stages = { image: [], section: [], summary: [], conclusion: [] };
    const report = await reader.describe(page, {
      // A stage whose output is invalid (e.g. JSON cut off at the token limit) is retried once with a corrective instruction.
      format: 'json', language: settings.language, maxNewTokens: settings.maxNewTokens, retries: { image: 1, section: 1, summary: 1, conclusion: 1 }, signal: controller.signal,
      onEvent(event) {
        if (event.type !== 'stage-start') return;
        const seen = stages[event.phase]; const retry = seen.includes(event.id);
        if (!retry) seen.push(event.id);
        setStatus(`${phaseNames[event.phase]}（第 ${seen.indexOf(event.id) + 1} 項${retry ? '，重試' : ''}）…`);
      },
    });
    render(report, (performance.now() - startedAt) / 1000);
    setStatus(`完成：${page.paragraphs.length} 段文字、${page.images.length} 張圖片。`);
  } catch (error) {
    setStatus(controller.signal.aborted ? '已停止。' : `解析失敗：${error instanceof Error ? error.message : String(error)}`, !controller.signal.aborted);
  } finally {
    running = false; stopButton.disabled = true; progress.hidden = true;
  }
}

$('.fab').addEventListener('click', read);
stopButton.addEventListener('click', () => controller?.abort());
$('[data-action="close"]').addEventListener('click', () => { panel.classList.remove('open'); panel.setAttribute('aria-hidden', 'true'); });
window.addEventListener('pagehide', () => { void neko?.dispose(); }, { once: true });
