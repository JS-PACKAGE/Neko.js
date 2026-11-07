// Hosts the single Neko.js instance for the whole browser. Content scripts connect a port named "neko:read",
// send one request, and receive status/progress messages followed by a result or an error.
import { createNeko, extractPage } from './neko/neko.js';

const IDLE_UNLOAD_MS = 10 * 60_000;
const MAX_COLLECTED_IMAGES = 500;
const MAX_DESCRIBED_IMAGES = 6;
// Pages without <main>/<article> fall back to the whole body; navigation labels are short, article text is not.
const MIN_PARAGRAPH_CHARACTERS = 20;
// Each section stage handles four sentence-level quotes and takes seconds; a long page (a Wikipedia article has hundreds of
// paragraphs) would take many minutes, so only the leading text is read.
const MAX_CHARACTERS = 3000;
const bundleDirectory = new URL('./neko/', import.meta.url).href;
// The pinned model files ship inside the extension package; the SDK reads and SHA-256-verifies them in place on each load.
const modelDirectory = new URL('./model/', import.meta.url);
let current; // Promise of the loaded instance
let released = Promise.resolve(); // Disposal of an idle instance; only one Neko runtime may be active at a time.
let active = 0;
let idleTimer;

/** Literal loopback, private, link-local and .local hosts; DNS names that resolve to them are not detected. */
function isPrivateHost(hostname) {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal')) return true;
  const v4 = host.replace(/^::ffff:/, '').match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])];
    return a === 0 || a === 10 || a === 127 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127);
  }
  return host === '::' || host === '::1' || /^f[cd]/.test(host) || /^fe[89ab]/.test(host);
}

// Simplified-only forms; a few in Traditional Chinese text mean it was really Simplified Chinese.
const SIMPLIFIED = /[这为与个从们来时会说对国学实应现过还进种无点开关颜红]/gu;
/**
 * The report is written in the language of the text being read. The 0.8B model does not reliably translate, so asking
 * for English about Chinese text (common: page templates declare lang="en") fails the SDK's language check on every retry.
 * The declared lang only picks between variants of the detected script.
 */
function reportLanguage(text, declared) {
  const count = (pattern) => text.match(pattern)?.length ?? 0;
  const han = count(/\p{Script=Han}/gu), latin = count(/\p{Script=Latin}/gu);
  const kana = count(/[\p{Script=Hiragana}\p{Script=Katakana}]/gu), hangul = count(/\p{Script=Hangul}/gu);
  let hint = 'und';
  try { hint = Intl.getCanonicalLocales(declared || 'und')[0]; } catch { /* invalid lang attribute: ignore it */ }
  const primary = hint.split('-')[0].toLowerCase();
  if (kana && kana * 4 >= han) return 'ja';
  if (hangul > latin && hangul > han) return 'ko';
  if (han && han * 2 >= latin) {
    if (primary === 'zh' && /-(?:hans|cn|sg|my)\b/i.test(hint)) return 'zh-CN';
    if (primary === 'zh' && /-(?:hant|tw|hk|mo)\b/i.test(hint)) return 'zh-TW';
    return count(SIMPLIFIED) * 50 > han ? 'zh-CN' : 'zh-TW';
  }
  if (latin) return primary && !['zh', 'ja', 'ko', 'und'].includes(primary) ? hint : 'en';
  return hint === 'und' ? 'en' : hint;
}

function instance(report) {
  if (current) return current;
  let loading = true;
  const files = new Map();
  const neko = (async () => {
    await released;
    const created = await createNeko({
      device: 'webgpu',
      modelSource: { baseUrl: modelDirectory.href },
      // A custom policy replaces the default one. Images are checked per request by validateDestination below.
      policy: { network(url, kind) {
        if (kind === 'runtime' || kind === 'worker') return url.href.startsWith(bundleDirectory);
        return kind === 'image';
      } },
      onCacheProgress(event) {
        if (!loading) return;
        files.set(`${event.phase}:${event.file}`, event);
        let loaded = 0; let total = 0;
        for (const file of files.values()) if (file.phase === event.phase) { loaded += file.loaded; total += file.total; }
        report({ type: 'progress', phase: event.phase, loaded, total });
      },
    });
    try { await created.load(); } catch (error) { await created.dispose().catch(() => undefined); throw error; }
    loading = false;
    return created;
  })();
  current = neko;
  // A failed load is not cached; the next request retries.
  neko.catch(() => { if (current === neko) current = undefined; });
  return neko;
}

function scheduleUnload() {
  clearTimeout(idleTimer);
  if (active) return;
  // Release the model (about 1 GB of memory) when unused; the next request reloads it from the extension package.
  idleTimer = setTimeout(() => {
    const idle = current; current = undefined;
    if (idle) released = idle.then((neko) => neko.dispose()).catch(() => undefined);
  }, IDLE_UNLOAD_MS);
}

async function read(request, port, signal) {
  const send = (message) => { if (!signal.aborted) port.postMessage(message); };
  const pageUrl = new URL(request.url);
  if (pageUrl.protocol !== 'http:' && pageUrl.protocol !== 'https:') throw new Error('Only http and https pages can be read.');
  send({ type: 'status', text: '載入模型中…' });
  const neko = await instance(send);
  signal.throwIfAborted();
  send({ type: 'status', text: '擷取頁面內容…' });
  // extractPage rejects pages with more images than maxImages, so collect generously and describe only the first few.
  const page = await extractPage(request.html, { baseUrl: pageUrl.href, content: 'main', maxHtmlBytes: 4 * 1024 * 1024, maxImages: MAX_COLLECTED_IMAGES });
  // Prefer images the page rendered large; without that information (or a match), skip SVG, which cannot be described anyway.
  const large = new Set(Array.isArray(request.images) ? request.images.filter((url) => typeof url === 'string') : []);
  const rendered = page.images.filter((image) => large.has(image.url));
  const candidates = rendered.length ? rendered : page.images.filter((image) => !/\.svg(?:[?#]|$)/i.test(image.url) && !/^data:image\/svg/i.test(image.url));
  const describedImageIds = candidates.slice(0, MAX_DESCRIBED_IMAGES).map((image) => image.id);
  const long = page.paragraphs.filter((paragraph) => paragraph.text.length >= MIN_PARAGRAPH_CHARACTERS);
  const readable = long.length >= 3 ? long : page.paragraphs;
  const paragraphIds = [];
  let characters = 0;
  for (const paragraph of readable) {
    if (paragraphIds.length && characters + paragraph.text.length > MAX_CHARACTERS) break;
    paragraphIds.push(paragraph.id); characters += paragraph.text.length;
  }
  if (!page.paragraphs.length) throw new Error('這個頁面沒有可解析的文字。');
  const selected = new Set(paragraphIds);
  const language = reportLanguage(page.paragraphs.filter((paragraph) => selected.has(paragraph.id)).map((paragraph) => paragraph.text).join('\n'), request.language);
  const stages = { image: [], section: [], summary: [], conclusion: [] };
  let latest;
  const names = { image: '描述圖片', section: '整理段落', summary: '撰寫摘要', conclusion: '撰寫結論' };
  const begun = performance.now();
  const report = await neko.describe(page, {
    // A blocked or broken image is reported as failed instead of failing the whole report. A stage whose output is
    // invalid (e.g. JSON cut off at the token limit) or in the wrong language is retried with a corrective instruction.
    format: 'json', language, maxNewTokens: 512, imageFailurePolicy: 'omit', sources: { paragraphIds, imageIds: describedImageIds },
    retries: { image: 2, section: 2, summary: 2, conclusion: 2 }, signal,
    // The extension can fetch any host, so a page must not be able to make it read internal addresses.
    validateDestination(url) {
      if (url.origin !== pageUrl.origin && isPrivateHost(url.hostname)) throw new Error(`Blocked private address: ${url.host}`);
    },
    // Completed stages are validated claims; showing them as they finish turns a silent minute into steady output.
    onCheckpoint(checkpoint) { latest = checkpoint; },
    onEvent(event) {
      if (event.type === 'stage-complete' && (event.phase === 'section' || event.phase === 'image')) {
        const claims = latest?.completed[event.id]?.claims.map((claim) => claim.text) ?? [];
        if (claims.length) send({ type: 'partial', phase: event.phase, claims });
      }
      if (event.type !== 'stage-start') return;
      const seen = stages[event.phase]; const retry = seen.includes(event.id);
      if (!retry) seen.push(event.id);
      send({ type: 'status', text: `${names[event.phase]}（第 ${seen.indexOf(event.id) + 1} 項${retry ? '，重試' : ''}）…` });
    },
  });
  send({ type: 'result', report, seconds: (performance.now() - begun) / 1000, paragraphs: paragraphIds.length, totalParagraphs: readable.length, images: page.images.length, describedImages: describedImageIds.length });
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'neko:read' || port.sender?.id !== chrome.runtime.id) return;
  const controller = new AbortController();
  port.onDisconnect.addListener(() => controller.abort());
  port.onMessage.addListener(async (request) => {
    if (request?.type !== 'read') return;
    active++; clearTimeout(idleTimer);
    try {
      await read(request, port, controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) port.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    } finally {
      active--; scheduleUnload();
    }
  });
});
