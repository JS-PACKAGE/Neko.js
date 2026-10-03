import { parse } from 'parse5';
import type { ImageDiscovery, Page, PageImage, Paragraph, PageContainer, PageTable, TableCell } from '../types.js';
import { policyDestination } from './policy.js';
import type { ResourcePolicy } from './policy.js';
import { awaitUser } from '../errors.js';

export interface ExtractOptions {
  baseUrl?: string;
  includeImages?: boolean;
  /** Prefer a unique visible main/article; ambiguous or empty candidates fall back to full body extraction. */
  content?: 'full' | 'main';
  signal?: AbortSignal;
  maxHtmlBytes?: number;
  maxImages?: number;
  maxImageBytes?: number;
  timeoutMs?: number;
  /** Called before each HTTP(S) request, including redirects; applications must enforce their own network policy. */
  validateDestination?: (url: URL) => void | Promise<void>;
}
export interface ExtractContext { policy?: ResourcePolicy | undefined; offline?: boolean | undefined; }

interface HtmlNode {
  nodeName: string;
  tagName?: string;
  value?: string;
  attrs?: Array<{ name: string; value: string }>;
  childNodes?: HtmlNode[];
  parentNode?: HtmlNode;
  sourceCodeLocation?: { startOffset: number; endOffset: number };
}

const blockedTextTags: Record<string, true> = { script: true, style: true, noscript: true, template: true, svg: true, canvas: true, head: true };
const textBlockTags: Record<string, true> = {
  h1: true, h2: true, h3: true, h4: true, h5: true, h6: true, p: true, li: true, blockquote: true,
  figcaption: true, caption: true, td: true, th: true, dt: true, dd: true,
};
const flowContainerTags: Record<string, true> = { div: true, section: true, article: true, main: true, header: true, footer: true, aside: true, nav: true };
const defaultHtmlLimit = 2 * 1024 * 1024;
const defaultImageCount = 20;
const defaultTimeout = 10_000;

function attr(node: HtmlNode, name: string): string | undefined {
  return node.attrs?.find((item) => item.name === name)?.value;
}

function textContent(node: HtmlNode): string {
  if (node.nodeName === '#text') return node.value ?? '';
  if (node.tagName && blockedTextTags[node.tagName]) return '';
  if (attr(node, 'hidden') !== undefined || attr(node, 'aria-hidden')?.toLowerCase() === 'true') return '';
  return (node.childNodes ?? []).map(textContent).join('');
}

function normalizeText(value: string): string {
  return value.replace(/[\s\u00a0]+/g, ' ').trim();
}

function parseSrcset(value: string): string[] {
  const urls: string[] = [];
  let cursor = 0;
  while (cursor < value.length) {
    while (cursor < value.length && /[\s,]/.test(value[cursor]!)) cursor++;
    if (cursor >= value.length) break;
    const start = cursor;
    const isData = value.slice(cursor, cursor + 5).toLowerCase() === 'data:';
    if (isData) {
      while (cursor < value.length && !/\s/.test(value[cursor]!)) cursor++;
    } else {
      while (cursor < value.length && !/[\s,]/.test(value[cursor]!)) cursor++;
    }
    const url = value.slice(start, cursor).replace(/,+$/, '');
    if (url) urls.push(url);
    while (cursor < value.length && value[cursor] !== ',') cursor++;
    if (value[cursor] === ',') cursor++;
  }
  return urls;
}

function cssImageUrls(value: string): string[] {
  const urls: string[] = [];
  const declarations = /(?:^|[;{])\s*(?:background(?:-image)?|border-image(?:-source)?)\s*:\s*([^;}]+)/gi;
  for (const declaration of value.matchAll(declarations)) {
    const pattern = /url\(\s*(?:"([^"]*)"|'([^']*)'|([^)]*?))\s*\)/gi;
    for (const match of declaration[1]!.matchAll(pattern)) {
      const url = (match[1] ?? match[2] ?? match[3] ?? '').trim();
      if (url) urls.push(url);
    }
  }
  return urls;
}

function resolveImage(raw: string, pageUrl: string): string | undefined {
  const candidate = raw.trim();
  if (!candidate || candidate.startsWith('#')) return undefined;
  if (/^data:/i.test(candidate)) return candidate;
  try {
    const url = pageUrl ? new URL(candidate, pageUrl) : new URL(candidate);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new TypeError(`Unsupported image URL protocol: ${url.protocol}`);
    return url.href;
  } catch (error) {
    if (error instanceof TypeError && error.message.startsWith('Unsupported image URL protocol:')) throw error;
    if (!pageUrl && !/^[a-z][a-z\d+.-]*:/i.test(candidate)) return candidate;
    throw new TypeError(`Could not resolve image URL: ${candidate}`, { cause: error });
  }
}

function isInsidePicture(node: HtmlNode): boolean {
  for (let parent = node.parentNode; parent; parent = parent.parentNode) {
    if (parent.tagName === 'picture') return true;
  }
  return false;
}

function figureCaption(node: HtmlNode): string | undefined {
  for (let parent = node.parentNode; parent; parent = parent.parentNode) {
    if (parent.tagName === 'figure') {
      const caption = (parent.childNodes ?? []).find((child) => child.tagName === 'figcaption');
      const value = caption ? normalizeText(textContent(caption)) : '';
      return value || undefined;
    }
  }
  return undefined;
}

function location(node: HtmlNode): Paragraph['source'] {
  return node.sourceCodeLocation
    ? { kind: 'html', startOffset: node.sourceCodeLocation.startOffset, endOffset: node.sourceCodeLocation.endOffset }
    : { kind: 'html' };
}

function visible(node: HtmlNode): boolean {
  return !(node.tagName && blockedTextTags[node.tagName]) && attr(node, 'hidden') === undefined && attr(node, 'aria-hidden')?.toLowerCase() !== 'true';
}

function contentRoot(document: HtmlNode, mode: 'full' | 'main'): { node: HtmlNode; extraction: NonNullable<Page['extraction']> } {
  const body = findFirst(document, 'body') ?? document;
  if (mode === 'main') {
    for (const tag of ['main', 'article'] as const) {
      const candidates: HtmlNode[] = [];
      const visit = (node: HtmlNode): void => {
        if (!visible(node)) return;
        if (node.tagName === tag && normalizeText(textContent(node))) { candidates.push(node); return; }
        for (const child of node.childNodes ?? []) visit(child);
      };
      visit(body);
      if (candidates.length === 1) return { node: candidates[0]!, extraction: { mode, root: tag, fallback: false } };
      if (candidates.length > 1) break;
    }
  }
  return { node: body, extraction: { mode, root: 'body', fallback: mode === 'main' } };
}

function extractStructure(root: HtmlNode): Pick<Page, 'paragraphs' | 'containers' | 'tables'> {
  const paragraphs: Paragraph[] = [];
  const containers: PageContainer[] = [];
  const containerNodes = new Map<HtmlNode, PageContainer>();
  const paragraphNodes = new Map<HtmlNode, string[]>();
  const tables: HtmlNode[] = [];
  const prepare = (node: HtmlNode, parent?: PageContainer): void => {
    if (!visible(node)) return;
    const tag = node.tagName;
    let container = parent;
    if (tag && (flowContainerTags[tag] || ['body', 'figure', 'table', 'td', 'th', 'ul', 'ol'].includes(tag))) {
      container = { id: `c${containers.length + 1}`, kind: tag, ...(parent ? { parentId: parent.id } : {}), paragraphIds: [], source: location(node) };
      containers.push(container); containerNodes.set(node, container);
    }
    if (tag === 'table') tables.push(node);
    for (const child of node.childNodes ?? []) prepare(child, container);
  };
  prepare(root);
  let heading: string | undefined;
  let bufferedText = '';
  let bufferStart: number | undefined;
  let bufferEnd: number | undefined;
  let bufferNode: HtmlNode | undefined;
  const add = (node: HtmlNode, text: string, source: Paragraph['source']): void => {
    let container: PageContainer | undefined;
    let section: PageContainer | undefined;
    const id = `p${paragraphs.length + 1}`;
    for (let current: HtmlNode | undefined = node; current; current = current.parentNode) {
      const ids = paragraphNodes.get(current) ?? []; ids.push(id); paragraphNodes.set(current, ids);
      const owner = containerNodes.get(current);
      if (owner) {
        owner.paragraphIds.push(id); container ??= owner;
        if (!section && ['section', 'article', 'main'].includes(owner.kind)) section = owner;
      }
      if (current === root) break;
    }
    paragraphs.push({ id, text, ...(heading ? { heading } : {}), ...(container ? { containerId: container.id } : {}), ...(section ? { sectionId: section.id } : {}), source });
  };
  const flushText = (): void => {
    const value = normalizeText(bufferedText);
    if (value && bufferNode) add(bufferNode, value, { kind: 'html', ...(bufferStart === undefined ? {} : { startOffset: bufferStart, endOffset: bufferEnd! }) });
    bufferedText = ''; bufferStart = undefined; bufferEnd = undefined; bufferNode = undefined;
  };
  const hasTable = (node: HtmlNode): boolean => (node.childNodes ?? []).some((child) => visible(child) && (child.tagName === 'table' || hasTable(child)));
  const visit = (node: HtmlNode): void => {
    if (!visible(node)) return;
    const tag = node.tagName;
    if (node.nodeName === '#text') {
      bufferedText += node.value ?? ''; bufferNode ??= node;
      if (node.sourceCodeLocation) {
        bufferStart = bufferStart === undefined ? node.sourceCodeLocation.startOffset : Math.min(bufferStart, node.sourceCodeLocation.startOffset);
        bufferEnd = bufferEnd === undefined ? node.sourceCodeLocation.endOffset : Math.max(bufferEnd, node.sourceCodeLocation.endOffset);
      }
      return;
    }
    if (tag && /^h[1-6]$/.test(tag)) {
      flushText(); const value = normalizeText(textContent(node));
      if (value) { heading = value; add(node, value, location(node)); }
      return;
    }
    if (tag && textBlockTags[tag] && !hasTable(node)) {
      flushText(); const value = normalizeText(textContent(node));
      if (value) add(node, value, location(node));
      return;
    }
    const boundary = !!tag && (!!flowContainerTags[tag] || ['table', 'tr', 'td', 'th', 'caption'].includes(tag));
    if (boundary) flushText();
    for (const child of node.childNodes ?? []) visit(child);
    if (boundary) flushText();
  };
  visit(root); flushText();
  const structuredTables = tables.map((table, index): PageTable => {
    const rows: { node: HtmlNode; group: HtmlNode }[] = [];
    const gatherRows = (node: HtmlNode, group: HtmlNode): void => {
      for (const child of node.childNodes ?? []) {
        if (!visible(child) || child.tagName === 'table') continue;
        if (child.tagName === 'tr') rows.push({ node: child, group });
        else gatherRows(child, ['thead', 'tbody', 'tfoot'].includes(child.tagName ?? '') ? child : group);
      }
    };
    gatherRows(table, table);
    const cells: TableCell[] = [];
    const nodes = new Map<HtmlNode, TableCell>();
    const occupied = new Map<number, number>();
    const rowGroups = new Map<HtmlNode, number>();
    const span = (node: HtmlNode, name: string, maximum: number): number => {
      const raw = attr(node, name)?.trim();
      const value = raw && /^\d+$/.test(raw) ? Number(raw) : 1;
      return Number.isSafeInteger(value) ? Math.max(1, Math.min(maximum, value)) : 1;
    };
    let columnCount = 0;
    for (let row = 0; row < rows.length; row++) {
      let column = 0;
      const entry = rows[row]!;
      if (!rowGroups.has(entry.group)) rowGroups.set(entry.group, rowGroups.size);
      for (const node of entry.node.childNodes ?? []) {
        if (!visible(node) || !['td', 'th'].includes(node.tagName ?? '')) continue;
        const columnSpan = span(node, 'colspan', 1000);
        let fits = false;
        while (!fits) {
          fits = true;
          for (let offset = 0; offset < columnSpan; offset++) if ((occupied.get(column + offset) ?? 0) > row) { fits = false; column++; break; }
        }
        let groupRows = 1;
        while (row + groupRows < rows.length && rows[row + groupRows]!.group === entry.group) groupRows++;
        const rowSpan = attr(node, 'rowspan')?.trim() === '0' ? groupRows : span(node, 'rowspan', Math.min(65534, groupRows));
        const scope = node.tagName === 'th' ? attr(node, 'scope')?.toLowerCase() : undefined;
        const cell: TableCell = { id: `t${index + 1}r${row + 1}c${column + 1}`, row, rowGroup: rowGroups.get(entry.group)!, column, rowSpan, columnSpan, kind: node.tagName === 'th' ? 'header' : 'data', ...(['row', 'col', 'rowgroup', 'colgroup'].includes(scope ?? '') ? { scope: scope as NonNullable<TableCell['scope']> } : {}), paragraphIds: paragraphNodes.get(node) ?? [], rowHeaderIds: [], columnHeaderIds: [], headerIds: [], source: location(node) };
        cells.push(cell); nodes.set(node, cell);
        for (let offset = 0; offset < columnSpan; offset++) occupied.set(column + offset, row + rowSpan);
        column += columnSpan; columnCount = Math.max(columnCount, column);
      }
    }
    const htmlIds = new Map<string, TableCell>();
    const duplicateIds = new Set<string>();
    for (const [node, cell] of nodes) {
      const id = attr(node, 'id');
      if (id) { if (htmlIds.has(id)) duplicateIds.add(id); else htmlIds.set(id, cell); }
    }
    for (const [node, cell] of nodes) {
      const explicit = attr(node, 'headers');
      if (explicit !== undefined) {
        cell.headerIds = [...new Set(explicit.split(/\s+/).map((id) => duplicateIds.has(id) ? undefined : htmlIds.get(id)).filter((header): header is TableCell => !!header && header.kind === 'header' && header !== cell).map(({ id }) => id))];
      }
      for (const header of cells) {
        if (header.kind !== 'header' || header === cell || (explicit !== undefined && !cell.headerIds.includes(header.id))) continue;
        const rowOverlap = header.row < cell.row + cell.rowSpan && cell.row < header.row + header.rowSpan;
        const columnOverlap = header.column < cell.column + cell.columnSpan && cell.column < header.column + header.columnSpan;
        const sameGroup = rows[header.row]?.group === rows[cell.row]?.group;
        if ((header.scope === 'rowgroup' && sameGroup) || ((header.scope === 'row' || !header.scope) && rowOverlap && header.column < cell.column)) cell.rowHeaderIds.push(header.id);
        if (header.scope === 'colgroup' ? columnOverlap : (header.scope === 'col' || !header.scope) && columnOverlap && header.row < cell.row) cell.columnHeaderIds.push(header.id);
      }
      if (explicit === undefined) cell.headerIds = [...new Set([...cell.rowHeaderIds, ...cell.columnHeaderIds])];
    }
    const captionNode = (table.childNodes ?? []).find((node) => node.tagName === 'caption' && visible(node));
    const captionText = captionNode ? normalizeText(textContent(captionNode)) : '';
    const owner = containerNodes.get(table);
    return { id: `t${index + 1}`, ...(owner ? { containerId: owner.id } : {}), ...(captionNode && captionText ? { caption: { text: captionText, paragraphIds: paragraphNodes.get(captionNode) ?? [], source: location(captionNode) } } : {}), rowCount: rows.length, columnCount, cells, paragraphIds: paragraphNodes.get(table) ?? [], source: location(table) };
  });
  return { paragraphs, containers, tables: structuredTables };
}

function collectImages(root: HtmlNode, pageUrl: string, maxImages: number): PageImage[] {
  const byUrl = new Map<string, PageImage>();
  const add = (raw: string, discovery: ImageDiscovery, node?: HtmlNode): void => {
    const url = resolveImage(raw, pageUrl);
    if (!url) return;
    const prior = byUrl.get(url);
    if (prior) {
      if (!prior.discoveredBy.includes(discovery)) prior.discoveredBy.push(discovery);
      if (!prior.alt && node?.tagName === 'img') {
        const alt = attr(node, 'alt');
        if (alt) prior.alt = alt;
      }
      return;
    }
    if (byUrl.size >= maxImages) throw new RangeError(`Page contains more than maxImages (${maxImages}) unique images`);
    const alt = node?.tagName === 'img' ? attr(node, 'alt') : undefined;
    const caption = node?.tagName === 'img' ? figureCaption(node) : undefined;
    byUrl.set(url, {
      id: `i${byUrl.size + 1}`,
      url,
      ...(alt ? { alt } : {}),
      ...(caption ? { caption } : {}),
      discoveredBy: [discovery],
      ...(node?.tagName ? { sourceElement: node.tagName } : {}),
    });
  };
  const visit = (node: HtmlNode): void => {
    if (node.tagName === 'img') {
      const kind: ImageDiscovery = isInsidePicture(node) ? 'picture' : 'img';
      const src = attr(node, 'src');
      if (src) add(src, kind, node);
      for (const url of parseSrcset(attr(node, 'srcset') ?? '')) add(url, kind, node);
    } else if (node.tagName === 'source' && isInsidePicture(node)) {
      for (const url of parseSrcset(attr(node, 'srcset') ?? '')) add(url, 'picture', node);
    }
    if (node.tagName === 'meta') {
      const property = (attr(node, 'property') ?? attr(node, 'name') ?? '').toLowerCase();
      if (property === 'og:image' || property === 'og:image:url') {
        const content = attr(node, 'content');
        if (content) add(content, 'og:image', node);
      }
    }
    for (const { name, value } of node.attrs ?? []) {
      if (name === 'background') add(value, 'background', node);
      else if (name === 'style') for (const url of cssImageUrls(value)) add(url, 'background', node);
    }
    if (node.tagName === 'style') {
      for (const url of cssImageUrls((node.childNodes ?? []).map((child) => child.value ?? '').join(''))) add(url, 'background', node);
    }
    for (const child of node.childNodes ?? []) visit(child);
  };
  visit(root);
  return [...byUrl.values()];
}

function timeoutSignal(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timed = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timed]) : timed;
}

export async function boundedBytes(response: Response, maxBytes: number, signal: AbortSignal): Promise<Uint8Array<ArrayBuffer>> {
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > maxBytes) {
    await response.body?.cancel();
    throw new RangeError(`Response exceeds byte limit (${maxBytes})`);
  }
  if (!response.body) throw new TypeError('Response has no body');
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel(signal.reason).catch(() => undefined); };
  signal.addEventListener('abort', cancel, { once: true });
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      signal.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) {
        await reader.cancel();
        throw new RangeError(`Response exceeds byte limit (${maxBytes})`);
      }
      chunks.push(value);
    }
  } finally {
    signal.removeEventListener('abort', cancel);
    reader.releaseLock();
  }
  const result = new Uint8Array(size);
  signal.throwIfAborted();
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

export async function fetchLimited(url: string, maxBytes: number, timeoutMs: number, signal?: AbortSignal, validateDestination?: ExtractOptions['validateDestination']): Promise<{ url: string; contentType: string; bytes: Uint8Array<ArrayBuffer> }> {
  let current = new URL(url);
  const abort = timeoutSignal(signal, timeoutMs);
  let response: Response | undefined;
  for (let redirects = 0; redirects <= 10; redirects++) {
    abort.throwIfAborted();
    if (current.protocol !== 'http:' && current.protocol !== 'https:') throw new TypeError('Only http and https URLs are supported');
    if (validateDestination) await awaitUser(() => validateDestination(current), abort, 'extract');
    abort.throwIfAborted();
    response = await fetch(current, { signal: abort, redirect: validateDestination ? 'manual' : 'follow' });
    if (![301, 302, 303, 307, 308].includes(response.status)) break;
    const destination = response.headers.get('location');
    await response.body?.cancel();
    if (!destination) throw new TypeError('Redirect response has no location');
    if (redirects === 10) throw new RangeError('HTTP redirect limit exceeded');
    current = new URL(destination, current);
  }
  if (!response) throw new Error('No HTTP response');
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`HTTP request failed with status ${response.status}`);
  }
  const finalUrl = new URL(response.url || current.href);
  if (finalUrl.protocol !== 'http:' && finalUrl.protocol !== 'https:') {
    await response.body?.cancel();
    throw new TypeError('Only http and https URLs are supported');
  }
  return {
    url: finalUrl.href,
    contentType: response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase() ?? '',
    bytes: await boundedBytes(response, maxBytes, abort),
  };
}

export async function extractPage(input: string, options: ExtractOptions = {}, context: ExtractContext = {}): Promise<Page> {
  const maxHtmlBytes = options.maxHtmlBytes ?? defaultHtmlLimit;
  const maxImages = options.maxImages ?? defaultImageCount;
  const timeoutMs = options.timeoutMs ?? defaultTimeout;
  if (!Number.isSafeInteger(maxHtmlBytes) || maxHtmlBytes < 1) throw new RangeError('maxHtmlBytes must be a positive safe integer');
  if (!Number.isSafeInteger(maxImages) || maxImages < 0) throw new RangeError('maxImages must be a non-negative safe integer');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2_147_483_647) throw new RangeError('timeoutMs must be between 1 and 2147483647');
  if (options.content !== undefined && options.content !== 'full' && options.content !== 'main') throw new TypeError('content must be full or main');
  let html: string;
  let pageUrl = options.baseUrl ?? '';
  if (/^[a-z][a-z\d+.-]*:\/\//i.test(input.trim())) {
    const source = new URL(input);
    if (source.protocol !== 'http:' && source.protocol !== 'https:') throw new TypeError('Only http and https URLs are supported');
    const destination = context.policy ? policyDestination(context.policy, 'page', context.offline, options.validateDestination, options.signal) : options.validateDestination;
    const result = await fetchLimited(source.href, maxHtmlBytes, timeoutMs, options.signal, destination);
    if (!['text/html', 'application/xhtml+xml'].includes(result.contentType)) throw new TypeError(`Expected an HTML response, got ${result.contentType || 'unknown content type'}`);
    pageUrl = result.url;
    html = new TextDecoder().decode(result.bytes);
  } else {
    if (/^[a-z][a-z\d+.-]*:/i.test(input.trim()) && !input.trim().startsWith('<')) throw new TypeError('Only http and https URLs are supported');
    if (new TextEncoder().encode(input).byteLength > maxHtmlBytes) throw new RangeError(`HTML input exceeds byte limit (${maxHtmlBytes})`);
    if (pageUrl) {
      const base = new URL(pageUrl);
      if (base.protocol !== 'http:' && base.protocol !== 'https:') throw new TypeError('Only http and https base URLs are supported');
      pageUrl = base.href;
    }
    html = input;
  }
  options.signal?.throwIfAborted();
  const document = parse(html, { scriptingEnabled: false, sourceCodeLocationInfo: true }) as unknown as HtmlNode;
  const titleNode = findFirst(document, 'title');
  const title = titleNode ? normalizeText(textContent(titleNode)) : '';
  const root = contentRoot(document, options.content ?? 'full');
  const structure = extractStructure(root.node);
  const images = options.includeImages === false ? [] : collectImages(root.extraction.root === 'body' ? document : root.node, pageUrl, maxImages);
  return { url: pageUrl || options.baseUrl || 'about:blank', ...(title ? { title } : {}), ...structure, images, extraction: root.extraction };
}

function findFirst(node: HtmlNode, tag: string): HtmlNode | undefined {
  if (node.tagName === tag) return node;
  for (const child of node.childNodes ?? []) {
    const found = findFirst(child, tag);
    if (found) return found;
  }
  return undefined;
}

