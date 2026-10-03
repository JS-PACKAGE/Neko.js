import type { Page, PageImage, Paragraph, PageSnapshot, SourceSelection, PageContainer, PageTable, TableCell } from '../types.js';
import type * as NodeCrypto from 'node:crypto';
import { awaitUser } from '../errors.js';

function record(value: unknown, name: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${name} must be an object`);
}
function text(value: unknown, name: string): asserts value is string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be non-empty text`);
}
export function validatePage(value: unknown): asserts value is Page {
  record(value, 'Page'); text(value.url, 'Page URL');
  const url = new URL(value.url);
  if (!['http:', 'https:', 'about:'].includes(url.protocol) || (url.protocol === 'about:' && url.href !== 'about:blank')) throw new TypeError('Page URL must be HTTP(S) or about:blank');
  if (value.title !== undefined && typeof value.title !== 'string') throw new TypeError('Page title must be text');
  if (!Array.isArray(value.paragraphs) || !Array.isArray(value.images)) throw new TypeError('Page sources must be arrays');
  const ids = new Set<string>();
  for (const paragraph of value.paragraphs) {
    record(paragraph, 'Paragraph'); text(paragraph.id, 'Paragraph ID'); text(paragraph.text, 'Paragraph text');
    if (ids.has(paragraph.id)) throw new TypeError(`Duplicate source ID ${paragraph.id}`); ids.add(paragraph.id);
    if (paragraph.heading !== undefined && typeof paragraph.heading !== 'string') throw new TypeError('Paragraph heading must be text');
    record(paragraph.source, 'Paragraph source');
    if (paragraph.source.kind !== 'html') throw new TypeError('Paragraph source must be html');
    const { startOffset, endOffset } = paragraph.source;
    if ((startOffset === undefined) !== (endOffset === undefined) || (startOffset !== undefined && (!Number.isSafeInteger(startOffset) || !Number.isSafeInteger(endOffset) || (startOffset as number) < 0 || (endOffset as number) < (startOffset as number)))) throw new TypeError('Paragraph source offsets must be an ordered pair of non-negative integers');
  }
  for (const image of value.images) {
    record(image, 'Image'); text(image.id, 'Image ID'); text(image.url, 'Image URL');
    if (ids.has(image.id)) throw new TypeError(`Duplicate source ID ${image.id}`); ids.add(image.id);
    if (!Array.isArray(image.discoveredBy) || !image.discoveredBy.length || image.discoveredBy.some((kind: unknown) => !['img', 'picture', 'background', 'og:image'].includes(String(kind))) || new Set(image.discoveredBy).size !== image.discoveredBy.length) throw new TypeError('Image discovery sources are invalid');
    for (const key of ['alt', 'caption', 'sourceElement']) if (image[key] !== undefined && typeof image[key] !== 'string') throw new TypeError(`Image ${key} must be text`);
    if (/^[a-z][a-z\d+.-]*:/i.test(image.url) && !/^(https?:|data:image\/(?:png|jpeg|webp|gif|avif|tiff|bmp)[;,])/i.test(image.url)) throw new TypeError('Page image must be HTTP(S), raster data, or an unresolved relative source');
  }
  validateRelations(value as unknown as Page);
}
function validateRelations(page: Page): void {
  const paragraphs = new Map(page.paragraphs.map((paragraph) => [paragraph.id, paragraph]));
  const containers = new Map<string, PageContainer>();
  const ids = new Set([...paragraphs.keys(), ...page.images.map(({ id }) => id)]);
  const addId = (id: unknown, name: string): void => {
    text(id, name);
    if (ids.has(id)) throw new TypeError(`Duplicate source ID ${id}`);
    ids.add(id);
  };
  const references = (value: unknown, available: ReadonlyMap<string, unknown>, name: string): string[] => {
    if (!Array.isArray(value) || value.some((id) => typeof id !== 'string' || !available.has(id)) || new Set(value).size !== value.length) throw new TypeError(`${name} must contain unique existing references`);
    return value as string[];
  };
  const source = (value: unknown): void => {
    record(value, 'Relation source');
    const { startOffset, endOffset } = value;
    if (value.kind !== 'html' || (startOffset === undefined) !== (endOffset === undefined) || (startOffset !== undefined && (!Number.isSafeInteger(startOffset) || !Number.isSafeInteger(endOffset) || (startOffset as number) < 0 || (endOffset as number) < (startOffset as number)))) throw new TypeError('Relation source must have ordered HTML offsets');
  };
  if (page.containers !== undefined) {
    if (!Array.isArray(page.containers)) throw new TypeError('Page containers must be an array');
    for (const container of page.containers) {
      record(container, 'Container'); addId(container.id, 'Container ID'); text(container.kind, 'Container kind');
      source(container.source); references(container.paragraphIds, paragraphs, 'Container paragraphIds');
      containers.set(container.id, container);
    }
    for (const container of page.containers) {
      const visited = new Set<string>([container.id]);
      let current = container;
      while (current.parentId !== undefined) {
        const parent = containers.get(current.parentId);
        if (!parent || visited.has(parent.id)) throw new TypeError('Container parent relation is missing or cyclic');
        if (current.paragraphIds.some((id) => !parent.paragraphIds.includes(id))) throw new TypeError('Container parent omits descendant paragraphs');
        visited.add(parent.id); current = parent;
      }
    }
  }
  for (const paragraph of page.paragraphs) {
    for (const key of ['containerId', 'sectionId'] as const) {
      const id = paragraph[key];
      if (id !== undefined) {
        const container = containers.get(id);
        if (!container || !container.paragraphIds.includes(paragraph.id) || (key === 'sectionId' && !['section', 'article', 'main'].includes(container.kind))) throw new TypeError(`Paragraph ${key} relation is invalid`);
      }
    }
    if (paragraph.sectionId && paragraph.containerId) {
      let current = containers.get(paragraph.containerId);
      while (current && current.id !== paragraph.sectionId) current = current.parentId ? containers.get(current.parentId) : undefined;
      if (!current) throw new TypeError('Paragraph section must contain its container');
    }
  }
  if (page.tables !== undefined) {
    if (!Array.isArray(page.tables)) throw new TypeError('Page tables must be an array');
    for (const table of page.tables) {
      record(table, 'Table'); addId(table.id, 'Table ID'); source(table.source);
      const tableParagraphs = references(table.paragraphIds, paragraphs, 'Table paragraphIds');
      if (!Number.isSafeInteger(table.rowCount) || table.rowCount < 0 || !Number.isSafeInteger(table.columnCount) || table.columnCount < 0 || !Array.isArray(table.cells) || (table.partial !== undefined && typeof table.partial !== 'boolean')) throw new TypeError('Table dimensions or selection metadata are invalid');
      if (table.containerId !== undefined) {
        const container = containers.get(table.containerId);
        if (!container || container.kind !== 'table' || tableParagraphs.some((id) => !container.paragraphIds.includes(id))) throw new TypeError('Table container is invalid');
      }
      const cells = new Map<string, TableCell>();
      const covered = new Set<string>();
      const rowGroups = new Map<number, number>();
      if (table.caption !== undefined) {
        record(table.caption, 'Table caption'); text(table.caption.text, 'Table caption text'); source(table.caption.source);
        const captionParagraphs = references(table.caption.paragraphIds, paragraphs, 'Caption paragraphIds');
        if (!captionParagraphs.length || captionParagraphs.map((id) => paragraphs.get(id)!.text).join(' ') !== table.caption.text) throw new TypeError('Table caption does not match its source paragraphs');
        for (const id of captionParagraphs) covered.add(id);
      }
      for (const cell of table.cells) {
        record(cell, 'Table cell'); addId(cell.id, 'Cell ID'); source(cell.source);
        if (!['header', 'data'].includes(cell.kind) || (cell.scope !== undefined && (cell.kind !== 'header' || !['row', 'col', 'rowgroup', 'colgroup'].includes(cell.scope)))) throw new TypeError('Table cell kind or scope is invalid');
        if ([cell.row, cell.column, cell.rowSpan, cell.columnSpan].some((number) => !Number.isSafeInteger(number)) || cell.row < 0 || cell.column < 0 || cell.rowSpan < 1 || cell.columnSpan < 1 || cell.columnSpan > 1000 || cell.row + cell.rowSpan > table.rowCount || cell.column + cell.columnSpan > table.columnCount) throw new TypeError('Table cell geometry is invalid');
        if (!Number.isSafeInteger(cell.rowGroup) || cell.rowGroup < 0 || cell.rowGroup >= table.rowCount || (rowGroups.has(cell.row) && rowGroups.get(cell.row) !== cell.rowGroup)) throw new TypeError('Cell row group is invalid');
        rowGroups.set(cell.row, cell.rowGroup);
        for (const id of references(cell.paragraphIds, paragraphs, 'Cell paragraphIds')) covered.add(id);
        cells.set(cell.id, cell);
      }
      const occupied = new Map<number, number>();
      for (const cell of [...cells.values()].sort((left, right) => left.row - right.row || left.column - right.column)) {
        for (let column = cell.column; column < cell.column + cell.columnSpan; column++) {
          if ((occupied.get(column) ?? 0) > cell.row) throw new TypeError('Table cells overlap');
          occupied.set(column, cell.row + cell.rowSpan);
        }
      }
      for (const cell of cells.values()) {
        for (const key of ['headerIds', 'rowHeaderIds', 'columnHeaderIds'] as const) {
          for (const id of references(cell[key], cells, `Cell ${key}`)) {
            const header = cells.get(id)!;
            if (id === cell.id || header.kind !== 'header' || (key !== 'headerIds' && !cell.headerIds.includes(id))) throw new TypeError('Cell header relation is invalid');
            if (key === 'rowHeaderIds' && !(header.scope === 'rowgroup' ? header.rowGroup === cell.rowGroup : (!header.scope || header.scope === 'row') && header.row < cell.row + cell.rowSpan && cell.row < header.row + header.rowSpan && header.column < cell.column)) throw new TypeError('Row header geometry is invalid');
            if (key === 'columnHeaderIds' && !(header.column < cell.column + cell.columnSpan && cell.column < header.column + header.columnSpan && (header.scope === 'colgroup' || (!header.scope || header.scope === 'col') && header.row < cell.row))) throw new TypeError('Column header geometry is invalid');
          }
        }
      }
      if (covered.size !== tableParagraphs.length || tableParagraphs.some((id) => !covered.has(id))) throw new TypeError('Table paragraphs do not match its cells and caption');
    }
  }
  if (page.extraction !== undefined) {
    record(page.extraction, 'Extraction');
    if (!['full', 'main'].includes(page.extraction.mode) || !['body', 'main', 'article'].includes(page.extraction.root) || typeof page.extraction.fallback !== 'boolean' || (page.extraction.mode === 'full' && (page.extraction.root !== 'body' || page.extraction.fallback)) || (page.extraction.mode === 'main' && page.extraction.fallback !== (page.extraction.root === 'body'))) throw new TypeError('Extraction metadata is invalid');
  }
}

function freezeRelations<T>(value: T): T {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freezeRelations(child);
    Object.freeze(value);
  }
  return value;
}
function canonical(value: unknown, ancestors: Set<object>): string {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (typeof value !== 'object' || value === null || ancestors.has(value)) throw new TypeError('Snapshot contents must be finite, acyclic JSON data');
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return `[${value.map((item) => canonical(item, ancestors)).join(',')}]`;
    return `{${Object.keys(value).sort().filter((key) => (value as Record<string, unknown>)[key] !== undefined).map((key) => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key], ancestors)}`).join(',')}}`;
  } finally { ancestors.delete(value); }
}
export async function hashBytes(bytes: Uint8Array | Uint8ClampedArray): Promise<string> {
  if (globalThis.crypto?.subtle) {
    const buffer = bytes.buffer instanceof ArrayBuffer ? bytes : new Uint8Array(bytes);
    const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', buffer as Uint8Array<ArrayBuffer>));
    let result = ''; for (const byte of digest) result += byte.toString(16).padStart(2, '0'); return result;
  }
  // Node crypto is platform-specific and cannot be imported by browser bundles.
  const protocol = 'node:';
  const crypto: typeof NodeCrypto = await import(`${protocol}crypto`);
  return crypto.createHash('sha256').update(bytes).digest('hex');
}
export function hashValue(value: unknown): Promise<string> { return hashBytes(new TextEncoder().encode(canonical(value, new Set()))); }

const ownedPages = new WeakSet<Page>();
const snapshots = new WeakMap<Page, Promise<PageSnapshot>>();
function ownPage(input: Page): Page {
  if (ownedPages.has(input)) return input;
  validatePage(input);
  const { url, title } = input;
  const copySource = ({ kind, startOffset, endOffset }: Paragraph['source']): Paragraph['source'] => ({ kind, ...(startOffset === undefined ? {} : { startOffset, endOffset: endOffset! }) });
  const paragraphs: Paragraph[] = input.paragraphs.map(({ id, text, heading, containerId, sectionId, source }) => Object.freeze({ id, text, ...(heading === undefined ? {} : { heading }), ...(containerId === undefined ? {} : { containerId }), ...(sectionId === undefined ? {} : { sectionId }), source: Object.freeze({ kind: source.kind, ...(source.startOffset === undefined ? {} : { startOffset: source.startOffset, endOffset: source.endOffset }) }) }));
  const images: PageImage[] = input.images.map(({ id, url, alt, caption, discoveredBy, sourceElement }) => {
    const discovery = [...discoveredBy]; Object.freeze(discovery);
    return Object.freeze({ id, url, ...(alt === undefined ? {} : { alt }), ...(caption === undefined ? {} : { caption }), discoveredBy: discovery, ...(sourceElement === undefined ? {} : { sourceElement }) });
  });
  Object.freeze(paragraphs); Object.freeze(images);
  const containers = input.containers?.map(({ id, kind, parentId, paragraphIds, source }): PageContainer => ({ id, kind, ...(parentId === undefined ? {} : { parentId }), paragraphIds: [...paragraphIds], source: copySource(source) }));
  const tables = input.tables?.map(({ id, containerId, caption, rowCount, columnCount, cells, paragraphIds, partial, source }): PageTable => ({
    id, ...(containerId === undefined ? {} : { containerId }), rowCount, columnCount, paragraphIds: [...paragraphIds], ...(partial === undefined ? {} : { partial }), source: copySource(source),
    ...(caption === undefined ? {} : { caption: { text: caption.text, paragraphIds: [...caption.paragraphIds], source: copySource(caption.source) } }),
    cells: cells.map(({ id, row, rowGroup, column, rowSpan, columnSpan, kind, scope, paragraphIds, rowHeaderIds, columnHeaderIds, headerIds, source }): TableCell => ({ id, row, rowGroup, column, rowSpan, columnSpan, kind, ...(scope === undefined ? {} : { scope }), paragraphIds: [...paragraphIds], rowHeaderIds: [...rowHeaderIds], columnHeaderIds: [...columnHeaderIds], headerIds: [...headerIds], source: copySource(source) })),
  }));
  const extraction = input.extraction === undefined ? undefined : Object.freeze({ mode: input.extraction.mode, root: input.extraction.root, fallback: input.extraction.fallback });
  const page: Page = Object.freeze({ url, ...(title === undefined ? {} : { title }), paragraphs, images, ...(containers === undefined ? {} : { containers: freezeRelations(containers) }), ...(tables === undefined ? {} : { tables: freezeRelations(tables) }), ...(extraction === undefined ? {} : { extraction }) });
  validatePage(page); ownedPages.add(page); return page;
}
function selectedIds(requested: string[] | undefined, sources: { id: string }[], name: string): Set<string> | undefined {
  if (requested === undefined) return undefined;
  if (!Array.isArray(requested) || requested.some((id) => typeof id !== 'string') || new Set(requested).size !== requested.length) throw new TypeError(`${name} must contain unique source IDs`);
  const available = new Set(sources.map(({ id }) => id));
  for (const id of requested) if (!available.has(id)) throw new TypeError(`Unknown ${name} source ${id}`);
  return new Set(requested);
}
export async function selectPage(input: Page, selection: SourceSelection = {}, signal?: AbortSignal): Promise<Page> {
  const source = ownPage(input);
  const paragraphIds = selectedIds(selection.paragraphIds, source.paragraphs, 'paragraphIds');
  const imageIds = selectedIds(selection.imageIds, source.images, 'imageIds');
  if (selection.paragraph !== undefined && typeof selection.paragraph !== 'function') throw new TypeError('paragraph selector must be a function');
  if (selection.image !== undefined && typeof selection.image !== 'function') throw new TypeError('image selector must be a function');
  const paragraphPredicate = selection.paragraph; const imagePredicate = selection.image;
  if (!paragraphIds && !imageIds && !paragraphPredicate && !imagePredicate) return source;
  const paragraphs: Paragraph[] = []; const images: PageImage[] = [];
  for (const paragraph of source.paragraphs) if ((!paragraphIds || paragraphIds.has(paragraph.id)) && (!paragraphPredicate || await awaitUser(() => paragraphPredicate(paragraph), signal, 'extract'))) paragraphs.push(paragraph);
  for (const image of source.images) if ((!imageIds || imageIds.has(image.id)) && (!imagePredicate || await awaitUser(() => imagePredicate(image), signal, 'extract'))) images.push(image);
  Object.freeze(paragraphs); Object.freeze(images);
  const retained = new Set(paragraphs.map(({ id }) => id));
  const containers = source.containers?.filter((container) => container.paragraphIds.some((id) => retained.has(id))).map((container) => ({ ...container, paragraphIds: container.paragraphIds.filter((id) => retained.has(id)) }));
  const tables = source.tables?.filter((table) => table.paragraphIds.some((id) => retained.has(id))).map((table): PageTable => {
    const retainedHeaders = new Set(table.cells.filter((cell) => cell.kind === 'header' && cell.paragraphIds.some((id) => retained.has(id))).map(({ id }) => id));
    const paragraphIds = table.paragraphIds.filter((id) => retained.has(id));
    const { caption, ...rest } = table;
    const captionIds = caption?.paragraphIds.filter((id) => retained.has(id));
    return { ...rest, paragraphIds, ...(paragraphIds.length !== table.paragraphIds.length || table.partial ? { partial: true } : {}), ...(caption && captionIds?.length ? { caption: { ...caption, text: captionIds.map((id) => paragraphs.find((paragraph) => paragraph.id === id)!.text).join(' '), paragraphIds: captionIds } } : {}), cells: table.cells.map((cell) => ({ ...cell, paragraphIds: cell.paragraphIds.filter((id) => retained.has(id)), headerIds: cell.headerIds.filter((id) => retainedHeaders.has(id)), rowHeaderIds: cell.rowHeaderIds.filter((id) => retainedHeaders.has(id)), columnHeaderIds: cell.columnHeaderIds.filter((id) => retainedHeaders.has(id)) })) };
  });
  const page: Page = Object.freeze({ url: source.url, ...(source.title === undefined ? {} : { title: source.title }), paragraphs, images, ...(containers === undefined ? {} : { containers: freezeRelations(containers) }), ...(tables === undefined ? {} : { tables: freezeRelations(tables) }), ...(source.extraction === undefined ? {} : { extraction: source.extraction }) });
  validatePage(page); ownedPages.add(page); return page;
}
export async function snapshotPage(page: Page): Promise<PageSnapshot> {
  const source = ownPage(page);
  const previous = snapshots.get(source); if (previous) return previous;
  const id = hashValue(source);
  const paragraphs = source.paragraphs.map((paragraph) => { const id = paragraph.id; const digest = hashValue(paragraph); return digest.then((versionId) => Object.freeze({ id, versionId })); });
  const images = source.images.map((image) => { const id = image.id; const digest = hashValue(image); return digest.then((metadataVersionId) => Object.freeze({ id, metadataVersionId })); });
  const snapshot = Promise.all([id, Promise.all(paragraphs), Promise.all(images)]).then((values): PageSnapshot => { Object.freeze(values[1]); Object.freeze(values[2]); return Object.freeze({ id: values[0], algorithm: 'sha256', source, paragraphs: values[1], images: values[2] }); });
  snapshots.set(source, snapshot); return snapshot;
}
