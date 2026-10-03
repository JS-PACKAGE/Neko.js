import type { StructTreeNode, TextContent, TextItem } from 'pdfjs-dist/types/src/display/api.js';
import type { DocumentBounds, ExtractedBlock, ExtractedTable } from './types.js';

export function unionBounds(blocks: ExtractedBlock[]): DocumentBounds {
  if (!blocks.length) return [0, 0, 0, 0];
  let left = Infinity; let top = Infinity; let right = -Infinity; let bottom = -Infinity;
  for (const { bounds: [x, y, width, height] } of blocks) { left = Math.min(left, x); top = Math.min(top, y); right = Math.max(right, x + width); bottom = Math.max(bottom, y + height); }
  return [left, top, right - left, bottom - top];
}

export function nativeLayout(content: TextContent, tree: StructTreeNode | null, pageNumber: number, transform: number[], width: number, height: number, limits: { maxItems: number; maxCharacters: number; maxCells: number }, signal?: AbortSignal): { blocks: ExtractedBlock[]; tables: ExtractedTable[]; tagged: boolean; itemCount: number; characters: number } {
  if (content.items.length > limits.maxItems) throw new RangeError('PDF text items exceed limit');
  const blocks: ExtractedBlock[] = []; const marks: string[] = []; const byMark = new Map<string, ExtractedBlock[]>(); let characters = 0;
  for (const [index, item] of content.items.entries()) {
    signal?.throwIfAborted();
    if (!('str' in item)) { if (item.type === 'beginMarkedContentProps' || item.type === 'beginMarkedContent') marks.push(item.id ?? ''); else if (item.type === 'endMarkedContent') marks.pop(); continue; }
    const textItem = item as TextItem;
    if (!textItem.str.trim()) continue;
    characters += textItem.str.length;
    if (characters > limits.maxCharacters) throw new RangeError('PDF text exceeds character limit');
    const m = textItem.transform as number[];
    if (m.length !== 6 || ![...m, textItem.width, textItem.height].every(Number.isFinite)) throw new TypeError('PDF text geometry is invalid');
    const a = transform[0]! * m[0]! + transform[2]! * m[1]!; const b = transform[1]! * m[0]! + transform[3]! * m[1]!;
    const c = transform[0]! * m[2]! + transform[2]! * m[3]!; const d = transform[1]! * m[2]! + transform[3]! * m[3]!;
    const x = transform[0]! * m[4]! + transform[2]! * m[5]! + transform[4]!; const y = transform[1]! * m[4]! + transform[3]! * m[5]! + transform[5]!;
    const horizontal = Math.hypot(a, b); const vertical = Math.hypot(c, d);
    const style = content.styles[textItem.fontName]; const ascent = style?.ascent ?? (style?.descent === undefined ? 0.8 : 1 + style.descent);
    const ux = horizontal ? a / horizontal : 1; const uy = horizontal ? b / horizontal : 0; const vx = vertical ? c / vertical : 0; const vy = vertical ? d / vertical : -1;
    const pageScale = Math.hypot(transform[0]!, transform[1]!);
    const advance = Math.abs(textItem.width) * pageScale; const glyphHeight = Math.max(Math.abs(textItem.height) * pageScale, vertical, 0.01);
    const points = [[x + vx * glyphHeight * ascent, y + vy * glyphHeight * ascent], [x + ux * advance + vx * glyphHeight * ascent, y + uy * advance + vy * glyphHeight * ascent], [x + vx * glyphHeight * (ascent - 1), y + vy * glyphHeight * (ascent - 1)], [x + ux * advance + vx * glyphHeight * (ascent - 1), y + uy * advance + vy * glyphHeight * (ascent - 1)]];
    const left = Math.max(0, Math.min(width, Math.min(...points.map((point) => point[0]!)))); const top = Math.max(0, Math.min(height, Math.min(...points.map((point) => point[1]!))));
    const right = Math.max(left, Math.min(width, Math.max(...points.map((point) => point[0]!)))); const bottom = Math.max(top, Math.min(height, Math.max(...points.map((point) => point[1]!))));
    let mark: string | undefined;
    for (let level = marks.length - 1; level >= 0; level--) if (marks[level]) { mark = marks[level]; break; }
    const block: ExtractedBlock = { id: `p${pageNumber}-b${index + 1}`, text: textItem.str, bounds: [left, top, right - left, bottom - top], order: blocks.length, kind: 'text', provenance: 'native-text', sourceItemIndices: [index], ...(mark ? { markedContentId: mark } : {}) };
    blocks.push(block);
    if (mark) { const grouped = byMark.get(mark) ?? []; grouped.push(block); byMark.set(mark, grouped); }
  }
  const tables: ExtractedTable[] = []; let cellCount = 0; let cellSlots = 0; let nodeCount = 0;
  const taggedOrder: ExtractedBlock[] = []; const orderedIds = new Set<string>();
  const contents = (node: StructTreeNode, depth = 0): ExtractedBlock[] => {
    if (depth > 128 || ++nodeCount > limits.maxItems) throw new RangeError('PDF structure tree exceeds bounds');
    const found: ExtractedBlock[] = [];
    for (const child of node.children) { const grouped = 'role' in child ? contents(child, depth + 1) : child.type === 'content' ? byMark.get(child.id) ?? [] : []; for (const block of grouped) found.push(block); }
    return found;
  };
  const walk = (node: StructTreeNode, depth = 0): void => {
    signal?.throwIfAborted();
    if (depth > 128 || ++nodeCount > limits.maxItems) throw new RangeError('PDF structure tree exceeds bounds');
    if (node.role === 'Table') {
      const table: ExtractedTable = { id: `p${pageNumber}-t${tables.length + 1}`, structure: 'pdf-tags', cells: [] }; let row = 0;
      const occupied = new Set<string>();
      const rows = (container: StructTreeNode, level: number): void => {
        if (level > 128 || ++nodeCount > limits.maxItems) throw new RangeError('PDF table tree exceeds bounds');
        for (const child of container.children) {
          if (!('role' in child)) continue;
          if (child.role !== 'TR') { if (child.role !== 'Table') rows(child, level + 1); continue; }
          let column = 0;
          for (const cell of child.children) {
            if (!('role' in cell) || !['TD', 'TH'].includes(cell.role)) continue;
            if (++cellCount > limits.maxCells) throw new RangeError('PDF table cells exceed limit');
            while (occupied.has(`${row}:${column}`)) column++;
            const rowSpan = cell.rowSpan ?? 1; const columnSpan = cell.colSpan ?? 1;
            if (!Number.isSafeInteger(rowSpan) || !Number.isSafeInteger(columnSpan) || rowSpan < 1 || columnSpan < 1 || row + rowSpan > 256 || column + columnSpan > 256 || rowSpan * columnSpan > limits.maxCells) throw new RangeError('PDF table spans exceed bounds');
            cellSlots += rowSpan * columnSpan;
            if (cellSlots > limits.maxCells) throw new RangeError('PDF table spans exceed aggregate cell limit');
            for (let r = row; r < row + rowSpan; r++) for (let c = column; c < column + columnSpan; c++) { const key = `${r}:${c}`; if (occupied.has(key)) throw new TypeError('PDF table cells overlap'); occupied.add(key); }
            const grouped = contents(cell); for (const block of grouped) block.kind = 'table-cell';
            table.cells.push({ row, column, rowSpan, columnSpan, header: cell.role === 'TH', blockIds: grouped.map(({ id }) => id), bounds: unionBounds(grouped) }); column += columnSpan;
          }
          row++;
        }
      };
      rows(node, depth); if (table.cells.length) tables.push(table);
    }
    for (const child of node.children) { if ('role' in child) walk(child, depth + 1); else if (child.type === 'content') for (const block of byMark.get(child.id) ?? []) if (!orderedIds.has(block.id)) { orderedIds.add(block.id); taggedOrder.push(block); } }
  };
  if (tree) walk(tree);
  const geometric = [...blocks].sort((a, b) => a.bounds[1] - b.bounds[1] || a.bounds[0] - b.bounds[0] || a.order - b.order);
  const ordered = taggedOrder.length ? [...taggedOrder, ...geometric.filter(({ id }) => !orderedIds.has(id))] : geometric;
  ordered.forEach((block, index) => { block.order = index; });
  if (!tables.length) {
    const rows: ExtractedBlock[][] = [];
    for (const block of geometric) { const last = rows.at(-1); if (last && Math.abs(last[0]!.bounds[1] - block.bounds[1]) <= Math.max(2, Math.min(last[0]!.bounds[3], block.bounds[3]) * 0.4)) last.push(block); else rows.push([block]); }
    const split = (row: ExtractedBlock[]): ExtractedBlock[][] => { const cells: ExtractedBlock[][] = []; for (const block of row.sort((a, b) => a.bounds[0] - b.bounds[0])) { const last = cells.at(-1); const previous = last?.at(-1); if (last && previous && block.bounds[0] - previous.bounds[0] - previous.bounds[2] < Math.max(8, block.bounds[3] * 1.5)) last.push(block); else cells.push([block]); } return cells; };
    let run: ExtractedBlock[][][] = [];
    const flush = () => { if (run.length >= 2) { const table: ExtractedTable = { id: `p${pageNumber}-t${tables.length + 1}`, structure: 'geometry-heuristic', cells: [] }; run.forEach((row, r) => row.forEach((cell, c) => { if (++cellCount > limits.maxCells) throw new RangeError('PDF table cells exceed limit'); cell.forEach((block) => { block.kind = 'table-cell'; }); table.cells.push({ row: r, column: c, rowSpan: 1, columnSpan: 1, header: false, blockIds: cell.map(({ id }) => id), bounds: unionBounds(cell) }); })); tables.push(table); } run = []; };
    for (const row of rows) { const cells = split(row); const previous = run.at(-1); const aligned = previous && previous.length === cells.length && cells.every((cell, index) => Math.abs(cell[0]!.bounds[0] - previous[index]![0]!.bounds[0]) < Math.max(4, cell[0]!.bounds[3])); const adjacent = previous && row[0]!.bounds[1] - previous[0]![0]!.bounds[1] <= Math.max(36, row[0]!.bounds[3] * 3); if (cells.length < 2) { flush(); continue; } if (run.length && (!aligned || !adjacent)) flush(); run.push(cells); }
    flush();
  }
  return { blocks: ordered, tables, tagged: taggedOrder.length > 0, itemCount: content.items.length, characters };
}
