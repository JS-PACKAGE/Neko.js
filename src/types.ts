import type { ErrorStage, ErrorCode } from './errors.js';
import type { ModelIdentity, LoadedBackend } from './core/engine.js';
import type { ClaimAudit } from './report/audit.js';
import type { ImageObservation } from './web/image.js';

export type ImageDiscovery = 'img' | 'picture' | 'background' | 'og:image';

export interface Paragraph {
  id: string;
  text: string;
  heading?: string;
  containerId?: string;
  sectionId?: string;
  source: {
    kind: 'html';
    startOffset?: number;
    endOffset?: number;
  };
}

export interface PageImage {
  id: string;
  url: string;
  alt?: string;
  caption?: string;
  discoveredBy: ImageDiscovery[];
  sourceElement?: string;
}

export interface PageContainer {
  id: string;
  kind: string;
  parentId?: string;
  paragraphIds: string[];
  source: Paragraph['source'];
}
export interface TableCell {
  id: string;
  row: number;
  rowGroup: number;
  column: number;
  rowSpan: number;
  columnSpan: number;
  kind: 'header' | 'data';
  scope?: 'row' | 'col' | 'rowgroup' | 'colgroup';
  paragraphIds: string[];
  rowHeaderIds: string[];
  columnHeaderIds: string[];
  headerIds: string[];
  source: Paragraph['source'];
}
export interface PageTable {
  id: string;
  containerId?: string;
  caption?: { text: string; paragraphIds: string[]; source: Paragraph['source'] };
  rowCount: number;
  columnCount: number;
  cells: TableCell[];
  paragraphIds: string[];
  /** Selection removed source paragraphs; empty cells retain geometry, not omitted text. */
  partial?: boolean;
  source: Paragraph['source'];
}

export interface Page {
  url: string;
  title?: string;
  paragraphs: Paragraph[];
  images: PageImage[];
  containers?: PageContainer[];
  tables?: PageTable[];
  extraction?: { mode: 'full' | 'main'; root: 'body' | 'main' | 'article'; fallback: boolean };
}

export interface SourceSelection {
  paragraphIds?: string[];
  imageIds?: string[];
  paragraph?: (source: Omit<Readonly<Paragraph>, 'source'> & { readonly source: Readonly<Paragraph['source']> }) => boolean | Promise<boolean>;
  image?: (source: Omit<Readonly<PageImage>, 'discoveredBy'> & { readonly discoveredBy: readonly ImageDiscovery[] }) => boolean | Promise<boolean>;
}
export interface PageSnapshot {
  id: string;
  algorithm: 'sha256';
  /** Selected source text and metadata are persisted; visual bytes are not. */
  source: Page;
  paragraphs: { id: string; versionId: string }[];
  /** These identify metadata, not visual contents; observations have separate pixel digests. */
  images: { id: string; metadataVersionId: string }[];
}
export interface DocumentAnswerClaim {
  text: string;
  citations: Extract<Citation, { kind: 'quote' }>[];
  audit: ClaimAudit;
}
export interface DocumentAnswer {
  question: string;
  status: 'answered' | 'insufficient-evidence';
  answer: string;
  claims: DocumentAnswerClaim[];
  snapshot: PageSnapshot;
  evidence: 'exact-quotes-heuristic-audit-not-fact-checked';
}
export type Citation =
  | { kind: 'quote'; snapshotId: string; paragraphId: string; versionId: string; startOffset: number; endOffset: number; quote: string }
  | ({ kind: 'image-observation'; snapshotId: string; imageId: string } & Pick<ImageObservation, 'versionId' | 'sourceVersionId' | 'sourceWidth' | 'sourceHeight' | 'region' | 'normalizedRegion'>);
export interface ReportClaim {
  id: string;
  target: string;
  startOffset: number;
  endOffset: number;
  citations: Citation[];
  verification: 'references-validated';
  audit: ClaimAudit;
}
export interface ExecutionInfo {
  mode: 'inline' | 'worker';
  runtime: 'node' | 'browser';
  workerId?: string;
  threadId?: number;
}

export interface ReportSection {
  heading?: string;
  keyPoints: string[];
  paragraphIds: string[];
}

export interface ReportImageSource {
  imageId: string;
  url: string;
  source: { kind: 'image'; imageId: string };
  alt?: string;
}
export type ReportImage = ReportImageSource & (
  | { status: 'described'; description: string; observation: ImageObservation & { verification: 'model-observation' } }
  | { status: 'failed'; error: { stage: ErrorStage; code: ErrorCode; message: string } }
  | { status: 'retained' }
);

export interface ReportSourceFact {
  id: string;
  /** Exact source text, not a generated interpretation or a verified real-world fact. */
  citation: Extract<Citation, { kind: 'quote' }>;
}
export interface ReportCoverage {
  selectedParagraphIds: string[];
  /** Character counts use JavaScript UTF-16 offsets, like quote citations. */
  selectedTextCharacters: number;
  retainedQuoteCount: number;
  retainedTextCharacters: number;
  modelCitedFactIds: string[];
  summaryCitedFactIds: string[];
  conclusionBasis: 'retained-source' | 'reduced-generated-claims';
  semanticRetention: 'not-measured';
}

export interface StructuredReport {
  schemaVersion: 3;
  mode: 'generated' | 'extractive';
  integrity: { algorithm: 'sha256'; checksum: string };
  sourceFacts: ReportSourceFact[];
  language: string;
  imageFailurePolicy: 'error' | 'omit';
  page: { url: string; title?: string; summary: string };
  sections: ReportSection[];
  images: ReportImage[];
  conclusion: string;
  snapshot: PageSnapshot;
  claims: ReportClaim[];
  metadata: {
    model: ModelIdentity;
    backend: LoadedBackend | null;
    usage: { inputTokens: number; outputTokens: number; totalTokens: number };
    timings: { loadMs: number; preprocessMs: number; generationMs: number; totalMs: number; queueWaitMs: number };
    memory: { jsHeapBytes: null; gpuBytes: null };
    execution: ExecutionInfo;
    resumedStages: number;
    retryAttempts: number;
    evidence: 'references-validated-heuristic-audit';
    coverage: ReportCoverage;
  };
}
