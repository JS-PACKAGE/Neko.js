import type { ErrorStage, ErrorCode } from './errors.js';
import type { ModelIdentity, LoadedBackend } from './core/engine.js';

export type ImageDiscovery = 'img' | 'picture' | 'background' | 'og:image';

export interface Paragraph {
  id: string;
  text: string;
  heading?: string;
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

export interface Page {
  url: string;
  title?: string;
  paragraphs: Paragraph[];
  images: PageImage[];
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
export type Citation =
  | { kind: 'quote'; snapshotId: string; paragraphId: string; versionId: string; startOffset: number; endOffset: number; quote: string }
  | { kind: 'image-observation'; snapshotId: string; imageId: string; versionId: string };
export interface ReportClaim {
  id: string;
  target: string;
  startOffset: number;
  endOffset: number;
  citations: Citation[];
  verification: 'references-validated';
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
  | { status: 'described'; description: string; observation: { versionId: string; width: number; height: number; verification: 'model-observation' } }
  | { status: 'failed'; error: { stage: ErrorStage; code: ErrorCode; message: string } }
);

export interface StructuredReport {
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
    backend: LoadedBackend;
    usage: { inputTokens: number; outputTokens: number; totalTokens: number };
    timings: { loadMs: number; preprocessMs: number; generationMs: number; totalMs: number; queueWaitMs: number };
    memory: { jsHeapBytes: null; gpuBytes: null };
    execution: ExecutionInfo;
    resumedStages: number;
    evidence: 'references-validated-not-fact-checked';
  };
}
