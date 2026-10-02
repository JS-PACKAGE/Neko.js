import type { ErrorStage, ErrorCode } from './errors.js';

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
  | { status: 'described'; description: string }
  | { status: 'failed'; error: { stage: ErrorStage; code: ErrorCode; message: string } }
);

export interface StructuredReport {
  language: string;
  imageFailurePolicy: 'error' | 'omit';
  page: { url: string; title?: string; summary: string };
  sections: ReportSection[];
  images: ReportImage[];
  conclusion: string;
}
