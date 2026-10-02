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

export interface ReportImage {
  imageId: string;
  url: string;
  description: string;
  source: { kind: 'image'; imageId: string };
  alt?: string;
}

export interface StructuredReport {
  language: string;
  page: { url: string; title?: string; summary: string };
  sections: ReportSection[];
  images: ReportImage[];
  conclusion: string;
}
