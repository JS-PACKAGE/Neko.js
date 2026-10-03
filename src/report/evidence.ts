import type { PageSnapshot, ReportClaim, ReportCoverage, ReportSourceFact, StructuredReport } from '../types.js';
import { hashValue } from '../web/source.js';

export async function reportChecksum(report: StructuredReport): Promise<string> {
  return hashValue({ ...report, integrity: undefined });
}

/** Checks exact contiguous source retention; deliberately makes no semantic claim. */
export function validateSourceFacts(value: unknown, snapshot: PageSnapshot): asserts value is ReportSourceFact[] {
  if (!Array.isArray(value)) throw new TypeError('Source fact ledger must be an array');
  const versions = new Map(snapshot.paragraphs.map((item) => [item.id, item.versionId]));
  let index = 0;
  for (const paragraph of snapshot.source.paragraphs) {
    let end = 0;
    while (end < paragraph.text.length) {
      const fact = value[index];
      if (!fact || typeof fact !== 'object' || fact.id !== `q${index + 1}`) throw new TypeError('Source fact ledger IDs or coverage are invalid');
      const quote = fact.citation;
      if (!quote || quote.kind !== 'quote' || quote.snapshotId !== snapshot.id || quote.paragraphId !== paragraph.id || quote.versionId !== versions.get(paragraph.id) || quote.startOffset !== end || !Number.isSafeInteger(quote.endOffset) || quote.endOffset <= end || quote.endOffset > paragraph.text.length || quote.quote !== paragraph.text.slice(end, quote.endOffset)) throw new TypeError('Source fact ledger quote or provenance is invalid');
      end = quote.endOffset; index++;
    }
  }
  if (index !== value.length) throw new TypeError('Source fact ledger contains extra quotes');
}

export function sourceCoverage(snapshot: PageSnapshot, facts: ReportSourceFact[], claims: ReportClaim[], conclusionBasis: ReportCoverage['conclusionBasis']): ReportCoverage {
  const cited = (predicate: (claim: ReportClaim) => boolean) => {
    const keys = new Set(claims.filter(predicate).flatMap((claim) => claim.citations.filter((citation) => citation.kind === 'quote').map((quote) => `${quote.paragraphId}:${quote.startOffset}:${quote.endOffset}`)));
    return facts.filter(({ citation }) => keys.has(`${citation.paragraphId}:${citation.startOffset}:${citation.endOffset}`)).map(({ id }) => id);
  };
  return {
    selectedParagraphIds: snapshot.source.paragraphs.map(({ id }) => id),
    selectedTextCharacters: snapshot.source.paragraphs.reduce((sum, paragraph) => sum + paragraph.text.length, 0),
    retainedQuoteCount: facts.length,
    retainedTextCharacters: facts.reduce((sum, fact) => sum + fact.citation.quote.length, 0),
    modelCitedFactIds: cited(() => true),
    summaryCitedFactIds: cited((claim) => claim.target === 'page.summary'),
    conclusionBasis,
    semanticRetention: 'not-measured',
  };
}
