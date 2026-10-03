import type { Citation, PageSnapshot } from '../types.js';

export type AuditKind = 'numbers' | 'units' | 'dates' | 'entities' | 'negation' | 'ranges';
export interface ClaimAudit {
  version: 1;
  status: 'supported' | 'contradicted' | 'unknown';
  method: 'conservative-lexical-v1';
  checks: { kind: AuditKind; status: 'matched' | 'conflict' | 'unresolved' | 'absent'; claim: string[]; evidence: string[] }[];
  limits: string[];
}
const patterns: Record<AuditKind, RegExp> = {
  numbers: /[+-]?\d+(?:[,.]\d+)*(?:%|％)?/gu,
  units: /(?:\b(?:kg|mg|g|km|cm|mm|m|ms|s|hours?|minutes?|seconds?|days?|USD|EUR|JPY|kilograms?|kilomet(?:er|re)s?|met(?:er|re)s?)\b|°[CF]|公斤|公里|公尺|毫秒|小時|分鐘|美元|元|%|％)/giu,
  dates: /\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b|\d{4}年\d{1,2}月\d{1,2}日/giu,
  entities: /\b[A-Z][\p{L}\p{N}]*(?:\s+[A-Z][\p{L}\p{N}]*)*\b/gu,
  negation: /\b(?:not|never|no|without|cannot|can't|doesn't|didn't|isn't|wasn't)\b|並非|不是|沒有|未曾|不會|不/giu,
  ranges: /\d+(?:\.\d+)?\s*(?:[-–—]|to|至|到)\s*\d+(?:\.\d+)?/giu,
};
const limits = [
  'Support means exact cited text retention, not real-world truth or semantic entailment.',
  'Contradictions require an otherwise identical sentence with one changed lexical fact or explicit negation.',
  'Paraphrases, translations, coreference, implicit negation, unit conversion, date equivalence, entity identity and image pixels are not verified.',
  'Lexical detection is incomplete and language-dependent; absence of a conflict is not support.',
];
const values = (text: string, kind: AuditKind) => [...new Set(Array.from(text.matchAll(patterns[kind]), (match) => match[0]))];
const normalized = (text: string) => text.replace(/\s+/gu, ' ').trim().replace(/[.!?。！？]+$/u, '');

/** Deliberately recognizes only exact quotations and narrowly aligned lexical conflicts. */
export function auditClaim(text: string, citations: readonly Citation[], snapshot?: PageSnapshot): ClaimAudit {
  const quotes = citations.filter((citation): citation is Extract<Citation, { kind: 'quote' }> => citation.kind === 'quote').filter((citation) => {
    if (!snapshot) return true;
    const paragraph = snapshot.source.paragraphs.find(({ id }) => id === citation.paragraphId);
    return citation.snapshotId === snapshot.id && Number.isSafeInteger(citation.startOffset) && Number.isSafeInteger(citation.endOffset) && citation.startOffset >= 0 && citation.endOffset > citation.startOffset && paragraph !== undefined && citation.endOffset <= paragraph.text.length && citation.versionId === snapshot.paragraphs.find(({ id }) => id === citation.paragraphId)?.versionId && paragraph.text.slice(citation.startOffset, citation.endOffset) === citation.quote;
  }).map(({ quote }) => quote);
  const exact = text.length > 0 && quotes.some((quote) => quote.includes(text));
  const conflicts = new Set<AuditKind>();
  if (!exact) for (const quote of quotes) {
    const candidates = [quote, ...Array.from(new Intl.Segmenter('en', { granularity: 'sentence' }).segment(quote), ({ segment }) => segment)];
    for (const candidate of candidates) {
      for (const kind of Object.keys(patterns) as AuditKind[]) {
        const a = values(text, kind); const b = values(candidate, kind);
        if (kind === 'negation') {
          if ((a.length === 0) !== (b.length === 0) && normalized(text.replace(patterns.negation, '')) === normalized(candidate.replace(patterns.negation, ''))) conflicts.add(kind);
        } else if (a.filter((value) => !b.includes(value)).length === 1 && b.filter((value) => !a.includes(value)).length === 1 && a.length === b.length && normalized(text.replace(patterns[kind], '<fact>')) === normalized(candidate.replace(patterns[kind], '<fact>'))) conflicts.add(kind);
      }
    }
  }
  return {
    version: 1, method: 'conservative-lexical-v1', status: exact ? 'supported' : conflicts.size ? 'contradicted' : 'unknown',
    checks: (Object.keys(patterns) as AuditKind[]).map((kind) => {
      const claim = values(text, kind); const evidence = [...new Set(quotes.flatMap((quote) => values(quote, kind)))];
      return { kind, status: conflicts.has(kind) ? 'conflict' : !claim.length ? 'absent' : exact ? 'matched' : 'unresolved', claim, evidence };
    }), limits: [...limits],
  };
}
