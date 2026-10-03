import { EVALUATOR_VERSION } from './evaluate.mjs';

const recordPattern = /(?:Observation\s+)?(?<code>H\d{3}):\s*station\s+(?<station>S\d{3})\s+at\s+(?<site>\w+)\s+plot\s+measured\s+(?<field>soil moisture|river temperature|wind speed|rainfall)\s+(?<value>\d+)\s+(?<unit>percent|degrees Celsius|kilometers per hour|millimeters)\s+on\s+(?<date>\d{4}-\d{2}-\d{2})\./gu;
const fields = ['code', 'station', 'site', 'field', 'value', 'unit', 'date'];

export function parseHierarchyRecords(source) {
  return String(source).trimEnd().split(/\r?\n/u).map((text, index) => {
    const matches = [...text.matchAll(new RegExp(recordPattern))];
    if (matches.length !== 1 || matches[0][0] !== text) throw new TypeError(`Invalid hierarchy oracle paragraph ${index + 1}`);
    return { ...matches[0].groups, sourceId: `p${index + 1}`, text };
  });
}

function scoreRecords(texts, records) {
  const byCode = new Map(records.map((record) => [record.code, record]));
  const matched = new Set();
  const falseClaims = [];
  const contradictions = [];
  const unsupported = [];
  const formatErrors = [];
  let duplicateCount = 0;
  for (const [textIndex, text] of texts.entries()) {
    const spans = [...text.matchAll(new RegExp(recordPattern))];
    for (const match of spans) {
      const claim = match.groups;
      const expected = byCode.get(claim.code);
      if (!expected) {
        unsupported.push({ id: claim.code, textIndex, excerpt: match[0] });
        falseClaims.push({ id: claim.code, reason: 'unknown-record' });
      } else if (!fields.every((field) => claim[field] === expected[field])) {
        contradictions.push({ id: claim.code, textIndex, excerpt: match[0], expected: expected.text });
        falseClaims.push({ id: claim.code, reason: 'record-binding-mismatch' });
      } else if (matched.has(claim.code)) duplicateCount++;
      else matched.add(claim.code);
    }
    for (const mention of text.matchAll(/\bH\d{3}\b/gu)) {
      if (!spans.some((span) => mention.index >= span.index && mention.index < span.index + span[0].length)) {
        const diagnostic = { id: mention[0], textIndex, reason: 'unparsed-record-mention' };
        formatErrors.push(diagnostic);
        falseClaims.push(diagnostic);
      }
    }
  }
  const missed = records.filter(({ code }) => !matched.has(code)).map(({ code }) => code);
  return {
    metrics: {
      expectedFacts: records.length,
      truePositiveClaims: matched.size,
      falsePositiveClaims: falseClaims.length,
      missedExpectedFacts: missed.length,
      claimPrecision: matched.size + falseClaims.length ? matched.size / (matched.size + falseClaims.length) : null,
      factRecall: records.length ? matched.size / records.length : null,
    },
    trueClaims: [...matched], missed, falseClaims, contradictions, unsupported,
    duplicateCount,
    formatCompliance: { compliant: formatErrors.length === 0, errors: formatErrors },
  };
}

/** Citation IDs never supply any part of a record's semantic signature. */
export function evaluateHierarchy(report, records) {
  const sourceFacts = Array.isArray(report?.sourceFacts) ? report.sourceFacts : [];
  const retainedTexts = [];
  const integrityErrors = [];
  for (const [index, fact] of sourceFacts.entries()) {
    const citation = fact?.citation;
    const source = records.find(({ sourceId }) => sourceId === citation?.paragraphId);
    if (citation?.kind !== 'quote' || typeof citation.quote !== 'string' || !citation.quote.length || !source
      || !Number.isSafeInteger(citation.startOffset) || !Number.isSafeInteger(citation.endOffset)
      || citation.startOffset < 0 || citation.endOffset <= citation.startOffset
      || source.text.slice(citation.startOffset, citation.endOffset) !== citation.quote) {
      integrityErrors.push({ index, reason: 'invalid-retained-quote' });
    } else retainedTexts.push(citation.quote);
  }
  const sections = Array.isArray(report?.sections) ? report.sections : [];
  const generatedTexts = [];
  const addText = (value) => { if (typeof value === 'string' && value.trim()) generatedTexts.push(value); };
  const overviewTexts = [report?.page?.summary, report?.conclusion].filter((value) => typeof value === 'string' && value.trim());
  for (const section of sections) {
    for (const point of Array.isArray(section?.keyPoints) ? section.keyPoints : []) addText(point);
  }
  const retained = scoreRecords(retainedTexts, records);
  retained.formatCompliance.errors.push(...integrityErrors);
  retained.formatCompliance.compliant = retained.formatCompliance.errors.length === 0;
  const generated = scoreRecords(generatedTexts, records);
  return {
    evaluatorVersion: EVALUATOR_VERSION,
    method: 'Separate exact source-quote coverage, generated section record-tuple recall, and overview/conclusion tuple safety findings. Code, station, site, field, value, unit and date must match together. Citation IDs alone receive no semantic credit. Non-record prose is not semantically scored.',
    retained,
    generated,
    overview: scoreRecords(overviewTexts, records),
    generatedTextCount: generatedTexts.length,
    reportVersionValid: report?.schemaVersion === 3,
  };
}
