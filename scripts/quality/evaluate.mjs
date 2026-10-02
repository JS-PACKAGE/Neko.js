const claimLine = /^\s*(?:(FACT)\s+)?([A-Z0-9-]+)\s*:\s*(.*?)\s*$/iu;

export const EVALUATOR_VERSION = 'quality-claims-v3';

function matchAll(pattern, text) {
  return [...text.matchAll(new RegExp(pattern, 'giu'))];
}

function isNegatedMention(text, index) {
  const before = text.slice(0, index);
  const clauseStart = Math.max(before.lastIndexOf('.'), before.lastIndexOf('!'), before.lastIndexOf('?'), before.lastIndexOf(';')) + 1;
  const suffix = before.slice(clauseStart);
  return /\b(?:no|not|never|without|neither|zero)\b(?:\s+[\p{L}-]+){0,3}\s*$/iu.test(suffix);
}

function findings(text, rules) {
  const matches = [];
  for (const rule of rules ?? []) {
    for (const match of matchAll(rule.pattern, text)) {
      if (!isNegatedMention(text, match.index)) matches.push({ id: rule.id, excerpt: match[0] });
    }
  }
  return matches;
}

function matchesFact(text, fact) {
  const signature = fact.pattern
    ? new RegExp(fact.pattern, 'iu').test(text)
    : Array.isArray(fact.match) && fact.match.every((pattern) => new RegExp(pattern, 'iu').test(text));
  if (!signature) return false;
  return fact.allowNegation === true || !(fact.negatedBy ?? []).some((pattern) => new RegExp(pattern, 'iu').test(text));
}

/** Scores explicit-ID lines against a closed-set lexical oracle; format compliance is reported separately. */
export function evaluateClaims(output, oracle) {
  const facts = oracle.facts ?? [];
  const byId = new Map(facts.map((fact) => [fact.id, fact]));
  const rejectForeignMatches = oracle.rejectForeignMatches === true;
  const seen = new Set();
  const trueClaims = [];
  const falseClaims = [];
  const contradictions = [];
  const unsupported = [];
  const formatErrors = [];
  let nonemptyLines = 0;
  let parsedLines = 0;

  for (const [index, rawLine] of String(output).split(/\r?\n/u).entries()) {
    const line = rawLine.trim();
    if (!line) continue;
    nonemptyLines++;
    const parsed = claimLine.exec(line);
    if (!parsed) {
      formatErrors.push({ lineNumber: index + 1, reason: 'unparsed-line', line });
      continue;
    }
    parsedLines++;
    const [, prefix, id, statement] = parsed;
    if (!prefix) formatErrors.push({ lineNumber: index + 1, reason: 'missing-fact-prefix', line });
    const fact = byId.get(id);
    const duplicate = seen.has(id);
    seen.add(id);
    const lineContradictions = findings(statement, oracle.contradictions);
    const lineUnsupported = findings(statement, oracle.unsupported);
    const foreignFacts = rejectForeignMatches
      ? facts.filter((other) => other.id !== id && matchesFact(statement, other)).map(({ id: foreignId }) => foreignId)
      : [];
    contradictions.push(...lineContradictions.map((finding) => ({ claimId: id, ...finding })));
    unsupported.push(...lineUnsupported.map((finding) => ({ claimId: id, ...finding })));
    const expectedMatch = fact ? matchesFact(statement, fact) : false;
    const crossIdMismatch = !expectedMatch && foreignFacts.length > 0;
    if (fact && expectedMatch && !duplicate && !lineContradictions.length && !lineUnsupported.length) {
      trueClaims.push(id);
    } else {
      falseClaims.push({
        id,
        reason: !fact ? 'unknown-id' : duplicate ? 'duplicate-id' : lineContradictions.length ? 'contradiction' : lineUnsupported.length ? 'unsupported-pattern' : crossIdMismatch ? 'cross-id-fact' : 'oracle-miss',
        ...(crossIdMismatch ? { foreignFacts } : {}),
        line,
      });
    }
  }

  const missed = facts.map(({ id }) => id).filter((id) => !trueClaims.includes(id));
  const tp = trueClaims.length;
  const fp = falseClaims.length;
  const fn = missed.length;
  return {
    evaluatorVersion: EVALUATOR_VERSION,
    method: 'closed-set explicit-ID lines; FACT prefix is optional for factual scoring and reported separately for format compliance; no free-form semantic claims are inferred',
    metrics: {
      claimPrecision: tp + fp ? tp / (tp + fp) : null,
      factRecall: tp + fn ? tp / (tp + fn) : null,
      truePositiveClaims: tp,
      falsePositiveClaims: fp,
      missedExpectedFacts: fn,
      expectedFacts: facts.length,
    },
    formatCompliance: {
      compliant: formatErrors.length === 0,
      nonemptyLines,
      parsedClaimLines: parsedLines,
      errors: formatErrors,
    },
    trueClaims,
    missed,
    falseClaims,
    contradictions,
    unsupported,
  };
}

/** Explicit heuristic negative-scope check shared with unsupported/contradiction diagnostics. */
export function detectUnsupported(text, rules) {
  return findings(String(text), rules);
}

export function isNegated(text, token) {
  const value = String(text);
  const index = value.indexOf(token);
  return index >= 0 && isNegatedMention(value, index);
}
