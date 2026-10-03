import assert from 'node:assert/strict';
import test from 'node:test';
import { auditClaim, type AuditKind } from '../../src/report/audit.js';
import type { Citation } from '../../src/types.js';

const quote = (text: string): Citation => ({ kind: 'quote', snapshotId: 'fixture', paragraphId: 'p1', versionId: 'fixture', startOffset: 0, endOffset: text.length, quote: text });

test('audits expose exact retention, aligned lexical conflicts and explicit uncertainty without confidence', () => {
  const cases: { source: string; claim: string; kind: AuditKind }[] = [
    { source: 'Admission costs 7 dollars.', claim: 'Admission costs 8 dollars.', kind: 'numbers' },
    { source: 'The route spans 4 km.', claim: 'The route spans 4 m.', kind: 'units' },
    { source: 'Opening is 2026-10-03.', claim: 'Opening is 2026-10-04.', kind: 'dates' },
    { source: 'The office is in Paris.', claim: 'The office is in Berlin.', kind: 'entities' },
    { source: 'Pets are not allowed.', claim: 'Pets are allowed.', kind: 'negation' },
    { source: 'The range is 2-4 km.', claim: 'The range is 3-5 km.', kind: 'ranges' },
  ];
  for (const { source, claim, kind } of cases) {
    const exact = auditClaim(source, [quote(source)]); assert.equal(exact.status, 'supported');
    const changed = auditClaim(claim, [quote(source)]); assert.equal(changed.status, 'contradicted');
    assert.equal(changed.checks.find((check) => check.kind === kind)!.status, 'conflict');
    assert.ok(changed.limits.length > 0); assert.equal(Object.hasOwn(changed, 'confidence'), false);
  }
  assert.equal(auditClaim('Entry is inexpensive.', [quote('Admission costs 7 dollars.')]).status, 'unknown');
  assert.equal(auditClaim('入場費用為 7 美元。', [quote('Admission costs 7 dollars.')]).status, 'unknown');
  assert.equal(auditClaim('A red square.', []).status, 'unknown');
  assert.equal(auditClaim('Admission costs 8 dollars.', [quote('Admission costs 7 dollars. Admission costs 8 dollars.')]).status, 'supported');
});
