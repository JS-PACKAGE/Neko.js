import type { Page, StructuredReport } from '../types.js';
import { ERROR_CODES, ERROR_STAGES } from '../errors.js';
import { validateGeneratedLanguage } from './language.js';
import { snapshotPage, validatePage } from '../web/source.js';
import { getRegisteredModelProfile, type ModelId } from '../cache/registry.js';
import type { PageSnapshot } from '../types.js';
import { reportChecksum, sourceCoverage, validateSourceFacts } from './evidence.js';
import { auditClaim } from './audit.js';
import { validateImageObservation } from '../web/image.js';

function object(value: unknown, name: string): asserts value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new TypeError(`${name} must be an object`);
}
function text(value: unknown, name: string): asserts value is string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be a non-empty string`);
}
function texts(value: unknown, name: string, allowEmpty = false): asserts value is string[] {
  if (!Array.isArray(value) || (!allowEmpty && !value.length)) throw new TypeError(`${name} must be a ${allowEmpty ? '' : 'non-empty '}string array`);
  for (const item of value) text(item, name);
}

function validateReportBody(report: unknown, page: Page): asserts report is StructuredReport {
  object(report, 'Report');
  if (report.schemaVersion !== 3) throw new TypeError('Unsupported StructuredReport schemaVersion; expected 3');
  if (report.mode !== 'generated' && report.mode !== 'extractive') throw new TypeError('Report mode is invalid');
  text(report.language, 'Report language');
  try { Intl.getCanonicalLocales(report.language); } catch { throw new TypeError('Report language must be a valid BCP 47 language tag'); }
  if (report.imageFailurePolicy !== 'error' && report.imageFailurePolicy !== 'omit') throw new TypeError('Report imageFailurePolicy is invalid');
  object(report.page, 'Report page');
  if (typeof report.page.url !== 'string') throw new TypeError('Report URL must be a string');
  text(report.page.summary, 'Report summary');
  text(report.conclusion, 'Report conclusion');
  if (report.mode === 'generated') {
    validateGeneratedLanguage(report.page.summary, report.language, 'Summary');
    validateGeneratedLanguage(report.conclusion, report.language, 'Conclusion');
  }
  if (report.page.url !== page.url) throw new TypeError('Report URL does not match the extracted page');
  if (report.page.title !== page.title) throw new TypeError('Report title does not match the extracted page');
  if (!Array.isArray(report.sections)) throw new TypeError('Report sections must be an array');
  const paragraphIds = new Set(page.paragraphs.map(({ id }) => id));
  const covered = new Set<string>();
  for (const section of report.sections) {
    object(section, 'Report section');
    if (section.heading !== undefined) text(section.heading, 'Section heading');
    texts(section.keyPoints, 'Section key points');
    texts(section.paragraphIds, 'Section paragraph IDs');
    if (report.mode === 'generated') for (const point of section.keyPoints) validateGeneratedLanguage(point, report.language, 'Section key point');
    if (new Set(section.paragraphIds).size !== section.paragraphIds.length) throw new TypeError('Section has duplicate paragraph references');
    for (const id of section.paragraphIds) {
      if (!paragraphIds.has(id)) throw new TypeError(`Report references unknown paragraph ${id}`);
      covered.add(id);
    }
  }
  if (covered.size !== paragraphIds.size) throw new TypeError('Report must cover every extracted paragraph');
  if (!Array.isArray(report.images) || report.images.length !== page.images.length) throw new TypeError('Report must describe every extracted image exactly once');
  const expected = new Map(page.images.map((image) => [image.id, image]));
  const seen = new Set<string>();
  for (const image of report.images) {
    object(image, 'Report image');
    text(image.imageId, 'Image ID');
    const source = expected.get(image.imageId);
    if (!source) throw new TypeError(`Report references unknown image ${image.imageId}`);
    if (seen.has(image.imageId)) throw new TypeError(`Report describes image ${image.imageId} more than once`);
    object(image.source, 'Image provenance');
    if (image.url !== source.url || image.source.kind !== 'image' || image.source.imageId !== image.imageId) throw new TypeError(`Image ${image.imageId} has inconsistent provenance`);
    if (image.alt !== source.alt) throw new TypeError(`Image ${image.imageId} alt text does not match extracted metadata`);
    if (image.status === 'retained' && report.mode === 'extractive') {
      if (Object.hasOwn(image, 'description') || Object.hasOwn(image, 'observation') || Object.hasOwn(image, 'error')) throw new TypeError('Retained image metadata cannot include generated output');
    } else if (image.status === 'described' && report.mode === 'generated') {
      if (typeof image.description !== 'string' || !image.description.trim()) throw new TypeError(`Image ${image.imageId} is missing its generated description`);
      validateGeneratedLanguage(image.description, report.language, `Image ${image.imageId} description`);
      if (image.error !== undefined) throw new TypeError('Described image cannot include an error');
    } else if (image.status === 'failed' && report.mode === 'generated' && report.imageFailurePolicy === 'omit') {
      object(image.error, 'Image failure');
      text(image.error.stage, 'Image error stage'); text(image.error.code, 'Image error code'); text(image.error.message, 'Image error message');
      if (!Object.hasOwn(ERROR_STAGES, image.error.stage) || !Object.hasOwn(ERROR_CODES, image.error.code)) throw new TypeError('Image error stage or code is invalid');
      if (image.description !== undefined) throw new TypeError('Failed image cannot claim a generated description');
    } else throw new TypeError('Report image status conflicts with its failure policy');
    seen.add(image.imageId);
  }
}

function digest(value: unknown, name: string): asserts value is string {
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value)) throw new TypeError(`${name} must be a SHA256 digest`);
}
function sourceEqual(actual: Page, expected: Page): boolean {
  if (actual.url !== expected.url || actual.title !== expected.title || actual.paragraphs.length !== expected.paragraphs.length || actual.images.length !== expected.images.length) return false;
  for (const [index, paragraph] of actual.paragraphs.entries()) {
    const source = expected.paragraphs[index]!;
    if (paragraph.id !== source.id || paragraph.text !== source.text || paragraph.heading !== source.heading || paragraph.source.kind !== source.source.kind || paragraph.source.startOffset !== source.source.startOffset || paragraph.source.endOffset !== source.source.endOffset) return false;
  }
  for (const [index, image] of actual.images.entries()) {
    const source = expected.images[index]!;
    if (image.id !== source.id || image.url !== source.url || image.alt !== source.alt || image.caption !== source.caption || image.sourceElement !== source.sourceElement || image.discoveredBy.length !== source.discoveredBy.length || image.discoveredBy.some((kind, offset) => kind !== source.discoveredBy[offset])) return false;
  }
  return true;
}
function validateEvidence(report: StructuredReport, expected: PageSnapshot): void {
  object(report.snapshot, 'Report snapshot'); validatePage(report.snapshot.source);
  if (report.snapshot.id !== expected.id || report.snapshot.algorithm !== 'sha256' || !sourceEqual(report.snapshot.source, expected.source)) throw new TypeError('Report snapshot does not match its persisted source');
  for (const kind of ['paragraphs', 'images'] as const) {
    const entries = report.snapshot[kind]; const sources = expected[kind];
    if (!Array.isArray(entries) || entries.length !== sources.length) throw new TypeError(`Snapshot ${kind} coverage is invalid`);
    for (const [index, entry] of entries.entries()) {
      object(entry, 'Snapshot source version'); const version = sources[index]!;
      if (entry.id !== version.id || ('versionId' in version ? !('versionId' in entry) || entry.versionId !== version.versionId : !('metadataVersionId' in entry) || entry.metadataVersionId !== version.metadataVersionId)) throw new TypeError('Snapshot source version does not match persisted content');
    }
  }
  const paragraphs = new Map(expected.source.paragraphs.map((paragraph) => [paragraph.id, paragraph]));
  const versions = new Map(expected.paragraphs.map((paragraph) => [paragraph.id, paragraph.versionId]));
  validateSourceFacts(report.sourceFacts, expected);
  const retainedQuotes = new Set(report.sourceFacts.map(({ citation }) => `${citation.paragraphId}:${citation.startOffset}:${citation.endOffset}`));
  const images = new Map(report.images.map((image) => [image.imageId, image]));
  const targets = new Map<string, string>([['page.summary', report.page.summary], ['conclusion', report.conclusion]]);
  for (const [index, section] of report.sections.entries()) for (const [point, value] of section.keyPoints.entries()) targets.set(`sections[${index}].keyPoints[${point}]`, value);
  for (const [index, image] of report.images.entries()) if (image.status === 'described') {
    validateImageObservation(image.observation);
    if (image.observation.verification !== 'model-observation') throw new TypeError('Image observation is invalid');
    targets.set(`images[${index}].description`, image.description);
  }
  if (!Array.isArray(report.claims) || !report.claims.length) throw new TypeError('Report must contain atomic cited claims');
  const ids = new Set<string>(); const spans = new Map<string, { start: number; end: number }[]>();
  for (const claim of report.claims) {
    object(claim, 'Report claim'); text(claim.id, 'Claim ID'); text(claim.target, 'Claim target');
    const target = targets.get(claim.target);
    if (ids.has(claim.id) || claim.verification !== 'references-validated' || target === undefined || !Number.isSafeInteger(claim.startOffset) || !Number.isSafeInteger(claim.endOffset) || claim.startOffset < 0 || claim.endOffset <= claim.startOffset || claim.endOffset > target.length || !target.slice(claim.startOffset, claim.endOffset).trim()) throw new TypeError('Atomic claim target or span is invalid');
    ids.add(claim.id);
    const ranges = spans.get(claim.target) ?? []; ranges.push({ start: claim.startOffset, end: claim.endOffset }); spans.set(claim.target, ranges);
    if (!Array.isArray(claim.citations) || !claim.citations.length || claim.citations.length > 8) throw new TypeError('Atomic claim needs explicit citations');
    const references = new Set<string>();
    for (const citation of claim.citations) {
      object(citation, 'Claim citation');
      if (citation.snapshotId !== expected.id) throw new TypeError('Citation references a different source snapshot');
      let reference: string;
      if (citation.kind === 'quote') {
        const paragraph = paragraphs.get(citation.paragraphId);
        if (!paragraph || citation.versionId !== versions.get(citation.paragraphId) || !Number.isSafeInteger(citation.startOffset) || !Number.isSafeInteger(citation.endOffset) || citation.startOffset < 0 || citation.endOffset <= citation.startOffset || citation.endOffset > paragraph.text.length || citation.quote !== paragraph.text.slice(citation.startOffset, citation.endOffset)) throw new TypeError('Citation quote, offsets, or paragraph version is invalid');
        if (!retainedQuotes.has(`${citation.paragraphId}:${citation.startOffset}:${citation.endOffset}`)) throw new TypeError('Claim quote is absent from the retained source fact ledger');
        const sectionTarget = /^sections\[(\d+)\]\.keyPoints\[\d+\]$/.exec(claim.target);
        if (sectionTarget && !report.sections[Number(sectionTarget[1])]!.paragraphIds.includes(citation.paragraphId)) throw new TypeError('Section claim cites a paragraph outside its planned source');
        reference = `${citation.paragraphId}:${citation.startOffset}:${citation.endOffset}`;
      } else if (citation.kind === 'image-observation') {
        const image = images.get(citation.imageId);
        if (!image || image.status !== 'described' || citation.versionId !== image.observation.versionId || citation.sourceVersionId !== image.observation.sourceVersionId || citation.sourceWidth !== image.observation.sourceWidth || citation.sourceHeight !== image.observation.sourceHeight || JSON.stringify(citation.region) !== JSON.stringify(image.observation.region) || JSON.stringify(citation.normalizedRegion) !== JSON.stringify(image.observation.normalizedRegion)) throw new TypeError('Citation references an unavailable or different image observation or region');
        reference = `image:${citation.imageId}`;
      } else throw new TypeError('Unsupported citation kind');
      if (references.has(reference)) throw new TypeError('Claim has duplicate citations'); references.add(reference);
    }
    const claimText = target.slice(claim.startOffset, claim.endOffset);
    const expectedAudit = auditClaim(claimText, claim.citations, expected);
    if (JSON.stringify(claim.audit) !== JSON.stringify(expectedAudit)) throw new TypeError('Claim audit is missing, altered, or inconsistent with its exact cited evidence');
    if (report.mode === 'extractive' && (expectedAudit.status !== 'supported' || !claim.citations.every((citation) => citation.kind === 'quote') || !claim.citations.some((citation) => citation.kind === 'quote' && citation.quote === claimText))) throw new TypeError('Extractive output must be exact cited source text');
  }
  for (const [target, text] of targets) {
    const ranges = spans.get(target); if (!ranges) throw new TypeError(`Generated text has no atomic claim coverage (${target})`);
    ranges.sort((a, b) => a.start - b.start); let end = 0;
    for (const range of ranges) { if (range.start < end || text.slice(end, range.start).trim()) throw new TypeError('Atomic claim ranges overlap or leave unsupported generated text'); end = range.end; }
    if (text.slice(end).trim()) throw new TypeError('Generated text is not fully covered by cited claims');
  }
  object(report.metadata, 'Report metadata'); object(report.metadata.model, 'Report model');
  const profile = getRegisteredModelProfile(report.metadata.model.profile, report.metadata.model.id as ModelId);
  if (report.metadata.model.id !== profile.id || report.metadata.model.revision !== profile.revision) throw new TypeError('Report model identity is not a registered pinned profile');
  object(report.metadata.model.dtype, 'Model dtype');
  for (const key of ['embed_tokens', 'decoder_model_merged', 'vision_encoder'] as const) if (report.metadata.model.dtype[key] !== profile.dtype[key]) throw new TypeError('Report model dtype does not match its profile');
  object(report.metadata.usage, 'Report usage');
  const usage = report.metadata.usage;
  if (!Number.isSafeInteger(usage.inputTokens) || !Number.isSafeInteger(usage.outputTokens) || !Number.isSafeInteger(usage.totalTokens) || usage.inputTokens < 0 || usage.outputTokens < 0 || usage.totalTokens !== usage.inputTokens + usage.outputTokens) throw new TypeError('Report token accounting is invalid');
  object(report.metadata.timings, 'Report timings');
  for (const key of ['loadMs', 'preprocessMs', 'generationMs', 'totalMs', 'queueWaitMs'] as const) if (typeof report.metadata.timings[key] !== 'number' || !Number.isFinite(report.metadata.timings[key]) || report.metadata.timings[key] < 0) throw new TypeError('Report timing is invalid');
  object(report.metadata.memory, 'Report memory');
  if (report.metadata.memory.jsHeapBytes !== null || report.metadata.memory.gpuBytes !== null) throw new TypeError('Report memory must mark unmeasured values as unknown');
  if (report.mode === 'generated') {
  object(report.metadata.backend, 'Report backend');
  if (!['node', 'browser'].includes(report.metadata.backend.runtime) || !['cpu', 'webgpu'].includes(report.metadata.backend.device) || report.metadata.backend.providerEvidence !== 'loaded-session-configuration' || !Array.isArray(report.metadata.backend.sessions) || !report.metadata.backend.sessions.length) throw new TypeError('Report backend evidence is invalid');
  const sessions = new Set<string>();
  for (const session of report.metadata.backend.sessions) {
    object(session, 'Loaded session');
    const name = session.name;
    if (typeof name !== 'string' || !Object.hasOwn(profile.dtype, name) || sessions.has(name) || session.device !== report.metadata.backend.device || session.dtype !== profile.dtype[name as keyof typeof profile.dtype]) throw new TypeError('Loaded session configuration does not match the pinned model');
    sessions.add(name);
  }
  if (sessions.size !== Object.keys(profile.dtype).length) throw new TypeError('Loaded session configuration is incomplete');
  } else if (report.metadata.backend !== null) throw new TypeError('Extractive reports cannot claim a loaded backend');
  object(report.metadata.execution, 'Report execution');
  if (!['inline', 'worker'].includes(report.metadata.execution.mode) || !['node', 'browser'].includes(report.metadata.execution.runtime) || report.metadata.backend !== null && report.metadata.execution.runtime !== report.metadata.backend.runtime || report.metadata.execution.mode === 'worker' && typeof report.metadata.execution.workerId !== 'string' || !Number.isSafeInteger(report.metadata.resumedStages) || report.metadata.resumedStages < 0 || !Number.isSafeInteger(report.metadata.retryAttempts) || report.metadata.retryAttempts < 0 || report.metadata.evidence !== 'references-validated-heuristic-audit') throw new TypeError('Report execution or evidence metadata is invalid');
  if (report.mode === 'extractive' && (usage.totalTokens !== 0 || report.metadata.retryAttempts !== 0 || report.metadata.timings.loadMs !== 0 || report.metadata.timings.generationMs !== 0)) throw new TypeError('Extractive reports cannot claim model usage, load/generation timings or retries');
  object(report.metadata.coverage, 'Report coverage');
  const coverage = report.metadata.coverage;
  if (coverage.conclusionBasis !== 'retained-source' && coverage.conclusionBasis !== 'reduced-generated-claims') throw new TypeError('Conclusion source basis is invalid');
  const expectedCoverage = sourceCoverage(expected, report.sourceFacts, report.claims, coverage.conclusionBasis, report.mode);
  for (const key of ['selectedParagraphIds', 'modelCitedFactIds', 'summaryCitedFactIds'] as const) if (!Array.isArray(coverage[key]) || coverage[key].length !== expectedCoverage[key].length || coverage[key].some((id, index) => id !== expectedCoverage[key][index])) throw new TypeError(`Report ${key} coverage is inconsistent`);
  for (const key of ['selectedTextCharacters', 'retainedQuoteCount', 'retainedTextCharacters', 'semanticRetention'] as const) if (coverage[key] !== expectedCoverage[key]) throw new TypeError(`Report ${key} coverage is inconsistent`);
}

/** Validates hashes, exact references and reproducible limited lexical audits, never real-world truth. */
export async function validateStructuredReport(report: unknown, page?: Page): Promise<StructuredReport> {
  object(report, 'Report');
  if (report.schemaVersion !== 3) throw new TypeError('Unsupported StructuredReport schemaVersion; expected 3');
  object(report.snapshot, 'Report snapshot'); validatePage(report.snapshot.source);
  const [expected, external] = await Promise.all([snapshotPage(report.snapshot.source), page === undefined || page === report.snapshot.source ? undefined : snapshotPage(page)]);
  if (external && external.id !== expected.id) throw new TypeError('Report source differs from the supplied selected Page');
  validateReportBody(report, expected.source); validateEvidence(report, expected);
  object(report.integrity, 'Report integrity');
  digest(report.integrity.checksum, 'Report checksum');
  if (report.integrity.algorithm !== 'sha256' || report.integrity.checksum !== await reportChecksum(report)) throw new TypeError('Report checksum is invalid; checksums are not authentication');
  return report;
}

export async function serializeStructuredReport(report: unknown, page?: Page): Promise<string> {
  return JSON.stringify(await validateStructuredReport(report, page));
}

export async function parseStructuredReport(json: string, page?: Page): Promise<StructuredReport> {
  if (typeof json !== 'string') throw new TypeError('Saved report must be JSON text');
  return validateStructuredReport(JSON.parse(json) as unknown, page);
}
