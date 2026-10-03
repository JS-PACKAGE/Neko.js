import type { ModelIdentity } from '../core/engine.js';
import type { ExecutionInfo, Page, PageSnapshot, ReportClaim, ReportSourceFact, StructuredReport } from '../types.js';
import { atStage, awaitUser, NekoError } from '../errors.js';
import { hashValue, selectPage, snapshotPage } from '../web/source.js';
import { auditClaim } from './audit.js';
import { reportChecksum, sourceCoverage, validateSourceFacts } from './evidence.js';
import { renderMarkdown } from './markdown.js';
import { validateStructuredReport } from './validate.js';
import { ReportError, validateDescribeOptions, validateReportCheckpoint, type DescribeOptions, type InternalReportContext, type ReportCheckpoint, type ReportPlan, type ReportPhase } from './generate.js';

export interface ExtractiveReportContext extends InternalReportContext { model: ModelIdentity; execution: ExecutionInfo; }

/** Retains whole paragraphs exactly; there is no tokenizer, model acquisition or generated translation. */
function retainedSource(snapshot: PageSnapshot): ReportSourceFact[] {
  if (!snapshot.source.paragraphs.length) throw new NekoError('Extractive reports require selected visible source text', 'report', 'INVALID_INPUT');
  const facts: ReportSourceFact[] = snapshot.source.paragraphs.map((paragraph, index) => ({ id: `q${index + 1}`, citation: { kind: 'quote', snapshotId: snapshot.id, paragraphId: paragraph.id, versionId: snapshot.paragraphs[index]!.versionId, startOffset: 0, endOffset: paragraph.text.length, quote: paragraph.text } }));
  validateSourceFacts(facts, snapshot); return facts;
}

export async function planExtractiveReport(input: Page, options: DescribeOptions = {}, internal: InternalReportContext = {}): Promise<ReportPlan> {
  validateDescribeOptions(options);
  const page = internal.selectionApplied ? input : await selectPage(input, options.sources, options.signal);
  const snapshot = await snapshotPage(page); const facts = retainedSource(snapshot);
  options.signal?.throwIfAborted();
  return { mode: 'extractive', snapshotId: snapshot.id, sectionCount: facts.length, imageCount: page.images.length, stages: [], knownInputTokens: 0, maxKnownOutputTokens: 0, reduction: { required: false, stageCount: null }, estimatedDurationMs: null, totalTokensUpperBound: null };
}

export async function generateExtractiveReport(input: Page, options: DescribeOptions, internal: ExtractiveReportContext): Promise<StructuredReport | string> {
  const started = internal.startedAt ?? performance.now(); const signal = options.signal;
  return atStage('report', signal, async () => {
    validateDescribeOptions(options);
    const page = internal.selectionApplied ? input : await selectPage(input, options.sources, signal);
    const snapshot = await snapshotPage(page); const sourceFacts = retainedSource(snapshot);
    const language = options.sourceLanguage ?? 'und'; const imageFailurePolicy = options.imageFailurePolicy ?? 'error';
    const authorization = { maxTotalTokens: options.budget?.maxTotalTokens ?? Number.MAX_SAFE_INTEGER, maxDurationMs: options.budget?.maxDurationMs ?? null, retries: { ...options.retries } };
    const identity = await hashValue({ version: 3, plan: 'evidence-first-v3', mode: 'extractive', snapshot: snapshot.id, model: internal.model, language, imageFailurePolicy });
    let checkpoint: ReportCheckpoint = { version: 3, plan: 'evidence-first-v3', mode: 'extractive', identity, checksum: '', snapshot, sourceFacts, sectionPlan: sourceFacts.map(({ id }) => [id]), model: internal.model, authorization, imageObservations: {}, completed: {}, attempts: [], usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 }, elapsedMs: 0, timings: { loadMs: 0, preprocessMs: 0, generationMs: 0, queueWaitMs: internal.queueWaitMs ?? 0 } };
    if (options.resume) {
      const saved = structuredClone(await validateReportCheckpoint(options.resume));
      if (saved.identity !== identity || await hashValue(saved.sourceFacts) !== await hashValue(sourceFacts) || saved.usage.totalTokens !== 0) throw new NekoError('Extractive checkpoint does not match selected source and identity', 'report', 'CHECKPOINT_INVALID');
      checkpoint = saved; checkpoint.authorization = authorization; checkpoint.timings.queueWaitMs += internal.queueWaitMs ?? 0;
    }
    const elapsedBefore = checkpoint.elapsedMs; const claims: ReportClaim[] = []; let resumedStages = 0; let callbackFailed = false;
    const used = new Set<string>();
    const save = async (notify = true) => {
      checkpoint.elapsedMs = elapsedBefore + performance.now() - started;
      checkpoint.checksum = await hashValue({ ...checkpoint, checksum: undefined });
      if (notify && options.onCheckpoint) try { await awaitUser(() => options.onCheckpoint!(structuredClone(checkpoint)), signal); } catch (cause) { callbackFailed = true; throw cause; }
    };
    const retain = async (id: string, phase: ReportPhase, fact: ReportSourceFact, target: string) => {
      signal?.throwIfAborted();
      if (authorization.maxDurationMs !== null && elapsedBefore + performance.now() - started >= authorization.maxDurationMs) throw new NekoError('Report aggregate duration budget exhausted', 'report', 'BUDGET_EXCEEDED');
      used.add(id); const items = [{ text: fact.citation.quote, evidenceIds: [fact.id] }];
      const requestHash = await hashValue({ id, phase, items, mode: 'extractive' }); const saved = checkpoint.completed[id];
      if (saved) {
        if (saved.requestHash !== requestHash || await hashValue(saved.claims) !== await hashValue(items)) throw new NekoError('Extractive stage differs from retained source', 'report', 'CHECKPOINT_INVALID');
        resumedStages++;
      } else {
        const timings = { loadMs: 0, preprocessMs: 0, generationMs: 0, firstTokenMs: null, totalMs: 0 };
        checkpoint.completed[id] = { phase, requestHash, claims: items, usage: { ...checkpoint.usage }, timings };
        checkpoint.attempts.push({ id, phase, attempt: 1, requestHash, outcome: 'complete', errorCode: null, usage: { ...checkpoint.usage }, timings });
      }
      claims.push({ id: `claim-${claims.length + 1}`, target, startOffset: 0, endOffset: fact.citation.quote.length, citations: [{ ...fact.citation }], verification: 'references-validated', audit: auditClaim(fact.citation.quote, [fact.citation], snapshot) });
      await save();
      if (options.onEvent) try { await awaitUser(() => options.onEvent!({ type: saved ? 'stage-reused' : 'stage-complete', id, phase, usage: { ...checkpoint.usage }, elapsedMs: checkpoint.elapsedMs }), signal); } catch (cause) { callbackFailed = true; throw cause; }
      return fact.citation.quote;
    };
    try {
      const sections: StructuredReport['sections'] = [];
      for (const [index, fact] of sourceFacts.entries()) sections.push({ ...(page.paragraphs[index]!.heading ? { heading: page.paragraphs[index]!.heading } : {}), keyPoints: [await retain(`section:${index}`, 'section', fact, `sections[${index}].keyPoints[0]`)], paragraphIds: [fact.citation.paragraphId] });
      const summary = await retain('summary', 'summary', sourceFacts[0]!, 'page.summary');
      const conclusion = await retain('conclusion', 'conclusion', sourceFacts[sourceFacts.length - 1]!, 'conclusion');
      if (Object.keys(checkpoint.completed).some((id) => !used.has(id))) throw new NekoError('Extractive checkpoint contains unplanned stages', 'report', 'CHECKPOINT_INVALID');
      await save();
      const report: StructuredReport = { schemaVersion: 3, mode: 'extractive', integrity: { algorithm: 'sha256', checksum: '' }, sourceFacts, language, imageFailurePolicy, page: { url: page.url, ...(page.title === undefined ? {} : { title: page.title }), summary }, sections, images: page.images.map(({ id, url, alt }) => ({ status: 'retained', imageId: id, url, ...(alt === undefined ? {} : { alt }), source: { kind: 'image', imageId: id } })), conclusion, snapshot, claims, metadata: { model: internal.model, backend: null, usage: { ...checkpoint.usage }, timings: { ...checkpoint.timings, totalMs: checkpoint.elapsedMs }, memory: { jsHeapBytes: null, gpuBytes: null }, execution: internal.execution, resumedStages, retryAttempts: 0, evidence: 'references-validated-heuristic-audit', coverage: sourceCoverage(snapshot, sourceFacts, claims, 'retained-source', 'extractive') } };
      report.integrity.checksum = await reportChecksum(report); await validateStructuredReport(report, page); signal?.throwIfAborted();
      return options.format === 'markdown' ? renderMarkdown(report) : report;
    } catch (cause) {
      await save(!callbackFailed && !signal?.aborted);
      const error = cause instanceof NekoError ? cause : signal?.aborted && !callbackFailed ? new NekoError('Report was cancelled', 'report', 'ABORTED', { cause: signal.reason }) : new NekoError(cause instanceof Error ? cause.message : String(cause), 'report', 'OPERATION_FAILED', { cause });
      throw new ReportError(error, structuredClone(checkpoint));
    }
  });
}
