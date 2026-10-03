import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { parseHierarchyRecords } from './hierarchy.mjs';

export const ARTIFACT_VERSION = 3;
export const FIXTURE_VERSION = 'quality-fixtures-v2';
export const CASE_IDS = ['text', 'image', 'boundaries', 'hierarchy'];
export const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const fixtureRoot = new URL('./fixtures/', import.meta.url);
const sourceUrl = (path) => new URL(path.replace('scripts/quality/fixtures/', ''), fixtureRoot);

export async function loadQualityContract(config = {}) {
  const oracleBytes = await readFile(new URL('oracle.json', fixtureRoot));
  const manifest = JSON.parse(oracleBytes);
  if (manifest.schemaVersion !== 2 || manifest.fixtureVersion !== FIXTURE_VERSION) throw new TypeError('Unsupported quality fixture manifest version');
  const selectedCase = config.selectedCase ?? 'all';
  if (selectedCase !== 'all' && !CASE_IDS.includes(selectedCase)) throw new TypeError('Unknown quality case');
  const tokenOverride = config.maxNewTokens ?? undefined;
  const contextOverride = config.contextWindowTokens ?? undefined;
  if (tokenOverride !== undefined && (!Number.isSafeInteger(tokenOverride) || tokenOverride < 1 || tokenOverride > 2048)) throw new TypeError('Invalid output token budget');
  if (contextOverride !== undefined && (!Number.isSafeInteger(contextOverride) || contextOverride < 32)) throw new TypeError('Invalid context token budget');
  const [textSource, imageSource, imageRaster, boundarySource, hierarchySource] = await Promise.all([
    readFile(sourceUrl(manifest.text.sourcePath)), readFile(sourceUrl(manifest.image.sourcePath)),
    readFile(new URL('image-garden.png', fixtureRoot)), readFile(sourceUrl(manifest.boundaries.sourcePath)),
    readFile(sourceUrl(manifest.hierarchy.sourcePath)),
  ]);
  const hierarchyRecords = parseHierarchyRecords(hierarchySource.toString('utf8'));
  const hierarchyHtml = `<html><head><title>Field observation catalog</title></head><body><main>${hierarchyRecords.map(({ text }) => `<p>${text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')}</p>`).join('')}</main></body></html>`;
  const textPrompt = `${manifest.text.prompt}\n\nSource text:\n${textSource.toString('utf8')}`;
  const boundaryPrompt = `${manifest.boundaries.prompt}\n\nSource text:\n${boundarySource.toString('utf8')}`;
  const imageDataUrl = `data:image/png;base64,${imageRaster.toString('base64')}`;
  const fixtures = [
    { id: manifest.text.id, source: manifest.text.sourcePath, sha256: sha256(textSource), bytes: textSource.length },
    { id: manifest.image.id, source: manifest.image.sourcePath, sha256: sha256(imageSource), bytes: imageSource.length },
    { id: `${manifest.image.id}-raster`, source: 'scripts/quality/fixtures/image-garden.png', sha256: sha256(imageRaster), bytes: imageRaster.length, ...Object.fromEntries(['format', 'width', 'height'].map((key) => [key, manifest.image.raster[key]])) },
    { id: manifest.boundaries.id, source: manifest.boundaries.sourcePath, sha256: sha256(boundarySource), bytes: boundarySource.length },
    { id: manifest.hierarchy.id, source: manifest.hierarchy.sourcePath, sha256: sha256(hierarchySource), bytes: hierarchySource.length, derivedHtmlSha256: sha256(hierarchyHtml), paragraphCount: hierarchyRecords.length },
  ];
  const allCases = [
    { id: 'text', fixtureId: manifest.text.id, kind: 'infer-text', prompt: textPrompt, maxNewTokens: tokenOverride ?? 512 },
    { id: 'image', fixtureId: manifest.image.id, kind: 'infer-image', prompt: manifest.image.prompt, imageDataUrl, maxNewTokens: tokenOverride ?? 256 },
    { id: 'boundaries', fixtureId: manifest.boundaries.id, kind: 'infer-text', prompt: boundaryPrompt, maxNewTokens: tokenOverride ?? 512 },
    { id: 'hierarchy', fixtureId: manifest.hierarchy.id, kind: 'describe', html: hierarchyHtml, maxNewTokens: tokenOverride ?? manifest.hierarchy.maxNewTokens, contextWindowTokens: contextOverride ?? manifest.hierarchy.contextWindowTokens },
  ];
  const cases = allCases.filter(({ id }) => selectedCase === 'all' || id === selectedCase);
  const caseInputs = cases.map((entry) => ({
    id: entry.id, fixtureId: entry.fixtureId, kind: entry.kind,
    inputSha256: sha256(entry.kind === 'infer-image' ? entry.imageDataUrl : entry.kind === 'describe' ? entry.html : entry.prompt),
    maxNewTokens: entry.maxNewTokens, contextWindowTokens: contextOverride ?? entry.contextWindowTokens ?? null,
  }));
  const stimulusKey = sha256(JSON.stringify({ fixtureVersion: FIXTURE_VERSION, fixtures, caseInputs }));
  const fixtureMismatch = sha256(imageSource) !== manifest.image.sourceSha256 || sha256(imageRaster) !== manifest.image.raster.sha256 || hierarchyRecords.length !== manifest.hierarchy.paragraphCount;
  return { manifest, oracleSha256: sha256(oracleBytes), fixtures, cases, caseInputs, stimulusKey, fixtureMismatch, hierarchyRecords, hierarchyHtml, imageSource, imageRaster, imageDataUrl, textPrompt };
}
