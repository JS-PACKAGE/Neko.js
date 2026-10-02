import type { Page, StructuredReport } from '../types.js';
import { ERROR_CODES, ERROR_STAGES } from '../errors.js';
import { validateGeneratedLanguage } from './language.js';

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

export function validateStructuredReport(report: unknown, page: Page): asserts report is StructuredReport {
  object(report, 'Report');
  text(report.language, 'Report language');
  try { Intl.getCanonicalLocales(report.language); } catch { throw new TypeError('Report language must be a valid BCP 47 language tag'); }
  if (report.imageFailurePolicy !== 'error' && report.imageFailurePolicy !== 'omit') throw new TypeError('Report imageFailurePolicy is invalid');
  object(report.page, 'Report page');
  text(report.page.url, 'Report URL');
  text(report.page.summary, 'Report summary');
  text(report.conclusion, 'Report conclusion');
  validateGeneratedLanguage(report.page.summary, report.language, 'Summary');
  validateGeneratedLanguage(report.conclusion, report.language, 'Conclusion');
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
    for (const point of section.keyPoints) validateGeneratedLanguage(point, report.language, 'Section key point');
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
    if (image.status === 'described') {
      if (typeof image.description !== 'string' || !image.description.trim()) throw new TypeError(`Image ${image.imageId} is missing its generated description`);
      validateGeneratedLanguage(image.description, report.language, `Image ${image.imageId} description`);
      if (image.error !== undefined) throw new TypeError('Described image cannot include an error');
    } else if (image.status === 'failed' && report.imageFailurePolicy === 'omit') {
      object(image.error, 'Image failure');
      text(image.error.stage, 'Image error stage'); text(image.error.code, 'Image error code'); text(image.error.message, 'Image error message');
      if (!Object.hasOwn(ERROR_STAGES, image.error.stage) || !Object.hasOwn(ERROR_CODES, image.error.code)) throw new TypeError('Image error stage or code is invalid');
      if (image.description !== undefined) throw new TypeError('Failed image cannot claim a generated description');
    } else throw new TypeError('Report image status conflicts with its failure policy');
    seen.add(image.imageId);
  }
}
