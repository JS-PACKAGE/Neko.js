import type { Page, StructuredReport } from '../types.js';

export function validateStructuredReport(report: StructuredReport, page: Page): void {
  if (report.page.url !== page.url) throw new TypeError('Report URL does not match the extracted page');
  const paragraphIds = new Set(page.paragraphs.map(({ id }) => id));
  for (const section of report.sections) {
    for (const id of section.paragraphIds) {
      if (!paragraphIds.has(id)) throw new TypeError(`Report references unknown paragraph ${id}`);
    }
  }
  if (report.images.length !== page.images.length) throw new TypeError('Report must describe every extracted image exactly once');
  const expected = new Map(page.images.map((image) => [image.id, image]));
  const seen = new Set<string>();
  for (const image of report.images) {
    const source = expected.get(image.imageId);
    if (!source) throw new TypeError(`Report references unknown image ${image.imageId}`);
    if (seen.has(image.imageId)) throw new TypeError(`Report describes image ${image.imageId} more than once`);
    if (image.url !== source.url || image.source.imageId !== image.imageId) throw new TypeError(`Image ${image.imageId} has inconsistent provenance`);
    if (image.alt !== undefined && image.alt !== source.alt) throw new TypeError(`Image ${image.imageId} alt text does not match extracted metadata`);
    if (!image.description.trim()) throw new TypeError(`Image ${image.imageId} is missing its generated description`);
    seen.add(image.imageId);
  }
}
