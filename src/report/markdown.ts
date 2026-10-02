import type { StructuredReport } from '../types.js';

const LABELS: Record<string, { summary: string; sections: string; images: string; conclusion: string; source: string }> = {
  en: { summary: 'Summary', sections: 'Key points', images: 'Image descriptions', conclusion: 'Conclusion', source: 'Source' },
  'zh-TW': { summary: '頁面摘要', sections: '分段重點', images: '圖片描述', conclusion: '整體結論', source: '來源' },
  zh: { summary: '頁面摘要', sections: '分段重點', images: '圖片描述', conclusion: '整體結論', source: '來源' },
};

function markdownText(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\\/g, '\\\\')
    .replace(/([`*_{}\[\]()#+.!|>])/g, '\\$1')
    .replace(/\r\n?/g, '\n')
    .replace(/\n/g, '  \n');
}

function markdownUrl(value: string): string | undefined {
  try {
    const url = new URL(value);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return undefined;
    return url.href.replace(/[()]/g, (character) => `%${character.charCodeAt(0).toString(16)}`);
  } catch {
    return undefined;
  }
}

export function renderMarkdown(report: StructuredReport): string {
  const locale = report.language.toLowerCase();
  const labels = LABELS[locale] ?? (locale.startsWith('zh-') ? LABELS.zh! : LABELS.en!);
  const lines = [`# ${markdownText(report.page.title || report.page.url)}`, '', `## ${labels.summary}`, '', markdownText(report.page.summary), ''];
  if (report.sections.length) {
    lines.push(`## ${labels.sections}`, '');
    for (const section of report.sections) {
      if (section.heading) lines.push(`### ${markdownText(section.heading)}`, '');
      for (const point of section.keyPoints) lines.push(`- ${markdownText(point)}`);
      if (section.keyPoints.length && section.paragraphIds.length) lines.push('');
      if (section.paragraphIds.length) lines.push(`_${labels.source}: ${section.paragraphIds.map(markdownText).join(', ')}_`, '');
    }
  }
  if (report.images.length) {
    lines.push(`## ${labels.images}`, '');
    for (const image of report.images) {
      if (!image.description.trim()) throw new TypeError(`Image ${image.imageId} is missing its generated description`);
      if (image.source.imageId !== image.imageId) throw new TypeError(`Image ${image.imageId} has inconsistent provenance`);
      const source = markdownUrl(image.url);
      const origin = source ? `[${markdownText(image.imageId)}](${source})` : markdownText(image.url);
      lines.push(`### ${markdownText(image.imageId)}${image.alt ? ` — ${markdownText(image.alt)}` : ''}`, '', markdownText(image.description), '', `_${labels.source}: ${origin}_`, '');
    }
  }
  lines.push(`## ${labels.conclusion}`, '', markdownText(report.conclusion));
  return `${lines.join('\n').trimEnd()}\n`;
}
