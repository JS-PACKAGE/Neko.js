import { NekoError } from '../errors.js';

/** Script checks detect clear language mismatches; they are not a fluency or translation-quality guarantee. */
export function validateGeneratedLanguage(value: string, language: string, field: string): void {
  const locale = language.toLowerCase();
  const han = value.match(/\p{Script=Han}/gu)?.length ?? 0;
  const latin = value.match(/\p{Script=Latin}/gu)?.length ?? 0;
  if (locale === 'zh-tw' && (han === 0 || latin > han * 2 || /[这为与个从们来时会说对国学实应现过还进种无点开关颜红蓝]/u.test(value))) {
    throw new NekoError(`${field} did not satisfy the requested Traditional Chinese language check`, 'report', 'LANGUAGE_MISMATCH');
  }
  if ((locale === 'en' || locale.startsWith('en-')) && (latin === 0 || han > latin)) {
    throw new NekoError(`${field} did not satisfy the requested English language check`, 'report', 'LANGUAGE_MISMATCH');
  }
}
