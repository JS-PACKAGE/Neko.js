import { NekoError } from '../errors.js';

/**
 * Script checks detect clear language mismatches; they are not a fluency or translation-quality guarantee.
 * Text that is exactly a retained source quote is source text rather than generated text (a navigation label such
 * as "3 C" has no script of its own), so `quotes` exempts it.
 */
export function validateGeneratedLanguage(value: string, language: string, field: string, quotes?: ReadonlySet<string>): void {
  if (quotes?.has(value.trim())) return;
  const locale = language.toLowerCase();
  const han = value.match(/\p{Script=Han}/gu)?.length ?? 0;
  const latin = value.match(/\p{Script=Latin}/gu)?.length ?? 0;
  // Brand names and identifiers ("TypeSafe AI", "serve-examples.mjs") are long Latin words in ordinary Chinese prose,
  // so Chinese is weighed against Latin words rather than letters.
  const latinWords = value.match(/\p{Script=Latin}+/gu)?.length ?? 0;
  if (locale === 'zh-tw' && (han === 0 || latinWords * 2 > han || /[这为与个从们来时会说对国学实应现过还进种无点开关颜红蓝]/u.test(value))) {
    throw new NekoError(`${field} did not satisfy the requested Traditional Chinese language check`, 'report', 'LANGUAGE_MISMATCH');
  }
  if ((locale === 'en' || locale.startsWith('en-')) && (latin === 0 || han > latin)) {
    throw new NekoError(`${field} did not satisfy the requested English language check`, 'report', 'LANGUAGE_MISMATCH');
  }
}
