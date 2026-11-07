import assert from 'node:assert/strict';
import test from 'node:test';
import { validateGeneratedLanguage } from '../../src/report/language.js';
import { NekoError } from '../../src/errors.js';

const mismatch = (error: unknown) => error instanceof NekoError && error.code === 'LANGUAGE_MISMATCH';

test('Traditional Chinese with long product names and identifiers passes the zh-TW check', () => {
  // Real model output that the former letter-count rule rejected (17 Latin letters against 8 Han characters).
  validateGeneratedLanguage('TypeSafe AI 是後來 ChatGPT 誕生的基石。', 'zh-TW', 'Claim');
  validateGeneratedLanguage('node scripts/serve-examples.mjs 可在本機提供範例網頁供試用', 'zh-TW', 'Claim');
});

test('Latin-dominated, Han-free or Simplified text still fails the zh-TW check', () => {
  for (const text of ['Open 週二到週日 from 10 to 18', 'The café 咖啡館 is closed on Mondays.', 'The café is closed on Mondays.', '这个页面介绍咖啡馆。']) {
    assert.throws(() => validateGeneratedLanguage(text, 'zh-TW', 'Claim'), mismatch, text);
  }
});
