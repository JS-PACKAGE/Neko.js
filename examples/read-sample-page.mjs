// Reads examples/sample-page/ the way an application reads a real web page: fetch it by URL,
// extract its main content, then generate a structured report (text sections plus image descriptions).
// Run from a built checkout (`npm run build`); the first run needs network access to download the model:
//   node examples/read-sample-page.mjs [--cache-dir=DIR] [--json] [--question=TEXT ...]
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { parseArgs } from 'node:util';
import { createNeko, extractPage } from 'neko.js';

const { values } = parseArgs({ options: {
  'cache-dir': { type: 'string' },
  json: { type: 'boolean', default: false },
  question: { type: 'string', multiple: true, default: [] },
} });

// Serve only the sample page and its two images, on loopback.
const files = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/cat.png', ['cat.png', 'image/png']],
  ['/chart.png', ['chart.png', 'image/png']],
]);
const server = createServer(async (request, response) => {
  const file = files.get(new URL(request.url ?? '/', 'http://localhost').pathname);
  if (!file || request.method !== 'GET') { response.writeHead(404).end(); return; }
  response.writeHead(200, { 'content-type': file[1] }).end(await readFile(new URL(`sample-page/${file[0]}`, import.meta.url)));
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/`;

// Fetching URLs from Node can reach internal services: pages and images may come only from the sample origin.
const sampleOrigin = new URL(url).origin;
const validateDestination = (destination) => {
  if (destination.origin !== sampleOrigin) throw new Error(`Blocked destination: ${destination.href}`);
};

let neko;
try {
  // Inert parsing: no scripts run and nothing but the page itself is fetched here.
  const page = await extractPage(url, { content: 'main', validateDestination });
  // With --json, stdout carries only the report.
  (values.json ? console.error : console.log)(`Read ${url}: "${page.title}", ${page.paragraphs.length} paragraphs, ${page.tables?.length ?? 0} table(s), ${page.images.length} image(s) from <${page.extraction?.root}>.\n`);

  // The first run downloads and verifies the pinned model (about 871 MB) into the cache.
  // A custom policy replaces the default one, so it also approves the model and runtime downloads.
  neko = await createNeko({
    device: 'cpu', ...(values['cache-dir'] ? { cacheDir: values['cache-dir'] } : {}),
    policy: { network: (target, kind) => (kind === 'page' || kind === 'image' ? target.origin === sampleOrigin : true) },
  });
  // Chinese summaries need more tokens than the 256-token default; a stage that runs out fails as invalid JSON.
  const report = await neko.describe(page, { format: values.json ? 'json' : 'markdown', language: 'zh-TW', maxNewTokens: 512, validateDestination });
  console.log(values.json ? JSON.stringify(report, null, 2) : report);

  for (const question of values.question) {
    const answer = await neko.ask(page, question);
    console.log(`\nQ: ${question}\n${answer.status}: ${answer.answer}`);
  }
} catch (error) {
  console.error(error instanceof Error ? `${error.name}: ${error.message}` : error);
  process.exitCode = 1;
} finally {
  await neko?.dispose();
  server.close();
}
