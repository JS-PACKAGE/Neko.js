import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import test from 'node:test';
import { extractPage } from '../../src/web/index.js';

test('extracts visible text once in DOM order and retains source offsets', async () => {
  const html = '<html><head><title>Page title</title><script>throw new Error("must not run")</script></head><body><div>Intro <b>text</b></div><h2>Heading</h2><p>Paragraph</p><div>Ending text</div></body></html>';
  const page = await extractPage(html);
  assert.deepEqual(page.paragraphs.map(({ text }) => text), ['Intro text', 'Heading', 'Paragraph', 'Ending text']);
  assert.equal(page.paragraphs[2]!.heading, 'Heading');
  assert.equal(page.title, 'Page title');
  assert.ok(page.paragraphs[2]!.source.startOffset !== undefined);
  assert.ok(page.paragraphs[2]!.source.endOffset !== undefined);
  assert.match(html.slice(page.paragraphs[2]!.source.startOffset, page.paragraphs[2]!.source.endOffset), /<p>Paragraph<\/p>/);
  assert.doesNotMatch(page.paragraphs.map(({ text }) => text).join(' '), /must not run|Page title/);
});

test('extracts image sources with provenance, resolves relative URLs, and ignores CSS font resources', async () => {
  const html = '<html><head><meta property="og:image" content="/social.png"><style>@font-face{src:url(/font.woff2)} .hero{background-image:url(/css-bg.webp)}</style></head><body><picture><source srcset="/small.png 1x, /large.png 2x"><img src="/small.png" alt="A scene"></picture><div style="background: url(\'/inline.jpg\')"></div></body></html>';
  const page = await extractPage(html, { baseUrl: 'https://example.test/article' });
  const imageByUrl = new Map(page.images.map((image) => [image.url, image]));
  assert.deepEqual([...imageByUrl.keys()].sort(), [
    'https://example.test/social.png',
    'https://example.test/small.png',
    'https://example.test/large.png',
    'https://example.test/css-bg.webp',
    'https://example.test/inline.jpg',
  ].sort());
  const pictureImage = imageByUrl.get('https://example.test/small.png');
  assert.deepEqual(pictureImage?.discoveredBy, ['picture']);
  assert.equal(pictureImage?.alt, 'A scene');
  assert.ok(!imageByUrl.has('https://example.test/font.woff2'));
  assert.ok(imageByUrl.get('https://example.test/social.png')?.discoveredBy.includes('og:image'));
  assert.ok(imageByUrl.get('https://example.test/css-bg.webp')?.discoveredBy.includes('background'));
  assert.ok(imageByUrl.get('https://example.test/inline.jpg')?.discoveredBy.includes('background'));
});

test('fails rather than truncating images or oversized HTML and rejects non-web schemes', async () => {
  await assert.rejects(extractPage('<img src="/one.png"><img src="/two.png">', { baseUrl: 'https://example.test', maxImages: 1 }), RangeError);
  await assert.rejects(extractPage('<p>too long</p>', { maxHtmlBytes: 4 }), RangeError);
  await assert.rejects(extractPage('file:///etc/passwd'), /Only http and https/);
  assert.deepEqual((await extractPage('<img src="https://example.test/a.png">', { includeImages: false })).images, []);
});

test('fetches only the HTML document and enforces response bounds, status, and MIME', async (t) => {
  let imageRequests = 0;
  const server = createServer((request, response) => {
    if (request.url === '/page') {
      response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      response.end('<main><p>Local page</p><img src="/image.png" alt="fixture"></main>');
    } else if (request.url === '/redirect') {
      response.writeHead(302, { location: '/page' }).end('redirect body');
    } else if (request.url === '/image.png') {
      imageRequests++;
      response.writeHead(200, { 'content-type': 'image/png' }).end('not fetched by HTML extraction');
    } else if (request.url === '/too-large') {
      response.writeHead(200, { 'content-type': 'text/html' });
      response.write('<html>');
      response.end('content larger than the configured cap');
    } else if (request.url === '/wrong-type') {
      response.writeHead(200, { 'content-type': 'application/json' }).end('{}');
    } else {
      response.writeHead(404).end();
    }
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const origin = `http://127.0.0.1:${address.port}`;

  const page = await extractPage(`${origin}/redirect`);
  assert.equal(page.url, `${origin}/page`);
  assert.equal(page.paragraphs[0]!.text, 'Local page');
  assert.equal(imageRequests, 0);
  await assert.rejects(extractPage(`${origin}/too-large`, { maxHtmlBytes: 10 }), RangeError);
  await assert.rejects(extractPage(`${origin}/wrong-type`), /Expected an HTML response/);
  await assert.rejects(extractPage(`${origin}/missing`), /status 404/);
});

test('destination policy sees redirects before any denied destination is requested', async (t) => {
  let deniedRequests = 0;
  const server = createServer((request, response) => {
    if (request.url === '/redirect') response.writeHead(302, { location: '/private' }).end();
    else { deniedRequests++; response.writeHead(200, { 'content-type': 'text/html' }).end('<p>private</p>'); }
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening'); t.after(() => server.close());
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const observed: string[] = [];
  await assert.rejects(extractPage(`http://127.0.0.1:${address.port}/redirect`, { validateDestination(url) { observed.push(url.pathname); if (url.pathname === '/private') throw new Error('Destination denied'); } }), /Destination denied/);
  assert.deepEqual(observed, ['/redirect', '/private']); assert.equal(deniedRequests, 0);
});

test('cancellation stops a pending streamed HTML response', async (t) => {
  let began: () => void = () => {};
  const started = new Promise<void>((resolve) => { began = resolve; });
  const server = createServer((_request, response) => { response.writeHead(200, { 'content-type': 'text/html' }); response.write('<p>'); began(); });
  server.listen(0, '127.0.0.1'); await once(server, 'listening'); t.after(() => { server.closeAllConnections(); server.close(); });
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const controller = new AbortController();
  const extraction = extractPage(`http://127.0.0.1:${address.port}/stream`, { signal: controller.signal });
  await started; controller.abort(); await assert.rejects(extraction, { name: 'AbortError' });
});

test('structured tables retain captions, spans, exact source offsets and paragraph/header relations', async () => {
  const html = '<main><section><h2>Counts 🌳</h2><table><caption>Tree counts</caption><thead><tr><th id="name" scope="col" rowspan="2">Species</th><th scope="colgroup" colspan="2">Trees</th></tr><tr><th id="old" scope="col">Old</th><th scope="col">New</th></tr></thead><tbody><tr><th id="oak" scope="row">Oak</th><td headers="oak old">4</td><td>2</td></tr></tbody></table></section></main>';
  const page = await extractPage(html);
  const table = page.tables![0]!;
  assert.equal(table.caption!.text, 'Tree counts');
  assert.equal(table.rowCount, 3); assert.equal(table.columnCount, 3);
  const species = table.cells[0]!;
  assert.equal(species.rowSpan, 2);
  const grouped = table.cells[1]!;
  assert.equal(grouped.columnSpan, 2);
  const four = table.cells.find((cell) => cell.paragraphIds.some((id) => page.paragraphs.find((paragraph) => paragraph.id === id)?.text === '4'))!;
  assert.deepEqual([four.row, four.column], [2, 1]);
  assert.equal(four.rowHeaderIds.length, 1); assert.equal(four.columnHeaderIds.length, 1);
  assert.deepEqual(new Set(four.headerIds), new Set([...four.rowHeaderIds, ...four.columnHeaderIds]));
  const paragraph = page.paragraphs.find(({ id }) => id === four.paragraphIds[0])!;
  assert.equal(page.containers!.find(({ id }) => id === paragraph.containerId)!.kind, 'td');
  assert.equal(page.containers!.find(({ id }) => id === paragraph.sectionId)!.kind, 'section');
  assert.equal(html.slice(paragraph.source.startOffset, paragraph.source.endOffset), '<td headers="oak old">4</td>');
  assert.equal(html.slice(table.source.startOffset, table.source.endOffset).startsWith('<table>'), true);
});

test('main extraction is opt-in and ambiguous or empty landmarks conservatively fall back', async () => {
  const html = '<nav><p>Navigation</p></nav><main><p>Article 🌳</p><img src="/main.png"></main><footer><p>Footer</p><img src="/footer.png"></footer>';
  const full = await extractPage(html, { baseUrl: 'https://example.test' });
  assert.deepEqual(full.paragraphs.map(({ text }) => text), ['Navigation', 'Article 🌳', 'Footer']);
  const main = await extractPage(html, { content: 'main', baseUrl: 'https://example.test' });
  assert.deepEqual(main.paragraphs.map(({ text }) => text), ['Article 🌳']);
  assert.deepEqual(main.images.map(({ url }) => url), ['https://example.test/main.png']);
  assert.deepEqual(main.extraction, { mode: 'main', root: 'main', fallback: false });
  assert.equal(html.slice(main.paragraphs[0]!.source.startOffset, main.paragraphs[0]!.source.endOffset), '<p>Article 🌳</p>');
  assert.equal((await extractPage('<article><p>Only article</p></article>', { content: 'main' })).extraction!.root, 'article');
  const ambiguous = await extractPage('<main><p>One</p></main><main><p>Two</p></main>', { content: 'main' });
  assert.deepEqual(ambiguous.paragraphs.map(({ text }) => text), ['One', 'Two']);
  assert.equal(ambiguous.extraction!.fallback, true);
  assert.equal((await extractPage('<main hidden><p>Hidden</p></main><p>Visible</p>', { content: 'main' })).extraction!.fallback, true);
  await assert.rejects(extractPage(html, { content: 'unknown' as 'main' }), /content must be/);
});

test('table rowSpan zero stops at the row group and nested tables retain source ownership', async () => {
  const page = await extractPage('<table><tbody><tr><th rowspan="0" scope="rowgroup">Group</th><td><table><tr><td>Nested</td></tr></table></td></tr><tr><td>Next</td></tr></tbody><tbody><tr><td>Other</td></tr></tbody></table>');
  assert.equal(page.tables![0]!.cells[0]!.rowSpan, 2);
  assert.equal(page.tables!.length, 2);
  const nested = page.paragraphs.find(({ text }) => text === 'Nested')!;
  assert.ok(page.tables![0]!.paragraphIds.includes(nested.id));
  assert.ok(page.tables![1]!.paragraphIds.includes(nested.id));
});
