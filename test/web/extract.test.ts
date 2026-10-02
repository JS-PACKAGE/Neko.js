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
