import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { expect, test } from '@playwright/test';

let server;
let origin;
let webSource;
let reportSource;
let backendSource;
const requests = [];
test.beforeAll(async () => {

  webSource = await readFile(new URL('../../dist/browser/web.js', import.meta.url), 'utf8');
  reportSource = await readFile(new URL('../../dist/browser/report.js', import.meta.url), 'utf8');
  backendSource = await readFile(new URL('../../dist/browser/backend.js', import.meta.url), 'utf8');
  server = createServer(async (request, response) => {
    requests.push(request.url);
    if (request.url === '/web.js') {
      response.writeHead(200, { 'content-type': 'text/javascript' }).end(webSource);
    } else if (request.url === '/report.js') {
      response.writeHead(200, { 'content-type': 'text/javascript' }).end(reportSource);
    } else if (request.url === '/backend.js') {
      response.writeHead(200, { 'content-type': 'text/javascript' }).end(backendSource);
    } else {
      response.writeHead(200, { 'content-type': 'text/html' }).end('<!doctype html><title>Prototype test</title>');
    }
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  origin = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
  if (server?.listening) await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});


test('browser backend capability is observed and an unavailable WebGPU request never becomes CPU', async ({ page }) => {
  await page.goto(origin);
  const result = await page.evaluate(async () => {
    const { inspectBackend } = await import('/backend.js');
    const cpu = await inspectBackend('cpu');
    try {
      return { cpu, webgpu: await inspectBackend('webgpu'), webgpuError: null };
    } catch (error) {
      return { cpu, webgpu: null, webgpuError: error instanceof Error ? error.message : String(error) };
    }
  });
  expect(result.cpu).toMatchObject({ runtime: 'browser', device: 'cpu', executionProviders: ['wasm'] });
  if (result.webgpu) {
    expect(result.webgpu.device).toBe('webgpu');
    expect(result.webgpu.runtime).toBe('browser');
  } else {
    expect(result.webgpuError).toMatch(/WebGPU API is not present|WebGPU adapter is unavailable|shader-f16/);
  }
});

test('bundled browser parser extracts ordered content without executing or fetching page resources', async ({ page }) => {
  await page.goto(origin);
  const result = await page.evaluate(async () => {
    const { extractPage } = await import('/web.js');
    const source = '<html><head><title>Meta title</title><script>window.injectedPageCode = true</script><link rel="stylesheet" href="/remote.css?passive-check"></head><body><div>Intro <b>text</b></div><h2>Section</h2><p>Body</p><picture><source srcset="/small.png 1x, /large.png 2x"><img src="/small.png" alt="Picture"></picture><div style="background-image:url(/background.webp)"></div><img src="/never-load.png?passive-check"></body></html>';
    const extracted = await extractPage(source, { baseUrl: location.href });
    return {
      paragraphs: extracted.paragraphs,
      images: extracted.images,
      executed: window.injectedPageCode === true,
    };
  });
  expect(result.paragraphs.map(({ text }) => text)).toEqual(['Intro text', 'Section', 'Body']);
  expect(result.paragraphs[2].source).toMatchObject({ kind: 'html', startOffset: expect.any(Number), endOffset: expect.any(Number) });
  expect(result.images.map(({ url }) => new URL(url).pathname)).toEqual(['/small.png', '/large.png', '/background.webp', '/never-load.png']);
  expect(result.images[0].discoveredBy).toContain('picture');
  expect(result.images[0].alt).toBe('Picture');
  expect(requests).not.toContain('/remote.css?passive-check');
  expect(requests).not.toContain('/never-load.png?passive-check');
});

test('loads and normalizes a real browser-created PNG through fetch and canvas', async ({ page }) => {
  await page.goto(origin);
  const fixtureBytes = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 16;
    const context = canvas.getContext('2d');
    context.fillStyle = '#3057a1';
    context.fillRect(0, 0, 32, 16);
    const encoded = canvas.toDataURL('image/png').split(',')[1];
    return Array.from(atob(encoded), (character) => character.charCodeAt(0));
  });
  await page.route('**/fixture.png', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: Buffer.from(fixtureBytes) }));
  const processed = await page.evaluate(async () => {
    const { loadImage } = await import('/web.js');
    return loadImage({ id: 'i1', url: `${location.origin}/fixture.png`, discoveredBy: ['img'] });
  });
  expect(processed.imageId).toBe('i1');
  expect(processed.mimeType).toBe('image/png');
  const dimensions = await page.evaluate(async (bytes) => {
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: 'image/png' }));
    const result = [bitmap.width, bitmap.height];
    bitmap.close();
    return result;
  }, Array.from(processed.data));
  expect(dimensions).toEqual([32, 16]);
});

test('rejects HTML/image limit overflow and malformed or non-raster image payloads', async ({ page }) => {
  await page.goto(origin);
  const result = await page.evaluate(async () => {
    const { extractPage, loadImage } = await import('/web.js');
    const errors = [];
    try { await extractPage('<img src="/a"><img src="/b">', { baseUrl: location.href, maxImages: 1 }); } catch (error) { errors.push(error.name); }
    try { await extractPage('<p>Too large</p>', { maxHtmlBytes: 2 }); } catch (error) { errors.push(error.name); }
    try { await loadImage({ id: 'bad', url: 'data:image/png;base64,PHN2Zz48L3N2Zz4=', discoveredBy: ['img'] }); } catch (error) { errors.push(error.name); }
    return errors;
  });
  expect(result).toEqual(['RangeError', 'RangeError', 'TypeError']);
});

test('renders untrusted model text as inert Markdown and only links HTTP(S) provenance', async ({ page }) => {
  await page.goto(origin);
  const markdown = await page.evaluate(async () => {
    const { renderMarkdown } = await import('/report.js');
    return renderMarkdown({
      schemaVersion: 2,
      language: 'en',
      imageFailurePolicy: 'error',
      page: { url: 'https://example.test', summary: '<img src=x onerror=alert(1)> [click](javascript:alert(1))' },
      sections: [{ heading: 'Heading', keyPoints: ['**bold** <svg/onload=alert(1)>'], paragraphIds: ['p1'] }],
      images: [{ imageId: 'i1', url: 'data:image/png;base64,AAAA', status: 'described', description: '![attack](javascript:alert(1)) <script>bad</script>', source: { kind: 'image', imageId: 'i1' }, alt: '<img>' }],
      conclusion: '[open](https://evil.test) & <b>unsafe</b>',
      sourceFacts: [{ id: 'q1', citation: { kind: 'quote', paragraphId: 'p1', startOffset: 0, endOffset: 37, quote: '<script>alert(1)</script> [source](x)' } }],
      metadata: { coverage: { selectedTextCharacters: 37, retainedTextCharacters: 37, retainedQuoteCount: 1, modelCitedFactIds: ['q1'], summaryCitedFactIds: ['q1'] } },
    });
  });
  expect(markdown).not.toContain('<img');
  expect(markdown).not.toContain('<script>');
  expect(markdown).not.toMatch(/\]\(javascript:/i);
  expect(markdown).not.toMatch(/\]\(data:/i);
  expect(markdown).toContain('&lt;img');
  expect(markdown).toContain('&lt;script');
  expect(markdown).toContain('i1');
  expect(markdown).toContain('p1');
});
