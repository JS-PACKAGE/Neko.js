import assert from 'node:assert/strict';
import { once } from 'node:events';
import test from 'node:test';
import { createSiteServer } from './preview-site.mjs';


test('built website serves inert HTML, safe links and bounded static routes', async () => {
  const server = createSiteServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const origin = `http://127.0.0.1:${server.address().port}`;
  try {
    const response = await fetch(`${origin}/`);
    assert.equal(response.status, 200);
    assert.match(response.headers.get('content-type'), /^text\/html; charset=utf-8$/);
    assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
    const html = await response.text();
    assert.doesNotMatch(html, /<script\b|<iframe\b|<form\b|<img\b/i);
    assert.match(html, /default-src 'none'/);
    const links = Array.from(html.matchAll(/<a\b[^>]*href="([^"]+)"/g), ([, href]) => href);
    for (const href of links) {
      if (href.startsWith('#')) {
        assert.ok(html.includes(`id="${href.slice(1)}"`), `Missing fragment ${href}`);
        continue;
      }
      const url = new URL(href);
      assert.equal(url.protocol, 'https:');
      assert.equal(url.hostname, 'github.com');
    }
    const head = await fetch(`${origin}/index.html`, { method: 'HEAD' });
    assert.equal(head.status, 200);
    assert.equal(await head.text(), '');
    assert.equal((await fetch(`${origin}/CNAME`)).status, 200);
    assert.equal((await fetch(`${origin}/.nojekyll`)).status, 200);
    assert.equal((await fetch(`${origin}/SECURITY.md`)).status, 404);
    assert.equal((await fetch(`${origin}/..%2Fpackage.json`)).status, 404);
    assert.equal((await fetch(`${origin}/`, { method: 'POST' })).status, 405);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});
