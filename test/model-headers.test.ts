import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdtemp, realpath, rm } from 'node:fs/promises';
import { createServer, type IncomingMessage } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { env } from '@huggingface/transformers';
import { captureModelSource, installVerifiedCache, type VerifiedCacheInstallation } from '../src/cache/model.js';
import { modelFileUrl } from '../src/cache/manifest.js';

// These sentinels are deliberately fake; the probe never reads credentials or environment variables.
const sentinels = { authorization: 'Bearer NEKO_FAKE_AUTHORIZATION_ONLY', cookie: 'neko_fake_cookie=sentinel_only', 'proxy-authorization': 'Basic NEKO_FAKE_PROXY_AUTHORIZATION_ONLY' };
const fixture = JSON.stringify({
  bos_token_id: 248044, do_sample: true, eos_token_id: [248046, 248044], pad_token_id: 248044,
  temperature: 0.6, top_k: 20, top_p: 0.95, transformers_version: '5.3.0.dev0', trust_remote_code: false,
}, null, 2) + '\n';

test('model fetch preserves same-origin headers but strips credentials permanently on mirror remaps and cross-origin redirect bounces', async () => {
  const directory = await mkdtemp(join(await realpath(tmpdir()), 'neko-header-boundary-'));
  const originalFetch = env.fetch;
  const canonical = modelFileUrl('generation_config.json');
  const observed: { path: string; headers: Record<string, string | undefined> }[] = [];
  const observe = (request: IncomingMessage) => observed.push({ path: request.url!, headers: Object.fromEntries(['authorization', 'cookie', 'proxy-authorization', 'accept', 'x-neko-probe'].map((name) => [name, request.headers[name] as string | undefined])) });
  let originA = '';
  let originB = '';
  const serverA = createServer((request, response) => {
    observe(request);
    if (request.url === '/direct/start') response.writeHead(302, { location: `${canonical}?same=1` }).end();
    else if (request.url === '/direct/same') response.writeHead(302, { location: `${originB}/cross` }).end();
    else response.writeHead(200, { 'content-length': String(Buffer.byteLength(fixture)) }).end(fixture);
  });
  const serverB = createServer((request, response) => { observe(request); response.writeHead(302, { location: `${originA}/bounce` }).end(); });
  let installation: VerifiedCacheInstallation | undefined;
  try {
    serverA.listen(0, '127.0.0.1'); serverB.listen(0, '127.0.0.1');
    await Promise.all([once(serverA, 'listening'), once(serverB, 'listening')]);
    originA = `http://127.0.0.1:${(serverA.address() as AddressInfo).port}`;
    originB = `http://127.0.0.1:${(serverB.address() as AddressInfo).port}`;
    // Transport only the pinned upstream fixture over loopback, leaving its SDK-visible origin intact.
    // All request headers are observed on real HTTP servers, not echoed by a fetch mock.
    env.fetch = (input, init) => {
      const url = new URL(String(input));
      return fetch(url.origin === new URL(canonical).origin ? `${originA}/direct/${url.search ? 'same' : 'start'}` : url, init);
    };
    const headers = new Headers({ ...sentinels, accept: 'application/json', 'x-neko-probe': 'public-probe' });
    class PrivateSource {
      #url = `${originA}/mirror`;
      reads = 0;
      get baseUrl() { this.reads++; return this.#url; }
      relocate() { this.#url = `${originB}/unapproved/`; }
    }
    const source = new PrivateSource();
    const captured = captureModelSource(source)!;
    source.relocate();
    assert.throws(() => Object.assign(captured, { baseUrl: `${originB}/unapproved/` }), TypeError);
    installation = await installVerifiedCache({ cacheDir: directory, modelSource: structuredClone(captured) });
    assert.equal(await (await env.fetch(canonical, { headers }) as Response).text(), fixture);
    assert.equal(source.reads, 1);
    installation.restore(); installation = undefined;
    installation = await installVerifiedCache({ cacheDir: directory, policy: { network: () => true } });
    assert.equal(await (await env.fetch(canonical, { headers }) as Response).text(), fixture);
    assert.deepEqual(observed.map(({ path }) => path), ['/mirror/generation_config.json', '/direct/start', '/direct/same', '/cross', '/bounce']);
    for (const [index, request] of observed.entries()) {
      for (const [name, sentinel] of Object.entries(sentinels)) assert.equal(request.headers[name], index === 1 || index === 2 ? sentinel : undefined);
      assert.equal(request.headers.accept, 'application/json');
      assert.equal(request.headers['x-neko-probe'], 'public-probe');
    }
    for (const [name, sentinel] of Object.entries(sentinels)) assert.equal(headers.get(name), sentinel);
  } finally {
    installation?.restore(); env.fetch = originalFetch;
    serverA.closeAllConnections(); serverB.closeAllConnections();
    await Promise.all([new Promise<void>((resolve) => serverA.close(() => resolve())), new Promise<void>((resolve) => serverB.close(() => resolve()))]);
    await rm(directory, { recursive: true, force: true });
  }
});
