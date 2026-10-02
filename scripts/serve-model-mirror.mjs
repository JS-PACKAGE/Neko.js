import { constants } from 'node:fs';
import { lstat, open, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { dirname, join, parse, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { pipeline } from 'node:stream/promises';
import { MODEL_ID, MODEL_REVISION, getModelProfile } from '../dist/src/cache/manifest.js';

export const MODEL_MIRROR_PATH = `/models/${MODEL_ID}/${MODEL_REVISION}/`;

/** Serves only registered pinned assets; no filesystem paths are derived from request input. */
export async function createModelMirror({ cacheDir, profile = 'default', allowedOrigin } = {}) {
  if (!cacheDir) throw new TypeError('cacheDir is required');
  const root = resolve(cacheDir);
  if (root === parse(root).root) throw new TypeError('A filesystem root cannot be a model cache');
  if (allowedOrigin) {
    const origin = new URL(allowedOrigin);
    if (!['http:', 'https:'].includes(origin.protocol) || origin.origin !== allowedOrigin) throw new TypeError('allowedOrigin must be an exact HTTP(S) origin');
  }
  const files = getModelProfile(profile).files;
  const uid = process.getuid?.();
  const owned = (info) => uid === undefined || info.uid === uid;
  async function verifiedHandle(name) {
    const spec = files[name];
    const entry = join(root, MODEL_ID, MODEL_REVISION, name);
    const ancestors = [];
    for (let current = dirname(entry); current !== dirname(current); current = dirname(current)) ancestors.push(current);
    for (const directory of ancestors.reverse()) {
      const info = await lstat(directory);
      if (info.isSymbolicLink() || !info.isDirectory() || ((directory === root || directory.startsWith(`${root}/`)) && !owned(info))) throw new Error('Unsafe model cache directory');
    }
    if (await realpath(root) !== root) throw new Error('Model cache directory changed during validation');
    const before = await lstat(entry);
    if (!before.isFile() || before.isSymbolicLink() || !owned(before)) throw new Error('Unsafe model cache entry');
    const handle = await open(entry, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const info = await handle.stat();
      if (!info.isFile() || !owned(info) || info.dev !== before.dev || info.ino !== before.ino || info.size !== spec.size) throw new Error('Pinned model size or file identity mismatch');
      const hash = createHash('sha256');
      const buffer = Buffer.allocUnsafe(Math.min(1024 * 1024, spec.size));
      let offset = 0;
      while (offset < spec.size) {
        const { bytesRead } = await handle.read(buffer, 0, Math.min(buffer.length, spec.size - offset), offset);
        if (!bytesRead) throw new Error('Pinned model file was truncated');
        hash.update(buffer.subarray(0, bytesRead));
        offset += bytesRead;
      }
      const after = await handle.stat();
      if (hash.digest('hex') !== spec.sha256 || after.size !== info.size || after.mtimeMs !== info.mtimeMs || after.ctimeMs !== info.ctimeMs) throw new Error('Pinned model SHA-256 mismatch or file changed during verification');
      return handle;
    } catch (error) { await handle.close(); throw error; }
  }
  // Fail before listening if any selected-profile asset is absent, unsafe, or corrupt.
  for (const name of Object.keys(files)) await (await verifiedHandle(name)).close();
  const paths = new Map(Object.keys(files).map((name) => [`${MODEL_MIRROR_PATH}${name}`, name]));
  return createServer(async (request, response) => {
    let handle;
    try {
      const address = response.socket.localAddress;
      const port = response.socket.localPort;
      if (!['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(address) || ![`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`].includes(request.headers.host)) {
        response.writeHead(403).end('Loopback host required'); return;
      }
      if (request.headers.origin && request.headers.origin !== allowedOrigin) { response.writeHead(403).end('Origin denied'); return; }
      const cors = allowedOrigin ? { 'access-control-allow-origin': allowedOrigin, vary: 'Origin', 'access-control-expose-headers': 'Content-Length, Content-Range, Accept-Ranges' } : {};
      if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405, { allow: 'GET, HEAD', ...cors }).end(); return; }
      // Exact raw path lookup rejects encoded traversal, queries, and unregistered profiles/assets.
      const name = paths.get(request.url);
      if (!name) { response.writeHead(404, cors).end('Not a pinned model asset'); return; }
      const size = files[name].size;
      if (request.headers.range && request.headers.range !== 'bytes=0-0') {
        response.writeHead(416, { 'content-range': `bytes */${size}`, ...cors }).end(); return;
      }
      handle = await verifiedHandle(name);
      const range = request.headers.range === 'bytes=0-0';
      response.writeHead(range ? 206 : 200, {
        ...cors, 'content-type': name.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        'content-length': String(range ? 1 : size), 'accept-ranges': 'bytes',
        'cache-control': 'no-store', 'x-content-type-options': 'nosniff',
        'cross-origin-resource-policy': allowedOrigin ? 'cross-origin' : 'same-origin',
        ...(range ? { 'content-range': `bytes 0-0/${size}` } : {}),
      });
      if (request.method === 'HEAD') { response.end(); return; }
      await pipeline(handle.createReadStream({ start: 0, end: range ? 0 : size - 1, autoClose: false }), response);
    } catch {
      if (!response.headersSent) response.writeHead(500).end('Model cache verification failed');
      else response.destroy();
    } finally { await handle?.close().catch(() => undefined); }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: {
    'cache-dir': { type: 'string' }, 'model-profile': { type: 'string', default: 'default' },
    port: { type: 'string', default: '8787' }, 'allow-origin': { type: 'string' },
  } });
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new TypeError('--port must be an integer from 0 through 65535');
  const server = await createModelMirror({ cacheDir: values['cache-dir'], profile: values['model-profile'], allowedOrigin: values['allow-origin'] });
  server.listen(port, '127.0.0.1', () => console.log(JSON.stringify({
    modelSource: { baseUrl: `http://127.0.0.1:${server.address().port}${MODEL_MIRROR_PATH}` }, profile: values['model-profile'],
  })));
}
