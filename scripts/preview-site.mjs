import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const defaultDirectory = fileURLToPath(new URL('../dist/site/', import.meta.url));
const resources = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/CNAME', ['CNAME', 'text/plain; charset=utf-8']],
  ['/.nojekyll', ['.nojekyll', 'text/plain; charset=utf-8']],
]);

export function createSiteServer(directory = defaultDirectory) {
  return createServer(async (request, response) => {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { allow: 'GET, HEAD' }).end();
      return;
    }
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    const resource = resources.get(pathname);
    if (!resource) {
      response.writeHead(404).end('Not found');
      return;
    }
    try {
      const bytes = await readFile(join(directory, resource[0]));
      response.writeHead(200, {
        'content-type': resource[1],
        'content-length': bytes.byteLength,
        'x-content-type-options': 'nosniff',
        'content-security-policy': "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'",
      }).end(request.method === 'HEAD' ? undefined : bytes);
    } catch (error) {
      response.writeHead(error?.code === 'ENOENT' ? 404 : 500).end('Website artifact unavailable; run node scripts/build-site.mjs');
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number(process.argv[2] ?? 4173);
  if (process.argv.length > 3 || !Number.isSafeInteger(port) || port < 0 || port > 65535) {
    throw new TypeError('Usage: node scripts/preview-site.mjs [port from 0 through 65535]');
  }
  const server = createSiteServer();
  server.listen(port, '127.0.0.1', () => {
    console.log(`Static website preview: http://127.0.0.1:${server.address().port}/`);
  });
}
