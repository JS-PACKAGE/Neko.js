import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const root = new URL('../examples/sample-page/', import.meta.url);
const resources = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/cat.png', ['cat.png', 'image/png']],
  ['/chart.png', ['chart.png', 'image/png']],
]);

/** Loopback-only static server for examples/sample-page/, e.g. to try the browser extension on it. */
export function createExampleServer() {
  return createServer(async (request, response) => {
    if (request.method !== 'GET' && request.method !== 'HEAD') { response.writeHead(405, { allow: 'GET, HEAD' }).end(); return; }
    const resource = resources.get(new URL(request.url ?? '/', 'http://localhost').pathname);
    if (!resource) { response.writeHead(404).end('Not found'); return; }
    try {
      const bytes = await readFile(new URL(resource[0], root));
      response.writeHead(200, { 'content-type': resource[1], 'content-length': bytes.byteLength, 'x-content-type-options': 'nosniff' })
        .end(request.method === 'HEAD' ? undefined : bytes);
    } catch {
      response.writeHead(500).end('Example file unavailable');
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { values } = parseArgs({ options: { port: { type: 'string', default: '4174' } } });
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new TypeError('--port must be an integer from 0 through 65535');
  const server = createExampleServer();
  server.listen(port, '127.0.0.1', () => console.log(`Sample page: http://127.0.0.1:${server.address().port}/`));
}
