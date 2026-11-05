import { readdir, readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { createModelMirrorHandler, MODEL_MIRROR_PATH } from './serve-model-mirror.mjs';

const root = new URL('../', import.meta.url);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.png': 'image/png' };

/** Loopback-only static server for the browser examples; only listed example and built bundle files are served. */
export async function createExampleServer() {
  const resources = new Map([
    ['/examples/sample-page/', 'examples/sample-page/index.html'],
    ['/examples/sample-page/index.html', 'examples/sample-page/index.html'],
    ['/examples/sample-page/cat.png', 'examples/sample-page/cat.png'],
    ['/examples/sample-page/chart.png', 'examples/sample-page/chart.png'],
    ['/examples/neko-reader.js', 'examples/neko-reader.js'],
  ]);
  try {
    for (const name of await readdir(new URL('dist/browser/', root))) if (name.endsWith('.js')) resources.set(`/dist/browser/${name}`, `dist/browser/${name}`);
    for (const name of await readdir(new URL('dist/browser/assets/', root))) if (/\.(?:m?js|wasm)$/.test(name)) resources.set(`/dist/browser/assets/${name}`, `dist/browser/assets/${name}`);
  } catch (error) {
    if (error?.code === 'ENOENT') throw new Error('Browser bundles are missing; run npm run build first');
    throw error;
  }
  let models;
  const server = createServer(async (request, response) => {
    if (request.url?.startsWith(MODEL_MIRROR_PATH)) {
      if (models) await models(request, response);
      else response.writeHead(404).end('Start with --cache-dir to serve pinned model files');
      return;
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') { response.writeHead(405, { allow: 'GET, HEAD' }).end(); return; }
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    if (pathname === '/') { response.writeHead(302, { location: '/examples/sample-page/' }).end(); return; }
    const file = resources.get(pathname);
    if (!file) { response.writeHead(404).end('Not found'); return; }
    try {
      const bytes = await readFile(new URL(file, root));
      response.writeHead(200, {
        'content-type': types[file.slice(file.lastIndexOf('.'))], 'content-length': bytes.byteLength,
        'cache-control': 'no-store', 'x-content-type-options': 'nosniff',
      }).end(request.method === 'HEAD' ? undefined : bytes);
    } catch {
      response.writeHead(500).end('Example file unavailable');
    }
  });
  /** Verifies the selected cached profile, then serves it same-origin under the pinned mirror path. */
  server.serveModels = async ({ cacheDir, profile }) => {
    models = await createModelMirrorHandler({ cacheDir, profile, allowedOrigin: `http://127.0.0.1:${server.address().port}` });
  };
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { values } = parseArgs({ options: {
    'cache-dir': { type: 'string' }, 'model-profile': { type: 'string', default: 'default' }, port: { type: 'string', default: '4174' },
  } });
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new TypeError('--port must be an integer from 0 through 65535');
  const server = await createExampleServer();
  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  if (values['cache-dir']) await server.serveModels({ cacheDir: values['cache-dir'], profile: values['model-profile'] });
  const origin = `http://127.0.0.1:${server.address().port}`;
  console.log(`Sample page with the Neko.js reader button: ${origin}/examples/sample-page/`);
  console.log(values['cache-dir'] ? `Model files (${values['model-profile']}): ${origin}${MODEL_MIRROR_PATH}` : 'No --cache-dir: the reader can only use models already in this browser\'s cache.');
}
