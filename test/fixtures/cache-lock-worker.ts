import { lstat } from 'node:fs/promises';
import { parentPort, workerData } from 'node:worker_threads';
import { withNodeInstallLock } from '../../src/cache/lock.js';

const directory = workerData.directory as string;
const port = parentPort;
if (!port) throw new Error('Lock owner fixture requires a worker');
await withNodeInstallLock({
  directory, key: 'terminated-worker',
  async validate() { const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('Unsafe fixture directory'); },
  async validateFile(path) {
    try { const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error('Unsafe fixture lock'); }
    catch (error) { if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'ENOENT') throw error; }
  },
}, async () => {
  const held = Promise.withResolvers<void>();
  port.once('message', () => held.resolve());
  port.postMessage('locked');
  await held.promise;
});
