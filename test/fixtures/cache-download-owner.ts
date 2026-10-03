import { createHash } from 'node:crypto';
import { lstat, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { downloadResumable } from '../../src/cache/download.js';
import { nodeDownloadStage } from '../../src/cache/node-download.js';
import { withNodeInstallLock } from '../../src/cache/lock.js';

const [directory, source, sizeText, sha256] = process.argv.slice(2);
if (!directory || !source || !sizeText || !sha256) throw new Error('Missing cache download owner fixture arguments');
const size = Number(sizeText);
const destination = join(directory, 'verified.bin');
const validate = async () => {
  const info = await lstat(directory);
  if (!info.isDirectory() || info.isSymbolicLink() || process.getuid && info.uid !== process.getuid()) throw new Error('Unsafe fixture directory');
};
const validateFile = async (path: string) => {
  try {
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || process.getuid && info.uid !== process.getuid()) throw new Error('Unsafe fixture file');
  } catch (error) { if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'ENOENT') throw error; }
};
if (process.send) {
  const start = Promise.withResolvers<void>();
  process.once('message', () => start.resolve());
  process.send('ready');
  await start.promise;
  process.disconnect();
}
await withNodeInstallLock({ directory, key: 'fixture', validate, validateFile }, async () => {
  try {
    const installed = await readFile(destination);
    if (installed.length !== size || createHash('sha256').update(installed).digest('hex') !== sha256) throw new Error('Corrupt concurrent fixture install');
    return;
  } catch (error) { if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'ENOENT') throw error; }
  const stage = await nodeDownloadStage({ directory, key: 'fixture', validate, validateFile, async destination() { return destination; } });
  await downloadResumable({
    source, size, sha256, stage,
    async fetch(headers, signal) {
      const response = await fetch(source, { headers, ...(signal ? { signal } : {}) });
      return { response, destination: source };
    },
  });
});
