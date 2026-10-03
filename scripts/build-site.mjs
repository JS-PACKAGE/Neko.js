import { copyFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const destination = join(root, 'dist', 'site');
await mkdir(destination, { recursive: true });
for (const file of ['index.html', 'CNAME', '.nojekyll']) {
  await copyFile(join(root, file), join(destination, file));
}
console.log('Built dist/site (static HTML; CNAME preserved; no deployment).');
