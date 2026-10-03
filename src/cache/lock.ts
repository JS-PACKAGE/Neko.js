import type * as NodeFs from 'node:fs/promises';
import type { constants as NodeConstants } from 'node:fs';
import type * as NodeNet from 'node:net';

let liveness: Promise<{ port: number; id: string }> | undefined;

async function nodeLiveness(): Promise<{ port: number; id: string }> {
  if (!liveness) {
    const { promise, resolve, reject } = Promise.withResolvers<{ port: number; id: string }>();
    liveness = promise;
    // Per-realm sockets close even when a worker is terminated while its process lives.
    const protocol = 'node:';
    let net: typeof NodeNet;
    try { net = await import(`${protocol}net`); }
    catch (error) { liveness = undefined; reject(error); return promise; }
    const id = globalThis.crypto.randomUUID();
    const server = net.createServer((socket) => { socket.unref(); socket.on('error', () => { socket.destroy(); }); socket.end(`${id}\n`); });
    server.on('error', (error) => { liveness = undefined; reject(error); });
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') { server.close(); liveness = undefined; reject(new Error('Model lock liveness endpoint is unavailable')); return; }
      server.unref();
      resolve({ port: address.port, id });
    });
  }
  return liveness;
}

interface NodeLockOptions {
  directory: string;
  key: string;
  validate(): Promise<void>;
  validateFile(path: string): Promise<void>;
}

/** Unique bakery tickets avoid the stale-lock unlink race of a shared lock pathname. */
export async function withNodeInstallLock<T>(options: NodeLockOptions, operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  // Node filesystem/process coordination has no browser equivalent.
  const protocol = 'node:';
  const fs: typeof NodeFs = await import(`${protocol}fs/promises`);
  const { constants }: { constants: typeof NodeConstants } = await import(`${protocol}fs`);
  const net: typeof NodeNet = await import(`${protocol}net`);
  const owner = await nodeLiveness();
  const prefix = `.neko-lock-${options.key}-`;
  const name = `${prefix}${process.pid}-${owner.port}-${owner.id}-${globalThis.crypto.randomUUID()}`;
  const entry = `${options.directory}/${name}`;
  const publication = `${entry}.ticket`;
  const missing = (error: unknown) => !!error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT';
  const pause = () => {
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    signal?.throwIfAborted();
    const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, 25);
    const abort = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort); reject(signal?.reason); };
    signal?.addEventListener('abort', abort, { once: true });
    return promise;
  };
  const inspect = async (peer: string): Promise<bigint | null | undefined> => {
    const match = /^([0-9]+)-([0-9]+)-([0-9a-f-]{36})-([0-9a-f-]{36})$/.exec(peer.slice(prefix.length));
    if (!match) return undefined;
    const port = Number(match[2]);
    if (!Number.isSafeInteger(port) || port < 1 || port > 65535) throw new Error('Invalid model lock liveness endpoint');
    const path = `${options.directory}/${peer}`;
    try {
      await options.validateFile(path);
      const { promise, resolve, reject } = Promise.withResolvers<boolean>();
      const socket = net.createConnection({ host: '127.0.0.1', port });
      let reply = '';
      const abort = () => { socket.destroy(); reject(signal?.reason); };
      signal?.addEventListener('abort', abort, { once: true });
      socket.setTimeout(5000, () => { socket.destroy(); reject(new Error('Model installation owner liveness check timed out')); });
      socket.on('data', (bytes: Uint8Array) => {
        reply += new TextDecoder().decode(bytes);
        if (reply.includes('\n') || reply.length > 64) { socket.destroy(); resolve(reply === `${match[3]}\n`); }
      });
      socket.on('end', () => resolve(reply === `${match[3]}\n`));
      socket.on('error', (error: NodeJS.ErrnoException) => {
        if (error.code === 'ECONNREFUSED') resolve(false);
        else reject(error);
      });
      let alive: boolean;
      try { signal?.throwIfAborted(); alive = await promise; }
      finally { socket.destroy(); signal?.removeEventListener('abort', abort); }
      if (!alive) {
        // This unique pathname belongs to a dead realm, never a replacement owner.
        await fs.unlink(path).catch((failure: unknown) => { if (!missing(failure)) throw failure; });
        await options.validateFile(`${path}.ticket`);
        await fs.unlink(`${path}.ticket`).catch((failure: unknown) => { if (!missing(failure)) throw failure; });
        return undefined;
      }
      const handle = await fs.open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
      try {
        const info = await handle.stat();
        if (!info.isFile() || info.size > 64 || process.getuid && info.uid !== process.getuid()) throw new Error('Unsafe model installation lock');
        const content = await handle.readFile('utf8');
        if (!content || content === 'choosing') return null;
        if (!/^[1-9][0-9]{0,30}$/.test(content)) throw new Error('Invalid model installation lock ticket');
        return BigInt(content);
      } finally { await handle.close(); }
    } catch (error) { if (missing(error)) return undefined; throw error; }
  };
  signal?.throwIfAborted();
  await options.validate();
  const handle = await fs.open(entry, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  try {
    try { await handle.writeFile('choosing'); await handle.sync(); }
    finally { await handle.close(); }
    let maximum = 0n;
    for (const peer of await fs.readdir(options.directory)) {
      if (!peer.startsWith(prefix) || peer === name) continue;
      const ticket = await inspect(peer);
      if (ticket !== null && ticket !== undefined && ticket > maximum) maximum = ticket;
    }
    const ticket = maximum + 1n;
    const selected = await fs.open(publication, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
    try { await selected.writeFile(String(ticket)); await selected.sync(); }
    finally { await selected.close(); }
    await options.validate();
    await fs.rename(publication, entry);
    while (true) {
      signal?.throwIfAborted();
      let waiting = false;
      for (const peer of await fs.readdir(options.directory)) {
        if (!peer.startsWith(prefix) || peer === name) continue;
        const other = await inspect(peer);
        if (other === null || other !== undefined && (other < ticket || other === ticket && peer < name)) { waiting = true; break; }
      }
      if (!waiting) break;
      await pause();
    }
    await options.validate();
    signal?.throwIfAborted();
    return await operation();
  } finally {
    await options.validate();
    await fs.unlink(entry).catch((error: unknown) => { if (!missing(error)) throw error; });
    await fs.unlink(publication).catch((error: unknown) => { if (!missing(error)) throw error; });
  }
}

export async function withBrowserInstallLock<T>(key: string, operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  if (typeof navigator === 'undefined' || !navigator.locks) throw new Error('Resumable model installation requires the browser Web Locks API');
  return navigator.locks.request(`neko:model-install:${key}`, { mode: 'exclusive', ...(signal ? { signal } : {}) }, async () => {
    signal?.throwIfAborted();
    return operation();
  });
}
