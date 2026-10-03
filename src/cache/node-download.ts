import type * as NodeFs from 'node:fs/promises';
import type { constants as NodeConstants } from 'node:fs';
import type * as NodeStream from 'node:stream';
import type { DownloadIdentity, DownloadStage } from './download.js';

export interface NodeDownloadStageOptions {
  directory: string;
  key: string;
  validate(): Promise<void>;
  validateFile(path: string): Promise<void>;
  destination(): Promise<string>;
}

/** Filesystem staging shares the verified cache's ancestry, UID, and no-symlink guards. */
export async function nodeDownloadStage(options: NodeDownloadStageOptions): Promise<DownloadStage> {
  // These platform-only modules cannot be statically imported by the browser bundle.
  const protocol = 'node:';
  const fs: typeof NodeFs = await import(`${protocol}fs/promises`);
  const { constants }: { constants: typeof NodeConstants } = await import(`${protocol}fs`);
  const { Readable }: typeof NodeStream = await import(`${protocol}stream`);
  const partial = `${options.directory}/.neko-stage-${options.key}.part`;
  const metadata = `${options.directory}/.neko-stage-${options.key}.json`;
  const missing = (error: unknown) => !!error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT';
  const reset = async () => {
    await options.validate();
    for (const path of [partial, metadata]) {
      await options.validateFile(path);
      await fs.unlink(path).catch((error: unknown) => { if (!missing(error)) throw error; });
    }
  };
  return {
    async inspect() {
      await options.validate();
      await options.validateFile(partial);
      await options.validateFile(metadata);
      let size = 0;
      let identity: DownloadIdentity | null = null;
      try { size = (await fs.lstat(partial)).size; }
      catch (error) { if (!missing(error)) throw error; }
      try {
        const handle = await fs.open(metadata, constants.O_RDONLY | constants.O_NOFOLLOW);
        try {
          if ((await handle.stat()).size > 16384) return { identity: null, size };
          const content = await handle.readFile('utf8');
          try { identity = JSON.parse(content) as DownloadIdentity; }
          catch { identity = null; }
        } finally { await handle.close(); }
      } catch (error) { if (!missing(error)) throw error; }
      return { identity, size };
    },
    async begin(identity, offset) {
      await options.validate();
      await options.validateFile(partial);
      await options.validateFile(metadata);
      const temporary = `${metadata}.${globalThis.crypto.randomUUID()}.tmp`;
      const meta = await fs.open(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
      try {
        await meta.writeFile(JSON.stringify(identity));
        await meta.sync();
        await meta.close();
        await options.validateFile(metadata);
        await fs.rename(temporary, metadata);
      } finally {
        await meta.close().catch(() => undefined);
        await fs.unlink(temporary).catch((error: unknown) => { if (!missing(error)) throw error; });
      }
      const handle = await fs.open(partial, constants.O_WRONLY | constants.O_CREAT | constants.O_NOFOLLOW | (offset ? 0 : constants.O_TRUNC), 0o600);
      const info = await handle.stat();
      if (!info.isFile() || info.size !== offset || process.getuid && info.uid !== process.getuid()) {
        await handle.close();
        throw new Error('Model download staging changed before append');
      }
      let position = offset;
      let closed = false;
      return {
        async write(bytes) {
          if (closed) throw new Error('Model staging writer is closed');
          let start = 0;
          while (start < bytes.length) {
            const { bytesWritten } = await handle.write(bytes, start, bytes.length - start, position);
            if (!bytesWritten) throw new Error('Model staging write made no progress');
            start += bytesWritten;
            position += bytesWritten;
          }
        },
        async close() {
          if (closed) return;
          closed = true;
          try { await handle.sync(); }
          finally { await handle.close(); }
        },
      };
    },
    async read() {
      await options.validate();
      await options.validateFile(partial);
      const handle = await fs.open(partial, constants.O_RDONLY | constants.O_NOFOLLOW);
      const info = await handle.stat();
      if (!info.isFile() || process.getuid && info.uid !== process.getuid()) { await handle.close(); throw new Error('Unsafe model download staging file'); }
      return Readable.toWeb(handle.createReadStream()) as ReadableStream<Uint8Array<ArrayBuffer>>;
    },
    reset,
    async promote() {
      await options.validate();
      await options.validateFile(partial);
      const destination = await options.destination();
      await fs.rename(partial, destination);
      await options.validateFile(destination);
      await options.validateFile(metadata);
      await fs.unlink(metadata).catch((error: unknown) => { if (!missing(error)) throw error; });
    },
  };
}
