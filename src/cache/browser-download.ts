import type { DownloadIdentity, DownloadStage } from './download.js';

const databaseName = 'neko-model-download-staging-v1';
let database: Promise<IDBDatabase> | undefined;

async function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') throw new Error('Resumable model installation requires browser IndexedDB');
  if (!database) {
    const { promise, resolve, reject } = Promise.withResolvers<IDBDatabase>();
    database = promise;
    const request = indexedDB.open(databaseName, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore('files');
      request.result.createObjectStore('chunks');
    };
    request.onsuccess = () => {
      request.result.onversionchange = () => { request.result.close(); database = undefined; };
      resolve(request.result);
    };
    request.onerror = () => { database = undefined; reject(request.error); };
    request.onblocked = () => { database = undefined; reject(new Error('Model download staging database is blocked')); };
  }
  return database;
}

/** Staging is distinct from Transformers' verified CacheStorage namespace. */
export async function browserDownloadStage(key: string, promote: (body: ReadableStream<Uint8Array>) => Promise<void>): Promise<DownloadStage> {
  const db = await openDatabase();
  const range = IDBKeyRange.bound([key, 0], [key, Number.MAX_SAFE_INTEGER]);
  const reset = () => {
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    const transaction = db.transaction(['files', 'chunks'], 'readwrite');
    transaction.objectStore('files').delete(key);
    const cursor = transaction.objectStore('chunks').openKeyCursor(range);
    cursor.onsuccess = () => { if (cursor.result) { transaction.objectStore('chunks').delete(cursor.result.primaryKey); cursor.result.continue(); } };
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error('Model staging reset aborted'));
    transaction.onerror = () => reject(transaction.error);
    return promise;
  };
  const read = async (): Promise<ReadableStream<Uint8Array>> => {
    let offset = 0;
    let ended = false;
    return new ReadableStream<Uint8Array>({
      async pull(controller) {
        if (ended) return;
        try {
          const { promise, resolve, reject } = Promise.withResolvers<Uint8Array | undefined>();
          const transaction = db.transaction('chunks');
          const request = transaction.objectStore('chunks').get([key, offset]);
          request.onsuccess = () => resolve(request.result as Uint8Array | undefined);
          request.onerror = () => reject(request.error);
          const chunk = await promise;
          if (ended) return;
          if (!chunk) { ended = true; controller.close(); return; }
          if (!(chunk instanceof Uint8Array) || chunk.length === 0 || chunk.length > 1024 * 1024) throw new Error('Invalid model download staging chunk');
          offset += chunk.length;
          controller.enqueue(chunk);
        } catch (error) { ended = true; controller.error(error); }
      },
      cancel() { ended = true; },
    });
  };
  return {
    async inspect() {
      const { promise, resolve, reject } = Promise.withResolvers<{ identity: DownloadIdentity | null; size: number }>();
        const transaction = db.transaction('files');
        const request = transaction.objectStore('files').get(key);
        request.onsuccess = () => {
          const record = request.result as { identity?: DownloadIdentity; size?: number } | undefined;
          resolve({ identity: record?.identity ?? null, size: record?.size ?? 0 });
        };
        request.onerror = () => reject(request.error);
      return promise;
    },
    async begin(identity, offset) {
      if (!offset) await reset();
      let position = offset;
      let closed = false;
      const persist = async (bytes?: Uint8Array) => {
        const { promise, resolve, reject } = Promise.withResolvers<void>();
        const transaction = db.transaction(['files', 'chunks'], 'readwrite');
        if (bytes) transaction.objectStore('chunks').put(bytes, [key, position]);
        transaction.objectStore('files').put({ identity, size: position + (bytes?.length ?? 0) }, key);
        transaction.oncomplete = () => resolve();
        transaction.onabort = () => reject(transaction.error ?? new Error('Model staging write aborted'));
        transaction.onerror = () => reject(transaction.error);
        return promise;
      };
      await persist();
      return {
        async write(bytes) {
          if (closed) throw new Error('Model staging writer is closed');
          for (let start = 0; start < bytes.length; start += 1024 * 1024) {
            const chunk = bytes.subarray(start, Math.min(bytes.length, start + 1024 * 1024));
            await persist(chunk);
            position += chunk.length;
          }
        },
        async close() { closed = true; },
      };
    },
    read,
    reset,
    async promote() { await promote(await read()); await reset(); },
  };
}
