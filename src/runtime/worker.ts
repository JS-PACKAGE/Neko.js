import { Neko } from '../index.js';
import { installWorkerServer } from './server.js';
import type { MainMessage, MessagePort, WorkerExecution, WorkerMessage } from './protocol.js';
import type * as NodeWorkers from 'node:worker_threads';

let port: MessagePort<MainMessage, WorkerMessage>;
let execution: WorkerExecution;
if (typeof process !== 'undefined' && process.release?.name === 'node') {
  // Browser bundles cannot import the Node-only worker_threads module statically.
  const protocol = 'node:';
  const { parentPort, threadId, isMainThread }: typeof NodeWorkers = await import(`${protocol}worker_threads`);
  if (isMainThread || !parentPort) throw new Error('SDK worker entrypoint must execute inside a worker thread');
  port = {
    post: (message) => parentPort.postMessage(message),
    listen(listener) { parentPort.on('message', listener); return () => { parentPort.off('message', listener); }; },
  };
  execution = { mode: 'worker', runtime: 'node', threadId, workerId: `node:${threadId}` };
} else {
  const scope = globalThis as unknown as {
    postMessage(message: WorkerMessage): void;
    addEventListener(type: 'message', listener: (event: MessageEvent<MainMessage>) => void): void;
    removeEventListener(type: 'message', listener: (event: MessageEvent<MainMessage>) => void): void;
  };
  if (typeof scope.postMessage !== 'function' || typeof document !== 'undefined') throw new Error('SDK worker entrypoint must execute inside a module Worker');
  port = {
    post: (message) => scope.postMessage(message),
    listen(listener) {
      const receive = (event: MessageEvent<MainMessage>) => listener(event.data);
      scope.addEventListener('message', receive);
      return () => { scope.removeEventListener('message', receive); };
    },
  };
  // Generated in the dedicated worker, never by the proxy; browsers expose no native threadId.
  execution = { mode: 'worker', runtime: 'browser', workerId: `browser:${crypto.randomUUID()}` };
}
installWorkerServer(port, (options) => Neko.create(options), execution);
