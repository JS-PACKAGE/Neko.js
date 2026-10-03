import type { ChatContent, ChatMessage, InferOptions, InferencePlan, InferenceResult, ModelIdentity } from './engine.js';
import type { ImageInput } from '../web/image.js';
import { awaitUser, NekoError } from '../errors.js';
import { validateInferenceOptions } from './preflight.js';

export interface SessionHost {
  infer(options: InferOptions): Promise<InferenceResult>;
  planInference(options: InferOptions): Promise<InferencePlan>;
}
export type SessionDefaults = Pick<InferOptions, 'maxNewTokens' | 'contextWindowTokens' | 'generation' | 'maxImageBytes' | 'timeoutMs' | 'hardDeadlineMs'>;
export interface SessionOptions extends SessionDefaults {
  system?: string;
  contextPolicy?: 'error' | 'drop-oldest';
  /** Includes the system message and reserves space for each complete user/assistant turn. */
  maxHistoryMessages?: number;
}
export type SessionSendOptions = SessionDefaults & Pick<InferOptions, 'signal' | 'onToken' | 'validateDestination'>;
export interface SessionControlOptions { signal?: AbortSignal; }
export type PortableSessionImage =
  | { type: 'data-url'; data: string }
  | { type: 'pixels'; data: string; width: number; height: number; channels: 1 | 2 | 3 | 4 };
export interface SessionSnapshot {
  version: 1;
  model: ModelIdentity;
  options: SessionOptions;
  messages: { role: ChatMessage['role']; content: string | ({ type: 'text'; text: string } | { type: 'image'; image: PortableSessionImage })[] }[];
}
interface CapturedSend { user: ChatMessage; options: SessionSendOptions; }
interface RestoredSession { options: SessionOptions; messages: ChatMessage[]; }

function record(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function keys(value: Record<string, unknown>, allowed: string[]): void {
  if (Object.keys(value).some((key) => !allowed.includes(key))) throw new TypeError('Unsupported session snapshot field');
}
function sameModel(a: ModelIdentity, b: ModelIdentity): boolean {
  return a.id === b.id && a.revision === b.revision && a.profile === b.profile
    && a.dtype?.embed_tokens === b.dtype.embed_tokens && a.dtype?.decoder_model_merged === b.dtype.decoder_model_merged
    && a.dtype?.vision_encoder === b.dtype.vision_encoder;
}
function cloneImage(image: ImageInput): ImageInput {
  if (typeof image === 'string') return image;
  if (image instanceof Blob) return image.slice(0, image.size, image.type);
  if (image instanceof URL) return new URL(image.href);
  return { width: image.width, height: image.height, channels: image.channels, data: new Uint8Array(image.data) };
}
function cloneContent(content: ChatMessage['content']): ChatMessage['content'] {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content) || !content.length) throw new TypeError('Session content must be text or a nonempty content array');
  return content.map((part) => {
    if (!part || typeof part !== 'object') throw new TypeError('Invalid session content');
    if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
    if (part.type === 'image' && part.image !== undefined) return { type: 'image', image: cloneImage(part.image) };
    throw new TypeError('Invalid session content');
  });
}
function cloneMessages(messages: ChatMessage[]): ChatMessage[] { return messages.map((message) => ({ role: message.role, content: cloneContent(message.content) })); }
function copyOptions<T extends SessionDefaults>(options: T): T {
  return { ...options, ...(options.generation ? { generation: { ...options.generation, ...(options.generation.stop ? { stop: [...options.generation.stop] } : {}), ...(options.generation.stopTokenIds ? { stopTokenIds: [...options.generation.stopTokenIds] } : {}) } } : {}) };
}
function settings(input: SessionOptions): SessionOptions {
  if (!record(input)) throw new TypeError('Session options must be an object');
  keys(input, ['system', 'contextPolicy', 'maxHistoryMessages', 'maxNewTokens', 'contextWindowTokens', 'generation', 'maxImageBytes', 'timeoutMs', 'hardDeadlineMs']);
  if (input.system !== undefined && typeof input.system !== 'string') throw new TypeError('Session system must be a string');
  if (input.contextPolicy !== undefined && input.contextPolicy !== 'error' && input.contextPolicy !== 'drop-oldest') throw new TypeError('Invalid session context policy');
  const maximum = input.maxHistoryMessages ?? 128;
  if (typeof maximum !== 'number' || !Number.isSafeInteger(maximum) || maximum < (input.system === undefined ? 2 : 3) || maximum > 128) throw new RangeError('maxHistoryMessages must fit a complete turn and be at most 128');
  validateInferenceOptions({ ...input, prompt: 'Validate session defaults.' }, true);
  return copyOptions<SessionOptions>({ ...input, contextPolicy: input.contextPolicy ?? 'error', maxHistoryMessages: maximum });
}
function encode(bytes: Uint8Array | Uint8ClampedArray): string {
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 16_384) binary += String.fromCharCode(...bytes.subarray(offset, offset + 16_384));
  return btoa(binary);
}
function decode(data: unknown, limit: number): Uint8Array {
  if (typeof data !== 'string' || data.length > Math.ceil(limit / 3) * 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(data)) throw new TypeError('Invalid portable image base64');
  const binary = atob(data);
  if (binary.length > limit) throw new RangeError('Portable image exceeds byte limit');
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  if (encode(bytes) !== data) throw new TypeError('Portable image base64 must be canonical');
  return bytes;
}
const rasterMime = /^image\/(?:png|jpeg|webp|gif|avif|tiff|bmp)$/i;
function portableDataUrl(value: unknown, limit: number): string {
  if (typeof value !== 'string') throw new TypeError('Invalid portable image data URL');
  const match = /^data:(image\/(?:png|jpeg|webp|gif|avif|tiff|bmp));base64,([A-Za-z0-9+/=]*)$/i.exec(value);
  if (!match || !decode(match[2], limit).length) throw new TypeError('Portable images require nonempty canonical raster base64 data URLs');
  return value;
}
async function portableImage(image: ImageInput, limit: number): Promise<PortableSessionImage> {
  if (typeof image === 'string') return { type: 'data-url', data: portableDataUrl(image, limit) };
  if (image instanceof URL) throw new TypeError('Session export does not serialize URL or filesystem image references; use embedded raster data');
  if (image instanceof Blob) {
    if (!rasterMime.test(image.type) || image.size > limit || !image.size) throw new TypeError('Session export requires a bounded raster image Blob');
    return { type: 'data-url', data: `data:${image.type};base64,${encode(new Uint8Array(await image.arrayBuffer()))}` };
  }
  if (image.data.byteLength > limit) throw new RangeError('Session export image exceeds snapshot byte limit');
  return { type: 'pixels', width: image.width, height: image.height, channels: image.channels, data: encode(image.data) };
}
function restoreImage(input: unknown, limit: number): ImageInput {
  if (!record(input)) throw new TypeError('Invalid portable session image');
  if (input.type === 'data-url') { keys(input, ['type', 'data']); return portableDataUrl(input.data, limit); }
  keys(input, ['type', 'data', 'width', 'height', 'channels']);
  if (input.type !== 'pixels' || typeof input.width !== 'number' || typeof input.height !== 'number' || typeof input.channels !== 'number' || ![1, 2, 3, 4].includes(input.channels)) throw new TypeError('Invalid portable pixel image');
  if (!Number.isSafeInteger(input.width) || !Number.isSafeInteger(input.height) || input.width < 1 || input.height < 1 || input.width * input.height > 40_000_000) throw new RangeError('Invalid portable pixel dimensions');
  const data = decode(input.data, limit);
  if (data.length !== input.width * input.height * input.channels) throw new TypeError('Portable pixel length does not match dimensions');
  return { data, width: input.width, height: input.height, channels: input.channels as 1 | 2 | 3 | 4 };
}
function freeze<T>(value: T): T {
  if (value && typeof value === 'object') { for (const child of Object.values(value)) freeze(child); Object.freeze(value); }
  return value;
}
function readSnapshot(input: unknown, model: ModelIdentity): RestoredSession {
  if (!record(input)) throw new TypeError('Session snapshot must be an object');
  keys(input, ['version', 'model', 'options', 'messages']);
  if (input.version !== 1) throw new TypeError('Unsupported session snapshot version');
  if (!record(input.model)) throw new TypeError('Invalid session model identity');
  keys(input.model, ['id', 'revision', 'profile', 'dtype']);
  if (!record(input.model.dtype)) throw new TypeError('Invalid session model dtype');
  keys(input.model.dtype, ['embed_tokens', 'decoder_model_merged', 'vision_encoder']);
  if (!sameModel(input.model as unknown as ModelIdentity, model)) throw new TypeError('Session snapshot model identity does not match this Neko');
  const options = settings(input.options as SessionOptions);
  if (!Array.isArray(input.messages) || input.messages.length > options.maxHistoryMessages!) throw new TypeError('Invalid session snapshot messages');
  const limit = options.maxImageBytes ?? 10 * 1024 * 1024;
  const messages: ChatMessage[] = input.messages.map((message: unknown) => {
    if (!record(message)) throw new TypeError('Invalid session message');
    keys(message, ['role', 'content']);
    if (message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') throw new TypeError('Invalid session message role');
    let content: ChatMessage['content'];
    if (typeof message.content === 'string') content = message.content;
    else {
      if (!Array.isArray(message.content) || !message.content.length) throw new TypeError('Invalid session message content');
      content = message.content.map((part: unknown): ChatContent => {
        if (!record(part)) throw new TypeError('Invalid session content part');
        if (part.type === 'text') { keys(part, ['type', 'text']); if (typeof part.text !== 'string') throw new TypeError('Invalid session text'); return { type: 'text', text: part.text }; }
        keys(part, ['type', 'image']);
        if (part.type !== 'image') throw new TypeError('Invalid session content type');
        return { type: 'image', image: restoreImage(part.image, limit) };
      });
    }
    return { role: message.role, content };
  });
  const start = options.system === undefined ? 0 : 1;
  if (start && (messages[0]?.role !== 'system' || messages[0]?.content !== options.system)) throw new TypeError('Session system message does not match its options');
  if ((messages.length - start) % 2) throw new TypeError('Session snapshots must contain complete turns');
  for (let index = start; index < messages.length; index++) {
    if (messages[index]!.role !== ((index - start) % 2 ? 'assistant' : 'user')) throw new TypeError('Session snapshots must alternate user and assistant turns');
    if ((index - start) % 2 === 0) validateInferenceOptions({ ...options, messages: messages.slice(0, index + 1) }, true);
  }
  return { options, messages };
}

export class ConversationSession {
  private messages: ChatMessage[];
  private options: SessionOptions;
  private readonly model: ModelIdentity;
  private readonly lifetime = new AbortController();
  private closed = false;
  private tail: Promise<void> = Promise.resolve();
  private disposal?: Promise<void>;
  constructor(private readonly host: SessionHost, model: ModelIdentity, options: SessionOptions = {}) {
    this.model = structuredClone(model);
    this.options = settings(options);
    this.messages = this.initialMessages();
  }
  private initialMessages(): ChatMessage[] { return this.options.system === undefined ? [] : [{ role: 'system', content: this.options.system }]; }
  private check(signal?: AbortSignal): void {
    if (this.closed) throw new NekoError('Conversation session is disposed', 'generate', 'DISPOSED');
    if (signal?.aborted) throw new NekoError('Session operation was cancelled', 'generate', 'ABORTED', { cause: signal.reason });
  }
  private enqueue<T>(signal: AbortSignal | undefined, operation: (signal: AbortSignal) => Promise<T>): Promise<T> {
    try { this.check(signal); if (signal !== undefined && !(signal instanceof AbortSignal)) throw new TypeError('signal must be an AbortSignal'); }
    catch (error) { return Promise.reject(error); }
    const abort = AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])]);
    const result = this.tail.then(async () => { this.check(abort); return operation(abort); });
    this.tail = result.then(() => {}, () => {});
    return awaitUser(() => result, abort, 'generate');
  }
  history(): ChatMessage[] { this.check(); return cloneMessages(this.messages); }
  private capture(content: string | ChatContent[], input: SessionSendOptions): CapturedSend {
    this.check(input.signal);
    if (!record(input)) throw new TypeError('Session send options must be an object');
    keys(input, ['maxNewTokens', 'contextWindowTokens', 'generation', 'maxImageBytes', 'timeoutMs', 'hardDeadlineMs', 'signal', 'onToken', 'validateDestination']);
    validateInferenceOptions({ ...this.options, ...input, messages: [...this.initialMessages(), { role: 'user', content }] }, true);
    const user: ChatMessage = { role: 'user', content: cloneContent(content) };
    const options = copyOptions(input);
    return { user, options };
  }
  private async prepare(user: ChatMessage, overrides: SessionSendOptions, signal: AbortSignal): Promise<{ messages: ChatMessage[]; options: InferOptions; plan: InferencePlan }> {
    // Clone before passing to a host; callers and custom hosts must not mutate retained history.
    const messages = cloneMessages([...this.messages, user]);
    const start = this.options.system === undefined ? 0 : 1;
    const drop = () => {
      if (this.options.contextPolicy !== 'drop-oldest' || messages.length - start < 3) throw new NekoError('Conversation exceeds its context or message limit', 'preprocess', 'CONTEXT_LIMIT');
      messages.splice(start, 2);
    };
    while (messages.length + 1 > this.options.maxHistoryMessages!) drop();
    const defaults = Object.fromEntries(Object.entries(this.options).filter(([key]) => !['system', 'contextPolicy', 'maxHistoryMessages'].includes(key))) as SessionDefaults;
    const options: InferOptions = copyOptions({ ...defaults, ...overrides, signal });
    const planning = { ...options };
    delete planning.onToken;
    while (true) {
      const plan = await this.host.planInference({ ...copyOptions(planning), messages: cloneMessages(messages) });
      this.check(signal);
      if (!sameModel(plan.model, this.model)) throw new NekoError('Conversation model identity changed', 'preprocess', 'INVALID_INPUT');
      if (plan.fits) return { messages, options, plan };
      drop();
    }
  }
  send(content: string | ChatContent[], options: SessionSendOptions = {}): Promise<InferenceResult> {
    let captured: CapturedSend;
    try { captured = this.capture(content, options); } catch (error) { return Promise.reject(error); }
    return this.enqueue(captured.options.signal, async (signal) => {
      const prepared = await this.prepare(captured.user, captured.options, signal);
      // Identical rendered input in plan/infer uses the engine's exact tokenized-prompt cache, not KV reuse.
      const result = await this.host.infer({ ...copyOptions(prepared.options), messages: cloneMessages(prepared.messages) });
      this.check(signal);
      if (!sameModel(result.model, this.model)) throw new NekoError('Conversation model identity changed', 'generate', 'INVALID_INPUT');
      if (typeof result.text !== 'string') throw new NekoError('Conversation inference returned invalid text', 'generate', 'MODEL_OUTPUT');
      this.messages = [...prepared.messages, { role: 'assistant', content: result.text }];
      return result;
    });
  }
  plan(content: string | ChatContent[], options: SessionSendOptions = {}): Promise<InferencePlan> {
    let captured: CapturedSend;
    try { captured = this.capture(content, options); } catch (error) { return Promise.reject(error); }
    return this.enqueue(captured.options.signal, async (signal) => (await this.prepare(captured.user, captured.options, signal)).plan);
  }
  reset(options: SessionControlOptions = {}): Promise<void> {
    return this.enqueue(options.signal, async () => { this.messages = this.initialMessages(); });
  }
  branch(options: SessionControlOptions = {}): Promise<ConversationSession> {
    return this.enqueue(options.signal, async () => {
      const branch = new ConversationSession(this.host, this.model, this.options);
      branch.messages = cloneMessages(this.messages);
      return branch;
    });
  }
  export(options: SessionControlOptions = {}): Promise<SessionSnapshot> {
    return this.enqueue(options.signal, async (signal) => {
      const messages: SessionSnapshot['messages'] = [];
      const limit = this.options.maxImageBytes ?? 10 * 1024 * 1024;
      for (const message of this.messages) {
        if (typeof message.content === 'string') messages.push({ role: message.role, content: message.content });
        else {
          const content: Exclude<SessionSnapshot['messages'][number]['content'], string> = [];
          for (const part of message.content) content.push(part.type === 'text' ? { type: 'text', text: part.text } : { type: 'image', image: await portableImage(part.image, limit) });
          messages.push({ role: message.role, content });
        }
      }
      this.check(signal);
      return freeze<SessionSnapshot>({ version: 1, model: structuredClone(this.model), options: copyOptions(this.options), messages });
    });
  }
  import(snapshot: unknown, options: SessionControlOptions = {}): Promise<void> {
    let captured: RestoredSession;
    try { this.check(options.signal); captured = readSnapshot(snapshot, this.model); } catch (error) { return Promise.reject(error); }
    return this.enqueue(options.signal, async () => { this.options = captured.options; this.messages = captured.messages; });
  }
  dispose(): Promise<void> {
    if (!this.disposal) { this.closed = true; this.lifetime.abort(new NekoError('Conversation session is disposed', 'generate', 'DISPOSED')); this.disposal = this.tail.then(() => { this.messages = []; }); }
    return this.disposal;
  }
  [Symbol.asyncDispose](): Promise<void> { return this.dispose(); }
}

export function createConversationSession(host: SessionHost, model: ModelIdentity, options: SessionOptions = {}): ConversationSession {
  return new ConversationSession(host, model, options);
}
