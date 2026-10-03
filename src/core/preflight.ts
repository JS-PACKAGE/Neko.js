import type { ChatMessage, InferOptions } from './engine.js';
import type { ImageInput } from '../web/image.js';
import { generationSettings } from './generation.js';
import { NekoError } from '../errors.js';

export function inferenceChat(options: Pick<InferOptions, 'prompt' | 'messages' | 'image' | 'images'>) {
  if (options.image !== undefined && options.images !== undefined) throw new TypeError('image and images are exclusive');
  if (options.messages !== undefined && (options.prompt !== undefined || options.image !== undefined || options.images !== undefined)) throw new TypeError('messages and prompt/image/images shorthand are exclusive');
  if (options.images !== undefined && (!Array.isArray(options.images) || !options.images.length)) throw new TypeError('images must be a nonempty array');
  const supplied = options.images ?? (options.image === undefined ? [] : [options.image]);
  const messages: ChatMessage[] = options.messages ?? [{ role: 'user', content: [...supplied.map((image) => ({ type: 'image' as const, image })), { type: 'text', text: options.prompt! }] }];
  if (!Array.isArray(messages) || messages.length < 1 || messages.length > 128) throw new TypeError('messages must contain between 1 and 128 entries');
  const images: ImageInput[] = [];
  let hasText = false;
  const rendered = messages.map((message, index) => {
    if (!message || !['system', 'user', 'assistant'].includes(message.role) || message.role === 'system' && index !== 0) throw new TypeError('Messages need valid roles; system must be first');
    const content = typeof message.content === 'string' ? [{ type: 'text' as const, text: message.content }] : message.content;
    if (!Array.isArray(content) || !content.length) throw new TypeError('Message content must be text or a nonempty content array');
    return { role: message.role, content: content.map((item) => {
      if (!item || typeof item !== 'object') throw new TypeError('Invalid message content');
      if (item.type === 'text') { if (typeof item.text !== 'string') throw new TypeError('Message text must be a string'); hasText ||= !!item.text.trim(); return { type: 'text', text: item.text }; }
      if (item.type !== 'image' || item.image === undefined || message.role === 'system') throw new TypeError('Invalid message image content');
      images.push(item.image); return { type: 'image' };
    }) };
  });
  if (!hasText) throw new TypeError('A nonempty prompt or message text is required');
  if (images.length > 16) throw new RangeError('At most 16 joint images are supported per request');
  return { rendered, images };
}

/** Checks requiring no tokenizer, image decode, cache access or model acquisition. */
export function validateInferenceOptions(options: InferOptions, planning = false): void {
  try {
    if (!options || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('Inference options must be an object');
    const { images } = inferenceChat(options);
    const output = options.maxNewTokens ?? 128;
    if (!Number.isSafeInteger(output) || output < 1 || output > 2048) throw new RangeError('maxNewTokens must be between 1 and 2048');
    const context = options.contextWindowTokens;
    if (context !== undefined && (!Number.isSafeInteger(context) || context < 32)) throw new RangeError('contextWindowTokens must be a safe integer of at least 32');
    if (options.hardDeadlineMs !== undefined && (!Number.isSafeInteger(options.hardDeadlineMs) || options.hardDeadlineMs < 1 || options.hardDeadlineMs > 2_147_483_647)) throw new RangeError('hardDeadlineMs must be between 1 and 2147483647');
    if (!planning && context !== undefined && output + 1 > context) throw new NekoError('Output budget plus minimum input exceeds context limit', 'preprocess', 'CONTEXT_LIMIT');
    generationSettings(options.generation, Number.MAX_SAFE_INTEGER);
    for (const name of ['onToken', 'validateDestination'] as const) if (options[name] !== undefined && typeof options[name] !== 'function') throw new TypeError(`${name} must be a function`);
    if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) throw new TypeError('signal must be an AbortSignal');
    const maxBytes = options.maxImageBytes ?? 10 * 1024 * 1024;
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new RangeError('maxImageBytes must be a positive safe integer');
    const timeout = options.timeoutMs ?? 10_000;
    if (!Number.isSafeInteger(timeout) || timeout < 1 || timeout > 2_147_483_647) throw new RangeError('timeoutMs must be between 1 and 2147483647');
    for (const image of images) {
      if (typeof image === 'string') { if (!image.length) throw new TypeError('Image source must not be empty'); continue; }
      if (image instanceof URL) continue;
      if (image instanceof Blob) { if (image.size > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`); continue; }
      if (!image || typeof image !== 'object') throw new TypeError('Invalid image source');
      if (!Number.isSafeInteger(image.width) || !Number.isSafeInteger(image.height) || image.width < 1 || image.height < 1 || image.width * image.height > 40_000_000) throw new RangeError('Decoded image exceeds pixel limit (40000000)');
      if (!(image.data instanceof Uint8Array || image.data instanceof Uint8ClampedArray) || ![1, 2, 3, 4].includes(image.channels) || image.data.length !== image.width * image.height * image.channels) throw new TypeError('Decoded image has invalid pixel data');
      if (image.data.byteLength > maxBytes) throw new RangeError(`Image exceeds byte limit (${maxBytes})`);
    }
  } catch (cause) {
    if (cause instanceof NekoError) throw cause;
    throw new NekoError(cause instanceof Error ? cause.message : 'Invalid inference options', 'preprocess', 'INVALID_INPUT', { cause });
  }
}
