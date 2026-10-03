import { LogitsProcessor } from '@huggingface/transformers';
import { NekoError } from '../errors.js';

export interface GrammarSchema {
  type: 'any' | 'object' | 'array' | 'string' | 'number' | 'integer' | 'boolean' | 'null';
  properties?: Record<string, GrammarSchema>;
  required?: readonly string[];
  items?: GrammarSchema;
  minLength?: number | undefined;
  maxLength?: number | undefined;
  minItems?: number | undefined;
  maxItems?: number | undefined;
  uniqueItems?: boolean | undefined;
  enum?: readonly unknown[];
}
const anySchema: GrammarSchema = { type: 'any' };
const canonical = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(',')}}`;
  return JSON.stringify(value);
};
interface Frame { schema: GrammarSchema; value: unknown[] | Record<string, unknown>; phase: 'key' | 'colon' | 'value' | 'comma' | 'next'; key?: string; }
interface Lexeme { kind: 'string' | 'number' | 'literal'; schema: GrammarSchema; text: string; key: boolean; escape: boolean; unicode: string | null; literal?: string; }
/** A copyable incremental recognizer. Tokens and UTF-8 code points can end at any byte boundary. */
export class JsonGrammarState {
  private frames: Frame[] = [];
  private lexeme: Lexeme | undefined;
  private rootDone = false;
  private utfRemaining = 0;
  private utfCode = 0;
  private utfMin = 0;
  constructor(private readonly schema: GrammarSchema) {}
  clone(): JsonGrammarState {
    const state = new JsonGrammarState(this.schema);
    state.frames = this.frames.map((frame) => ({ ...frame, value: Array.isArray(frame.value) ? [...frame.value] : { ...frame.value } }));
    state.lexeme = this.lexeme && { ...this.lexeme };
    state.rootDone = this.rootDone; state.utfRemaining = this.utfRemaining; state.utfCode = this.utfCode; state.utfMin = this.utfMin;
    return state;
  }
  /** Ordinary string bytes depend only on the pending UTF-8 prefix; delimiters still use the full state. */
  get freeStringClass(): number | null {
    const token = this.lexeme;
    if (token?.kind !== 'string' || token.key || token.escape || token.unicode !== null || token.schema.enum !== undefined || token.schema.maxLength !== undefined) return null;
    if (this.utfRemaining === 2) return this.utfCode === 0 ? 4 : this.utfCode === 13 ? 5 : 2;
    if (this.utfRemaining === 3) return this.utfCode === 0 ? 6 : this.utfCode === 4 ? 7 : 3;
    return this.utfRemaining;
  }
  get complete(): boolean {
    if (this.utfRemaining) return false;
    const copy = this.clone();
    if (copy.lexeme?.kind === 'number' && !copy.finishNumber()) return false;
    return copy.rootDone && !copy.lexeme;
  }
  push(bytes: Uint8Array): boolean { for (const byte of bytes) if (!this.pushByte(byte)) return false; return true; }
  pushByte(byte: number): boolean {
    if (this.utfRemaining) {
      if (byte < 0x80 || byte > 0xbf) return false;
      if (this.utfRemaining === 2 && this.utfCode === 0 && byte < 0xa0 || this.utfRemaining === 2 && this.utfCode === 13 && byte > 0x9f || this.utfRemaining === 3 && this.utfCode === 0 && byte < 0x90 || this.utfRemaining === 3 && this.utfCode === 4 && byte > 0x8f) return false;
      this.utfCode = this.utfCode * 64 + (byte & 63);
      --this.utfRemaining;
      const candidates = this.stringCandidates();
      if (candidates && !candidates.some((candidate) => candidate.startsWith(this.lexeme!.text) && candidate.length > this.lexeme!.text.length && candidate.codePointAt(this.lexeme!.text.length)! >>> (this.utfRemaining * 6) === this.utfCode)) return false;
      if (this.utfRemaining) return true;
      if (this.utfCode < this.utfMin || this.utfCode > 0x10ffff || this.utfCode >= 0xd800 && this.utfCode <= 0xdfff) return false;
      return this.character(String.fromCodePoint(this.utfCode));
    }
    if (byte < 128) return this.character(String.fromCharCode(byte));
    if (this.lexeme?.kind !== 'string' || this.lexeme.escape || this.lexeme.unicode !== null) return false;
    if (this.lexeme.schema.maxLength !== undefined && Array.from(this.lexeme.text).length >= this.lexeme.schema.maxLength) return false;
    if (byte >= 0xc2 && byte <= 0xdf) { this.utfRemaining = 1; this.utfCode = byte & 31; this.utfMin = 0x80; }
    else if (byte >= 0xe0 && byte <= 0xef) { this.utfRemaining = 2; this.utfCode = byte & 15; this.utfMin = 0x800; }
    else if (byte >= 0xf0 && byte <= 0xf4) { this.utfRemaining = 3; this.utfCode = byte & 7; this.utfMin = 0x10000; }
    else return false;
    const candidates = this.stringCandidates();
    return candidates === undefined || candidates.some((candidate) => candidate.startsWith(this.lexeme!.text) && candidate.length > this.lexeme!.text.length && candidate.codePointAt(this.lexeme!.text.length)! >>> (this.utfRemaining * 6) === this.utfCode);
  }
  private stringCandidates(): string[] | undefined {
    const token = this.lexeme!;
    const frame = this.frames.at(-1);
    return token.key && frame?.schema.type === 'object'
      ? Object.keys(frame.schema.properties ?? {}).filter((key) => !Object.hasOwn(frame.value, key))
      : token.schema.enum?.filter((value): value is string => typeof value === 'string');
  }
  private stringPrefix(): boolean {
    const token = this.lexeme!;
    if (token.schema.maxLength !== undefined && Array.from(token.text).length > token.schema.maxLength) return false;
    const candidates = this.stringCandidates();
    return candidates === undefined || candidates.some((value) => value.startsWith(token.text));
  }
  private valid(schema: GrammarSchema, value: unknown): boolean {
    if (schema.enum && !schema.enum.some((candidate) => canonical(candidate) === canonical(value))) return false;
    if (typeof value === 'string') { const length = Array.from(value).length; if (length < (schema.minLength ?? 0) || length > (schema.maxLength ?? Infinity)) return false; }
    if (schema.type === 'integer' && (typeof value !== 'number' || !Number.isInteger(value))) return false;
    return true;
  }
  private finish(value: unknown, schema: GrammarSchema): boolean {
    if (!this.valid(schema, value)) return false;
    const frame = this.frames.at(-1);
    if (!frame) { this.rootDone = true; return true; }
    if (Array.isArray(frame.value)) {
      if (frame.schema.uniqueItems && frame.value.some((item) => canonical(item) === canonical(value))) return false;
      frame.value.push(value);
    } else { if (frame.key === undefined) return false; Object.defineProperty(frame.value, frame.key, { value, enumerable: true, configurable: true, writable: true }); }
    frame.phase = 'comma'; return true;
  }
  private finishNumber(): boolean {
    const token = this.lexeme!;
    if (!/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(token.text)) return false;
    const value = Number(token.text);
    if (!Number.isFinite(value) || !this.finish(value, token.schema)) return false;
    this.lexeme = undefined; return true;
  }
  private close(frame: Frame): boolean {
    if (Array.isArray(frame.value)) { if (frame.value.length < (frame.schema.minItems ?? 0)) return false; }
    else if (frame.schema.required?.some((key) => !Object.hasOwn(frame.value, key))) return false;
    this.frames.pop(); return this.finish(frame.value, frame.schema);
  }
  private start(character: string, schema: GrammarSchema): boolean {
    const parent = this.frames.at(-1);
    if (schema.enum && parent?.schema.uniqueItems && Array.isArray(parent.value)) {
      const usedValues = parent.value;
      const remaining = schema.enum.filter((candidate) => !usedValues.some((used) => canonical(candidate) === canonical(used)));
      if (!remaining.length) return false;
      schema = { ...schema, enum: remaining };
    }
    const accepts = (type: GrammarSchema['type']) => schema.type === 'any' || schema.type === type;
    if (character === '{' && accepts('object')) { this.frames.push({ schema, value: {}, phase: 'key' }); return true; }
    if (character === '[' && accepts('array')) { this.frames.push({ schema, value: [], phase: 'value' }); return true; }
    if (character === '"' && accepts('string')) { this.lexeme = { kind: 'string', schema, text: '', key: false, escape: false, unicode: null }; return this.stringPrefix(); }
    if ((character === '-' || /\d/.test(character)) && (accepts('number') || accepts('integer'))) {
      if (schema.enum && !schema.enum.some((value) => JSON.stringify(value).startsWith(character))) return false;
      this.lexeme = { kind: 'number', schema, text: character, key: false, escape: false, unicode: null }; return true;
    }
    const literal = character === 't' && accepts('boolean') ? 'true' : character === 'f' && accepts('boolean') ? 'false' : character === 'n' && accepts('null') ? 'null' : undefined;
    if (!literal || schema.enum && !schema.enum.some((value) => JSON.stringify(value) === literal)) return false;
    this.lexeme = { kind: 'literal', schema, text: character, key: false, escape: false, unicode: null, literal }; return true;
  }
  private character(character: string): boolean {
    const token = this.lexeme;
    if (token?.kind === 'string') {
      if (token.unicode !== null) {
        if (!/^[\da-fA-F]$/.test(character)) return false;
        token.unicode += character;
        const candidates = this.stringCandidates();
        if (candidates && !candidates.some((candidate) => candidate.startsWith(token.text) && candidate.length > token.text.length && candidate.charCodeAt(token.text.length).toString(16).padStart(4, '0').startsWith(token.unicode!.toLowerCase()))) return false;
        if (token.unicode.length === 4) { token.text += String.fromCharCode(parseInt(token.unicode, 16)); token.unicode = null; return this.stringPrefix(); }
        return true;
      }
      if (token.escape) {
        token.escape = false;
        if (character === 'u') { token.unicode = ''; return true; }
        const escaped: Record<string, string> = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
        if (!Object.hasOwn(escaped, character)) return false;
        token.text += escaped[character]; return this.stringPrefix();
      }
      if (character === '\\') {
        const candidates = this.stringCandidates();
        if (candidates && !candidates.some((candidate) => candidate.startsWith(token.text) && candidate.length > token.text.length)) return false;
        if (token.schema.maxLength !== undefined && Array.from(token.text).length >= token.schema.maxLength && !/[\uD800-\uDBFF]$/.test(token.text)) return false;
        token.escape = true; return true;
      }
      if (character === '"') {
        this.lexeme = undefined;
        if (token.key) { const frame = this.frames.at(-1)!; if (Object.hasOwn(frame.value, token.text)) return false; if (frame.schema.type === 'object' && !Object.hasOwn(frame.schema.properties ?? {}, token.text)) return false; frame.key = token.text; frame.phase = 'colon'; return true; }
        return this.finish(token.text, token.schema);
      }
      if (character.codePointAt(0)! < 32) return false;
      token.text += character; return this.stringPrefix();
    }
    if (token?.kind === 'literal') {
      if (character !== token.literal![token.text.length]) return false;
      token.text += character;
      if (token.text === token.literal) { this.lexeme = undefined; return this.finish(JSON.parse(token.text), token.schema); }
      return true;
    }
    if (token?.kind === 'number') {
      if (/[\deE+.-]/.test(character)) {
        const next = token.text + character;
        if (!/^-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$/.test(next) || /^-?0\d/.test(next) || /^-?\./.test(next) || /(?:^-?$|\.)[eE]/.test(next)) return false;
        if (token.schema.enum && !token.schema.enum.some((value) => JSON.stringify(value).startsWith(next))) return false;
        token.text = next; return true;
      }
      if (!this.finishNumber()) return false;
      return this.character(character);
    }
    if (/^[ \t\r\n]$/.test(character)) return true;
    if (this.rootDone) return false;
    const frame = this.frames.at(-1);
    if (!frame) return this.start(character, this.schema);
    if (Array.isArray(frame.value)) {
      const values = frame.value;
      if (frame.phase === 'comma') {
        if (character === ']') return this.close(frame);
        if (character !== ',' || frame.value.length >= (frame.schema.maxItems ?? Infinity)) return false;
        if (frame.schema.uniqueItems && frame.schema.items?.enum?.every((candidate) => values.some((used) => canonical(candidate) === canonical(used)))) return false;
        frame.phase = 'next'; return true;
      }
      if (character === ']' && frame.phase === 'value') return this.close(frame);
      if (frame.value.length >= (frame.schema.maxItems ?? Infinity)) return false;
      return this.start(character, frame.schema.items ?? anySchema);
    }
    if (frame.phase === 'comma') {
      if (character === '}') return this.close(frame);
      if (character !== ',' || frame.schema.type === 'object' && Object.keys(frame.schema.properties ?? {}).every((key) => Object.hasOwn(frame.value, key))) return false;
      frame.phase = 'next'; return true;
    }
    if (frame.phase === 'key' || frame.phase === 'next') {
      if (character === '}' && frame.phase === 'key') return this.close(frame);
      if (character !== '"') return false;
      this.lexeme = { kind: 'string', schema: anySchema, text: '', key: true, escape: false, unicode: null }; return this.stringPrefix();
    }
    if (frame.phase === 'colon') { if (character !== ':') return false; frame.phase = 'value'; return true; }
    return this.start(character, frame.schema.properties?.[frame.key!] ?? anySchema);
  }
}
const utfLeadBytes = [0, 0xc2, 0xe1, 0xf1, 0xe0, 0xed, 0xf0, 0xf4] as const;
interface TrieNode { children: Map<number, TrieNode>; ids: number[]; }
export class TokenByteTrie {
  readonly root: TrieNode = { children: new Map(), ids: [] };
  readonly pieces = new Map<number, Uint8Array>();
  private readonly structuralRoot: TrieNode = { children: new Map(), ids: [] };
  private readonly structuralIds = new Set<number>();
  private readonly continuations = new Map<number, number[]>();
  constructor(pieces: Iterable<readonly [number, Uint8Array]>) {
    for (const [id, bytes] of pieces) {
      if (!bytes.length) continue;
      this.pieces.set(id, bytes);
      let structural = false;
      for (const byte of bytes) if (byte < 32 || byte === 34 || byte === 92) { structural = true; break; }
      if (structural) this.structuralIds.add(id);
      let structuralNode = structural ? this.structuralRoot : undefined;
      let node = this.root;
      for (const byte of bytes) {
        let child = node.children.get(byte);
        if (!child) { child = { children: new Map(), ids: [] }; node.children.set(byte, child); }
        node = child;
        if (structuralNode) {
          let child = structuralNode.children.get(byte);
          if (!child) { child = { children: new Map(), ids: [] }; structuralNode.children.set(byte, child); }
          structuralNode = child;
        }
      }
      node.ids.push(id);
      structuralNode?.ids.push(id);
    }
  }
  allowed(state: JsonGrammarState, deliver: (id: number) => void): void {
    const continuation = state.freeStringClass;
    if (continuation !== null) {
      let allowed = this.continuations.get(continuation);
      if (!allowed) {
        const seed = new JsonGrammarState({ type: 'string' });
        seed.pushByte(34);
        if (continuation) seed.pushByte(utfLeadBytes[continuation]!);
        allowed = [];
        for (const [id, bytes] of this.pieces) if (!this.structuralIds.has(id) && seed.clone().push(bytes)) allowed.push(id);
        this.continuations.set(continuation, allowed);
      }
      for (const id of allowed) deliver(id);
    }
    const visit = (node: TrieNode, current: JsonGrammarState) => {
      for (const id of node.ids) deliver(id);
      for (const [byte, child] of node.children) { const next = current.clone(); if (next.pushByte(byte)) visit(child, next); }
    };
    visit(continuation === null ? this.root : this.structuralRoot, state);
  }
}
export interface ByteLevelTokenizer {
  readonly _tokenizerJSON: { decoder?: unknown; added_tokens?: readonly { id: number; content: string; special?: boolean }[] };
  readonly all_special_ids: readonly number[];
  get_vocab(): Iterable<readonly [string, number]>;
}
/** ByteLevel is inspected, not inferred from a lossy one-token UTF-8 decode. */
export function tokenizerByteTrie(tokenizer: ByteLevelTokenizer): TokenByteTrie {
  const decoder: unknown = tokenizer._tokenizerJSON?.decoder;
  if (typeof decoder !== 'object' || decoder === null || !('type' in decoder) || decoder.type !== 'ByteLevel') throw new NekoError('Constrained decoding requires the pinned ByteLevel tokenizer decoder', 'preprocess', 'SCHEMA_UNSUPPORTED');
  const visible = Array.from({ length: 256 }, (_, byte) => byte).filter((byte) => byte >= 33 && byte <= 126 || byte >= 161 && byte <= 172 || byte >= 174);
  const mapping = new Map<string, number>();
  for (const byte of visible) mapping.set(String.fromCharCode(byte), byte);
  let code = 256;
  for (let byte = 0; byte < 256; byte++) if (!visible.includes(byte)) mapping.set(String.fromCharCode(code++), byte);
  const specials = new Set(tokenizer.all_special_ids);
  const added = new Map<number, string>();
  for (const token of tokenizer._tokenizerJSON.added_tokens ?? []) if (!token.special) added.set(token.id, token.content);
  const pieces: [number, Uint8Array][] = [];
  for (const [token, id] of tokenizer.get_vocab()) {
    if (specials.has(id)) continue;
    if (added.has(id)) { pieces.push([id, new TextEncoder().encode(added.get(id)!)]); continue; }
    const bytes = Array.from(token, (character) => mapping.get(character));
    if (bytes.some((byte) => byte === undefined)) throw new NekoError('Tokenizer vocabulary is incompatible with ByteLevel decoding', 'preprocess', 'SCHEMA_UNSUPPORTED');
    pieces.push([id, Uint8Array.from(bytes as number[])]);
  }
  return new TokenByteTrie(pieces);
}
interface Logits { dims: number[]; data: ArrayLike<number | bigint>; }
interface GrammarProcessor { <T extends Logits>(ids: bigint[][], logits: T): T; _call<T extends Logits>(ids: bigint[][], logits: T): T; }
class GrammarProcessorImplementation extends LogitsProcessor {
  private readonly state: JsonGrammarState;
  private consumed: number;
  private mask?: Uint8Array;
  constructor(schema: GrammarSchema, private readonly trie: TokenByteTrie, promptTokens: number, private readonly eos: readonly number[]) { super(); this.state = new JsonGrammarState(schema); this.consumed = promptTokens; }
  override _call<T extends Logits>(ids: bigint[][], logits: T): T {
    if (ids.length !== 1) throw new NekoError('Constrained decoding supports one sequence at a time', 'generate', 'STRUCTURED_OUTPUT');
    const sequence = ids[0]!;
    for (; this.consumed < sequence.length; this.consumed++) {
      const bytes = this.trie.pieces.get(Number(sequence[this.consumed]));
      if (!bytes || !this.state.push(bytes)) throw new NekoError('Model emitted a token outside the JSON grammar', 'generate', 'STRUCTURED_OUTPUT');
    }
    const width = logits.dims.at(-1)!;
    this.mask ??= new Uint8Array(width); this.mask.fill(0);
    this.trie.allowed(this.state, (id) => { if (id < width) this.mask![id] = 1; });
    if (this.state.complete) for (const id of this.eos) if (id < width) this.mask[id] = 1;
    const data = logits.data as Float32Array;
    let possible = false;
    for (let id = 0; id < width; id++) { if (!this.mask[id]) data[id] = -Infinity; else if (Number.isFinite(data[id])) possible = true; }
    if (!possible) throw new NekoError('No token can extend the constrained JSON output', 'generate', 'STRUCTURED_OUTPUT');
    return logits;
  }
}
export const JsonGrammarProcessor: new (schema: GrammarSchema, trie: TokenByteTrie, promptTokens: number, eos: readonly number[]) => GrammarProcessor = GrammarProcessorImplementation;
