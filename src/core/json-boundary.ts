/** Locates only a leading JSON value boundary; JSON.parse remains the syntax authority. */
export class JsonBoundary {
  private kind?: 'container' | 'string' | 'literal' | 'number';
  private readonly closers: string[] = [];
  private quoted = false;
  private escaped = false;
  private literal = '';
  private literalIndex = 0;
  complete = false;
  invalid = false;
  push(text: string): void {
    for (const character of text) {
      if (this.invalid) return;
      if (this.complete) { if (!/[\x20\t\r\n]/.test(character)) this.invalid = true; continue; }
      if (!this.kind) {
        if (/[\x20\t\r\n]/.test(character)) continue;
        if (character === '{' || character === '[') { this.kind = 'container'; this.closers.push(character === '{' ? '}' : ']'); continue; }
        if (character === '"') { this.kind = 'string'; this.quoted = true; continue; }
        if ('tfn'.includes(character)) { this.kind = 'literal'; this.literal = character === 't' ? 'true' : character === 'f' ? 'false' : 'null'; this.literalIndex = 1; continue; }
        if (character === '-' || /[0-9]/.test(character)) { this.kind = 'number'; continue; }
        this.invalid = true; continue;
      }
      if (this.kind === 'literal') {
        if (character !== this.literal[this.literalIndex++]) this.invalid = true;
        else if (this.literalIndex === this.literal.length) this.complete = true;
        continue;
      }
      if (this.kind === 'number') {
        if (/[\x20\t\r\n]/.test(character)) this.complete = true;
        else if (!/[0-9eE+.\-]/.test(character)) this.invalid = true;
        continue;
      }
      if (this.quoted) {
        if (this.escaped) this.escaped = false;
        else if (character === '\\') this.escaped = true;
        else if (character === '"') { this.quoted = false; if (this.kind === 'string') this.complete = true; }
        continue;
      }
      if (character === '"') this.quoted = true;
      else if (character === '{' || character === '[') this.closers.push(character === '{' ? '}' : ']');
      else if (character === '}' || character === ']') {
        if (this.closers.pop() !== character) this.invalid = true;
        else if (!this.closers.length) this.complete = true;
      }
    }
  }
  /** Numbers have no unambiguous boundary until whitespace or end-of-sequence. */
  end(): void { if (this.kind === 'number' && !this.invalid) this.complete = true; }
}
