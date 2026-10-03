export interface DiagnosticCapture { maxCharacters: number; }
export type GenerationDiagnosticOptions = true | { capture?: DiagnosticCapture; };
export interface GenerationDiagnostic {
  version: 1;
  stageId: string;
  attempt: number;
  stage: string;
  code: string;
  finishReason: 'stop' | 'length' | 'aborted' | 'error' | 'not-started';
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
  aggregateUsage?: { inputTokens: number; outputTokens: number; totalTokens: number };
  outputCharacters: number;
  json?: { complete: boolean; invalid: boolean; position?: number };
  schema?: { keyword: string; keywordLocation: string; instanceLocation: string }[];
  capture?: { output: string; truncated: boolean; maxCharacters: number };
}
export function validateDiagnosticOptions(options: GenerationDiagnosticOptions | undefined): void {
  if (options === undefined || options === true) return;
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('diagnostics must be true or an options object');
  if (options.capture !== undefined && (!options.capture || !Number.isSafeInteger(options.capture.maxCharacters) || options.capture.maxCharacters < 1 || options.capture.maxCharacters > 65536)) throw new RangeError('Diagnostic capture maxCharacters must be between 1 and 65536');
}
export function getGenerationDiagnostic(error: unknown): GenerationDiagnostic | undefined {
  if (typeof error !== 'object' || error === null || !('diagnostics' in error)) return undefined;
  return error.diagnostics as GenerationDiagnostic | undefined;
}
/** Adds metadata without replacing the original exception or invoking user callbacks. */
export function attachGenerationDiagnostic(error: unknown, diagnostic: GenerationDiagnostic): void {
  if (typeof error === 'object' && error !== null && Object.isExtensible(error)) Object.defineProperty(error, 'diagnostics', { value: diagnostic, enumerable: true, configurable: true });
}
export function jsonErrorPosition(error: unknown): number | undefined {
  if (!(error instanceof SyntaxError)) return undefined;
  const match = /position (\d+)/.exec(error.message);
  return match ? Number(match[1]) : undefined;
}
