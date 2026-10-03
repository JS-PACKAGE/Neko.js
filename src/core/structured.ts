import { Validator, type Schema } from '@cfworker/json-schema';
import { NekoError } from '../errors.js';
import type { GrammarSchema } from './json-grammar.js';
export type StructuredMode = 'constrained' | 'validation-only';
type RequiredKeys<S> = S extends { required: readonly (infer K)[] } ? Extract<K, string> : never;
type ObjectValue<P, R extends string> = { [K in keyof P as K extends R ? K : never]-?: SchemaValue<P[K]> } & { [K in keyof P as K extends R ? never : K]?: SchemaValue<P[K]> };
/** Types follow the supplied schema, never a caller-selected arbitrary result generic. */
export type SchemaValue<S> =
  S extends { const: infer C } ? C :
  S extends { enum: readonly (infer E)[] } ? E :
  S extends { type: 'string' } ? string :
  S extends { type: 'number' | 'integer' } ? number :
  S extends { type: 'boolean' } ? boolean :
  S extends { type: 'null' } ? null :
  S extends { type: 'array'; items: infer I } ? SchemaValue<I>[] :
  S extends { type: 'object'; properties: infer P } ? ObjectValue<P, RequiredKeys<S>> : unknown;
export interface CompiledStructuredSchema<S = unknown> {
  json: string;
  mode: StructuredMode;
  grammar?: GrammarSchema;
  /** Retains the schema type without trusting generated data. */
  readonly schema: S;
  validator: { validate(value: unknown): { valid: boolean; errors: { error: string; keyword: string; keywordLocation: string; instanceLocation: string }[] } };
}

export function validateStructuredValue<S>(compiled: CompiledStructuredSchema<S>, value: unknown): SchemaValue<S> {
  const validation = compiled.validator.validate(value);
  if (!validation.valid) throw new NekoError('Generated JSON does not match the schema', 'generate', 'STRUCTURED_OUTPUT');
  // The runtime validator above establishes the schema-derived type.
  return value as SchemaValue<S>;
}

function constrainedSchema(schema: unknown): GrammarSchema {
  if (schema === true) return { type: 'any' };
  if (typeof schema !== 'object' || schema === null || Array.isArray(schema)) throw new Error('Constrained mode requires a typed schema or true');
  const node = schema as Record<string, unknown>;
  const annotations: Record<string, true> = { $schema: true, $id: true, $comment: true, title: true, description: true, default: true, examples: true, readOnly: true, writeOnly: true };
  const constraints: Record<string, readonly string[]> = {
    object: ['type', 'properties', 'required', 'additionalProperties'],
    array: ['type', 'items', 'minItems', 'maxItems', 'uniqueItems'],
    string: ['type', 'minLength', 'maxLength', 'enum', 'const'],
    number: ['type', 'enum', 'const'], integer: ['type', 'enum', 'const'],
    boolean: ['type', 'enum', 'const'], null: ['type', 'enum', 'const'],
  };
  if (typeof node.type !== 'string' || !Object.hasOwn(constraints, node.type)) throw new Error('Constrained mode requires one explicit supported type');
  for (const key of Object.keys(node)) if (!Object.hasOwn(annotations, key) && !constraints[node.type]!.includes(key)) throw new Error(`Constrained mode does not support keyword ${key}; use structuredMode:"validation-only" explicitly`);
  const type = node.type as GrammarSchema['type'];
  const grammar: GrammarSchema = { type };
  if (type === 'object') {
    if (node.additionalProperties !== false) throw new Error('Constrained objects require additionalProperties:false');
    const properties = (node.properties ?? {}) as Record<string, unknown>;
    grammar.properties = Object.fromEntries(Object.entries(properties).map(([key, child]) => [key, constrainedSchema(child)]));
    grammar.required = (node.required ?? []) as string[];
    if (grammar.required.some((key) => !Object.hasOwn(properties, key))) throw new Error('Required properties must have declared schemas');
  }
  if (type === 'array') {
    if (node.items === undefined || Array.isArray(node.items)) throw new Error('Constrained arrays require one items schema');
    grammar.items = constrainedSchema(node.items);
    grammar.minItems = node.minItems as number | undefined;
    grammar.maxItems = node.maxItems as number | undefined;
    grammar.uniqueItems = node.uniqueItems as boolean | undefined;
    if ((grammar.minItems ?? 0) > (grammar.maxItems ?? Infinity)) throw new Error('Array bounds are unsatisfiable');
  }
  if (type === 'string') {
    grammar.minLength = node.minLength as number | undefined;
    grammar.maxLength = node.maxLength as number | undefined;
    if ((grammar.minLength ?? 0) > (grammar.maxLength ?? Infinity)) throw new Error('String bounds are unsatisfiable');
  }
  if (node.enum !== undefined || Object.hasOwn(node, 'const')) {
    grammar.enum = Object.hasOwn(node, 'const') ? [node.const] : node.enum as unknown[];
    if (node.enum !== undefined && Object.hasOwn(node, 'const') && !(node.enum as unknown[]).some((value) => JSON.stringify(value) === JSON.stringify(node.const))) throw new Error('Enum and const constraints are unsatisfiable');
    if (!grammar.enum!.every((value) => type === 'null' ? value === null : type === 'integer' ? typeof value === 'number' && Number.isInteger(value) : typeof value === type)) throw new Error('Enum values must match the explicit primitive type');
    if (type === 'string') {
      grammar.enum = grammar.enum!.filter((value) => Array.from(value as string).length >= (grammar.minLength ?? 0) && Array.from(value as string).length <= (grammar.maxLength ?? Infinity));
      if (!grammar.enum.length) throw new Error('String enum is unsatisfiable');
    }
  }
  return grammar;
}

// Draft-07's official meta-schema, bundled so schema checking never needs remote resolution.
const ref = { $ref: '#' };
const integer = { type: 'integer', minimum: 0 };
const stringArray = { type: 'array', items: { type: 'string' }, uniqueItems: true };
const types = { enum: ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] };
const schemaArray = { type: 'array', minItems: 1, items: ref };
const schemaMap = { type: 'object', additionalProperties: ref };
const meta = new Validator({
  $id: 'http://json-schema.org/draft-07/schema#', type: ['object', 'boolean'],
  properties: {
    $id: { type: 'string', format: 'uri-reference' }, $schema: { type: 'string', format: 'uri' }, $ref: { type: 'string', format: 'uri-reference' },
    $comment: { type: 'string' }, title: { type: 'string' }, description: { type: 'string' }, default: true,
    readOnly: { type: 'boolean' }, writeOnly: { type: 'boolean' }, examples: { type: 'array' },
    multipleOf: { type: 'number', exclusiveMinimum: 0 }, maximum: { type: 'number' }, minimum: { type: 'number' }, exclusiveMaximum: { type: 'number' }, exclusiveMinimum: { type: 'number' },
    maxLength: integer, minLength: integer, pattern: { type: 'string', format: 'regex' }, additionalItems: ref,
    items: { anyOf: [ref, schemaArray] }, maxItems: integer, minItems: integer, uniqueItems: { type: 'boolean' }, contains: ref,
    maxProperties: integer, minProperties: integer, required: stringArray, additionalProperties: ref, definitions: schemaMap, properties: schemaMap,
    patternProperties: { ...schemaMap, propertyNames: { format: 'regex' } }, dependencies: { type: 'object', additionalProperties: { anyOf: [ref, stringArray] } },
    propertyNames: ref, const: true, enum: { type: 'array', minItems: 1, uniqueItems: true }, type: { anyOf: [types, { type: 'array', minItems: 1, uniqueItems: true, items: types }] },
    format: { type: 'string' }, contentMediaType: { type: 'string' }, contentEncoding: { type: 'string' }, if: ref, then: ref, else: ref,
    allOf: schemaArray, anyOf: schemaArray, oneOf: schemaArray, not: ref,
  },
} as Schema, '7');
const singles: Record<string, true> = { additionalItems: true, additionalProperties: true, contains: true, propertyNames: true, if: true, then: true, else: true, not: true };
const maps: Record<string, true> = { definitions: true, properties: true, patternProperties: true, dependencies: true };
const arrays: Record<string, true> = { allOf: true, anyOf: true, oneOf: true };
const newer: Record<string, true> = { $defs: true, $anchor: true, $dynamicRef: true, $dynamicAnchor: true, $recursiveRef: true, $recursiveAnchor: true, prefixItems: true, unevaluatedItems: true, unevaluatedProperties: true, dependentSchemas: true, dependentRequired: true, minContains: true, maxContains: true };

export function compileStructuredSchema<S>(input: S, mode: StructuredMode = 'constrained'): CompiledStructuredSchema<S> {
  try {
    if (mode !== 'constrained' && mode !== 'validation-only') throw new Error('structuredMode must be constrained or validation-only');
    // Reject non-JSON values/cycles, and avoid the validator's hidden property mutations on caller data.
    const json = JSON.stringify(input, (_key, value: unknown) => {
      if (typeof value === 'number' && !Number.isFinite(value) || ['undefined', 'function', 'symbol', 'bigint'].includes(typeof value)) throw new Error('Schema must contain only JSON values');
      return value;
    });
    const schema: unknown = JSON.parse(json);
    if (!meta.validate(schema).valid) throw new Error('Schema is not valid Draft-07');
    const refs: string[] = [];
    const visit = (node: unknown): void => {
      if (typeof node === 'boolean') return;
      const object = node as Record<string, unknown>;
      for (const [key, value] of Object.entries(object)) {
        if (Object.hasOwn(newer, key)) throw new Error(`Unsupported non-Draft-07 keyword: ${key}`);
        if (key === '$schema' && value !== 'http://json-schema.org/draft-07/schema#' && value !== 'https://json-schema.org/draft-07/schema#') throw new Error('Only Draft-07 schemas are supported');
        if (key === '$id' && object !== schema) throw new Error('Nested $id changes reference scope; only root-local references are supported');
        if (key === '$ref') {
          if (typeof value !== 'string' || !value.startsWith('#') || !/^(?:\/|$)/.test(decodeURIComponent(value.slice(1)))) throw new Error('Only root-local JSON Pointer references are supported');
          refs.push(value);
        } else if (Object.hasOwn(singles, key)) visit(value);
        else if (key === 'items') { if (Array.isArray(value)) value.forEach(visit); else visit(value); }
        else if (Object.hasOwn(arrays, key)) (value as unknown[]).forEach(visit);
        else if (Object.hasOwn(maps, key)) for (const child of Object.values(value as Record<string, unknown>)) { if (!Array.isArray(child)) visit(child); }
      }
    };
    visit(schema);
    for (const reference of refs) {
      let target: unknown = schema;
      for (const segment of decodeURIComponent(reference.slice(1)).split('/').slice(1)) {
        if (/~(?![01])/.test(segment)) throw new Error('Invalid JSON Pointer escape');
        const key = segment.replaceAll('~1', '/').replaceAll('~0', '~');
        if (typeof target !== 'object' || target === null || !Object.hasOwn(target, key)) throw new Error(`Unresolved local schema reference: ${reference}`);
        target = (target as Record<string, unknown>)[key];
      }
      if (!meta.validate(target).valid) throw new Error('Reference target is not a schema');
    }
    if (typeof schema === 'object' && schema !== null) delete (schema as Record<string, unknown>).$id;
    let grammar: GrammarSchema | undefined;
    if (mode === 'constrained') {
      try { grammar = constrainedSchema(schema); }
      catch (cause) { throw new NekoError(cause instanceof Error ? cause.message : 'Unsupported constrained schema', 'preprocess', 'SCHEMA_UNSUPPORTED', { cause }); }
    }
    return { validator: new Validator(schema as Schema | boolean, '7'), json, schema: input, mode, ...(grammar ? { grammar } : {}) };
  } catch (cause) {
    if (cause instanceof NekoError) throw cause;
    throw new NekoError(cause instanceof Error ? cause.message : 'Invalid JSON schema', 'preprocess', 'SCHEMA_INVALID', { cause });
  }
}
