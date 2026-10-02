import { Validator, type Schema } from '@cfworker/json-schema';
import { NekoError } from '../errors.js';
export interface CompiledStructuredSchema {
  json: string;
  validator: { validate(value: unknown): { valid: boolean; errors: { error: string; keyword: string; keywordLocation: string; instanceLocation: string }[] } };
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

export function compileStructuredSchema(input: unknown): CompiledStructuredSchema {
  try {
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
    return { validator: new Validator(schema as Schema | boolean, '7'), json };
  } catch (cause) { throw new NekoError(cause instanceof Error ? cause.message : 'Invalid JSON schema', 'preprocess', 'SCHEMA_INVALID', { cause }); }
}
