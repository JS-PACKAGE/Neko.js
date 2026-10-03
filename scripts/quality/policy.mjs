export const POLICY_VERSION = 'quality-thresholds-v1';

// These are release acceptance targets, not estimates of the current model's ability.
export const QUALITY_POLICY = Object.freeze({
  version: POLICY_VERSION,
  requiredCases: Object.freeze(['text', 'image', 'boundaries', 'hierarchy']),
  claims: Object.freeze({ minPrecision: 1, minRecall: 1, maxContradictions: 0, maxUnsupported: 0, maxFormatErrors: 0 }),
  hierarchyRetained: Object.freeze({ minPrecision: 1, minRecall: 1, maxContradictions: 0, maxUnsupported: 0, maxFormatErrors: 0 }),
  hierarchyGenerated: Object.freeze({ minPrecision: 1, minRecall: 0.9, maxContradictions: 0, maxUnsupported: 0, maxFormatErrors: 0 }),
  hierarchyOverview: Object.freeze({ maxContradictions: 0, maxUnsupported: 0, maxFormatErrors: 0 }),
});
