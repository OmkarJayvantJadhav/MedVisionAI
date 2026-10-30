/**
 * Published test-split results, keyed by the exact model version that `/meta`
 * reports. Copied from docs/model_card.md, where `make evaluate` regenerates
 * them. The page shows a version's results only when that version is served,
 * so it never states a number that belongs to a different model.
 */
export const PUBLISHED_RESULTS = {
  "skin-effb0-v2.2.0": {
    testImages: 2527,
    malignantRecall: 0.904,
    malignantPrecision: 0.681,
    referralRate: 0.49,
    ece: 0.026,
    confidentShare: 0.59,
    confidentAccuracy: 0.869,
    lesionsAccepted: 0.984,
    photosRejected: 0.996,
    accuracy: 0.708,
    macroF1: 0.508,
  },
};

export function publishedResults(modelVersion) {
  return PUBLISHED_RESULTS[modelVersion] ?? null;
}
