/** Human-readable labels for API values. Keep in sync with the enums in openapi.json. */

export const STATUS = {
  ok: {
    label: "Assessment complete",
    tone: "positive",
    detail: "The model produced a confident result.",
  },
  low_confidence: {
    label: "Low confidence",
    tone: "caution",
    detail: "The model is unsure; treat the result with extra care.",
  },
  rejected: {
    label: "Image not usable",
    tone: "neutral",
    detail: "The photo could not be assessed. Try a sharper, well-lit close-up.",
  },
};

export const RISK = {
  malignant_suspected: {
    label: "Possible malignancy: see a dermatologist",
    tone: "danger",
  },
  benign: { label: "No malignancy flag", tone: "positive" },
  uncertain: { label: "Uncertain", tone: "caution" },
};

const ERROR_MESSAGES = {
  invalid_image:
    "That file could not be read as an image. Please choose a JPEG, PNG or WebP photo.",
  unsupported_image_type: "Only JPEG, PNG and WebP images are supported.",
  image_too_large: "The image is too large. Please use a smaller photo.",
  unknown_class: "That class is not known to the model.",
  validation_error: "The request was incomplete. Please choose an image and try again.",
  rate_limited: "Too many requests. Please wait a minute and try again.",
  model_unavailable: "The model is still loading or unavailable. Please try again shortly.",
  network_error: "Could not reach the server. Check your connection and try again.",
  internal_error: "Something went wrong on the server. Please try again.",
};

const FALLBACK = { label: "Unknown", tone: "neutral" };

export function statusInfo(status) {
  return STATUS[status] ?? { ...FALLBACK, detail: "" };
}

export function riskInfo(risk) {
  return RISK[risk] ?? FALLBACK;
}

export function errorMessage(error) {
  return ERROR_MESSAGES[error?.code] ?? error?.message ?? "Something went wrong.";
}

export function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}
