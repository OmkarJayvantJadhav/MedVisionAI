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

/** Why an image was rejected or flagged, in the user's words. */
export const REASONS = {
  too_dark: "The photo is too dark.",
  too_bright: "The photo is overexposed.",
  too_blurry: "The photo is too blurry.",
  too_noisy: "The photo is grainy or noisy.",
  out_of_distribution: "It does not look like the dermoscopy images the model was trained on.",
  low_confidence: "The model's top answer is below its confidence threshold.",
  malignant_probability_above_threshold:
    "Classes that can be cancerous or pre-cancerous together passed the safety threshold.",
};

/**
 * One headline verdict per result, with what to do next. The malignancy flag
 * wins over every other state, so a warning is never hidden behind "unsure".
 */
export function verdict(result) {
  if (result.status === "rejected" && result.reasons?.includes("out_of_distribution")) {
    return {
      tone: "neutral",
      title: "This photo is outside what the model knows",
      guidance:
        "The model only learned from dermoscopy images: magnified, evenly lit close-ups taken with a dermatoscope. Other pictures, including ordinary phone photos of skin, cannot be assessed. Try one of the samples, and see a dermatologist about any spot that worries you.",
    };
  }
  if (result.status === "rejected") {
    return {
      tone: "neutral",
      title: "This photo could not be assessed",
      guidance:
        "Retake it in good light, in focus, with the spot filling most of the frame, then try again.",
    };
  }
  if (result.risk_flag === "malignant_suspected") {
    return {
      tone: "danger",
      title: "Possible malignancy: please see a dermatologist",
      guidance:
        "The model flagged patterns it associates with skin cancer or pre-cancer. Only a clinician can tell what this is, so please book a dermatologist appointment and show them this spot.",
    };
  }
  if (result.status === "low_confidence") {
    return {
      tone: "caution",
      title: "The model is unsure about this photo",
      guidance:
        "Try a sharper, closer, well-lit photo. If the spot is new, changing, bleeding or itching, see a dermatologist whatever the result.",
    };
  }
  return {
    tone: "positive",
    title: "No malignancy flag",
    guidance:
      "The model did not flag signs it associates with skin cancer. This does not rule anything out: see a dermatologist if the spot changes in size, shape or colour.",
  };
}

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
