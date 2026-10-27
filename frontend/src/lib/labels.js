/** Human-readable labels for API values. Keep in sync with the enums in openapi.json. */

export const STATUS = {
  ok: {
    label: "Assessment complete",
    tone: "positive",
  },
  low_confidence: {
    label: "Low confidence",
    tone: "caution",
  },
  rejected: {
    label: "Image not usable",
    tone: "neutral",
  },
};

export const RISK = {
  malignant_suspected: {
    label: "Malignancy suspected",
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
      title: "Not a dermoscopy image: not assessed",
      guidance:
        "The model only assesses dermoscopy images of a single lesion, taken with a dermatoscope. Clinical and phone photos cannot be assessed, so base the referral decision on your clinical assessment.",
    };
  }
  if (result.status === "rejected") {
    return {
      tone: "neutral",
      title: "This image could not be assessed",
      guidance:
        "Retake the dermoscopy image in focus and evenly lit, with the lesion filling most of the frame, then try again.",
    };
  }
  if (result.risk_flag === "malignant_suspected") {
    return {
      tone: "danger",
      title: "Refer: possible malignancy",
      guidance:
        "The combined probability of malignant and pre-malignant lesion types passed the referral threshold. Consider referral to a dermatologist; the decision rests on your clinical assessment.",
    };
  }
  if (result.status === "low_confidence") {
    return {
      tone: "caution",
      title: "Unsure: the model cannot decide",
      guidance:
        "Retake the image if its quality is poor. If the lesion is new, changing, bleeding or otherwise concerning, refer on clinical grounds whatever the result.",
    };
  }
  return {
    tone: "positive",
    title: "No malignancy flag",
    guidance:
      "The referral threshold was not reached. This does not rule out malignancy, including cancers outside the eight lesion types the model knows: decide on the history, examination and any change over time.",
  };
}

const FALLBACK = { label: "Unknown", tone: "neutral" };

export function statusInfo(status) {
  return STATUS[status] ?? FALLBACK;
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
