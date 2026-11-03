/** Plain-language wording for API values. Keep in sync with the enums in openapi.json. */

const ERROR_MESSAGES = {
  invalid_image: "That file could not be opened as a picture. Please choose a JPEG, PNG or WebP.",
  unsupported_image_type: "Only JPEG, PNG and WebP pictures can be checked.",
  image_too_large: "The picture is too large. Please use a smaller one.",
  unknown_class: "Something went wrong. Please try again.",
  validation_error: "No picture was received. Please choose one and try again.",
  rate_limited: "Too many checks in a short time. Please wait a minute and try again.",
  model_unavailable: "The system is still starting up. Please try again in a minute.",
  network_error: "Could not connect. Check your internet connection and try again.",
  internal_error: "Something went wrong on our side. Please try again.",
};

/** Why an image could not be checked, in plain words. */
export const REASONS = {
  too_dark: "The picture is too dark.",
  too_bright: "The picture is too bright.",
  too_blurry: "The picture is blurry.",
  too_noisy: "The picture is grainy.",
  out_of_distribution: "It does not look like a dermoscopy image.",
  low_confidence: "The system is not sure which type of lesion this is.",
  malignant_probability_above_threshold:
    "The chance of cancer or pre-cancer passed the referral line.",
};

/**
 * The referral answer for a result: a short decision, a headline and what it
 * means for the clinician. A rejected image is never given a score-based
 * answer; otherwise the cancer flag wins over "unsure", so a warning is never
 * hidden behind uncertainty (plan §8.9).
 */
export function verdict(result) {
  if (result.status === "rejected" && result.reasons?.includes("out_of_distribution")) {
    return {
      tone: "neutral",
      decision: "Not checked",
      title: "Not checked: this is not a dermoscopy image",
      guidance:
        "The system only works with close-up pictures of one skin lesion taken through a dermatoscope. Ordinary photos cannot be checked, so please decide using your own examination.",
    };
  }
  if (result.status === "rejected") {
    return {
      tone: "neutral",
      decision: "Not checked",
      title: "Not checked: the picture is not clear enough",
      guidance:
        "Take the picture again, in focus and evenly lit, with the lesion filling most of the picture, then try again.",
    };
  }
  if (result.risk_flag === "malignant_suspected") {
    return {
      tone: "danger",
      decision: "Refer",
      title: "Refer: possible skin cancer",
      guidance:
        "The chance of cancer or pre-cancer is above the referral line. Consider referring the patient to a dermatologist. The final decision is yours.",
    };
  }
  if (result.status === "low_confidence") {
    return {
      tone: "caution",
      decision: "Unsure",
      title: "Unsure: the system is not confident",
      guidance:
        "The system could not match this lesion with confidence. If the picture is poor, take it again. If the lesion is new, changing, bleeding or worrying, refer it anyway.",
    };
  }
  return {
    tone: "positive",
    decision: "No flag",
    title: "No flag raised",
    guidance:
      "The chance of cancer or pre-cancer is below the referral line. This does not rule out cancer, including rare types the system does not know. Decide using the patient's history, your examination and any change over time.",
  };
}

export function errorMessage(error) {
  return ERROR_MESSAGES[error?.code] ?? error?.message ?? "Something went wrong.";
}

export function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}
