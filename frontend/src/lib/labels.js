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
    "The chance of cancer or pre-cancer passed the alert line.",
};

/**
 * Why a rejected image was not checked, or null for a checked image. A
 * rejected image is never given a lesion type or an alert.
 */
export function rejection(result) {
  if (result.status !== "rejected") return null;
  if (result.reasons?.includes("out_of_distribution")) {
    return {
      title: "Not checked: this is not a dermoscopy image",
      guidance:
        "The model only works with close-up pictures of one skin lesion taken through a dermatoscope. Ordinary photos cannot be checked, so please decide using your own examination.",
    };
  }
  return {
    title: "Not checked: the picture is not clear enough",
    guidance:
      "Take the picture again, in focus and evenly lit, with the lesion filling most of the picture, then try again.",
  };
}

/**
 * The high-risk alert. It is raised from the combined chance of cancer and
 * pre-cancer, whatever the lesion type and however sure the model is, and the
 * result shows it above everything else so it is never hidden (plan §8.9).
 */
export function hasAlert(result) {
  return result.status !== "rejected" && result.risk_flag === "malignant_suspected";
}

export const ALERT = {
  title: "High-risk alert: possible skin cancer",
  guidance:
    "The chance of cancer or pre-cancer is above the alert line, whatever the lesion type below. Consider referral to a dermatologist. The final decision is yours.",
};

export const NO_ALERT =
  "No high-risk alert. This does not rule out cancer, including rare types the model does not know: decide using the patient's history, your examination and any change over time.";

/** How sure the model is about the lesion type. */
export function confidence(result) {
  if (result.status === "low_confidence") {
    return {
      label: "Uncertain",
      tone: "caution",
      note: "The model is not sure which type this is. Treat the match with caution, and if the lesion is new, changing, bleeding or worrying, consider referral anyway.",
    };
  }
  return {
    label: "Confident",
    tone: "positive",
    note: "The closest of the eight lesion types the model knows. This is a comparison, not a diagnosis.",
  };
}

export function errorMessage(error) {
  return ERROR_MESSAGES[error?.code] ?? error?.message ?? "Something went wrong.";
}

export function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}
