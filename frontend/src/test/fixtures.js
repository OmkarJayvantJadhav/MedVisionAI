// A tiny valid PNG (1x1), used as the heatmap in fixtures.
export const PNG_1PX =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

export const explainResponse = {
  status: "ok",
  predicted_class: "Melanoma",
  confidence: 0.71,
  top_k: [
    { class_name: "Melanoma", probability: 0.71 },
    { class_name: "Melanocytic Nevus", probability: 0.2 },
    { class_name: "Benign Keratosis", probability: 0.05 },
  ],
  risk_flag: "malignant_suspected",
  malignant_probability: 0.74,
  reasons: ["malignant_probability_above_threshold"],
  model_version: "skin-effb0-v2.0.0",
  calibrated: false,
  thresholds_fitted: false,
  summary: "The image most resembles Melanoma (71%).",
  disclaimer: "Not a medical diagnosis. Please consult a qualified dermatologist.",
  latency_ms: 74.2,
  explained_class: "Melanoma",
  heatmap_png: PNG_1PX,
  overlay_png: PNG_1PX,
};

export const metaResponse = {
  model_version: "skin-effb0-v2.0.0",
  backbone: "efficientnetb0",
  class_names: ["Melanoma", "Melanocytic Nevus", "Benign Keratosis"],
  malignant_classes: ["Melanoma"],
  image_size: 224,
  calibrated: false,
  thresholds: { low_confidence: 0.5, malignant: 0.3, ood_energy: null, fitted: false },
  max_upload_mb: 8,
  disclaimer: "Not a medical diagnosis. Please consult a qualified dermatologist.",
};

export function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export function imageFile(name = "lesion.png", type = "image/png", size = 1024) {
  return new File([new Uint8Array(size)], name, { type });
}
