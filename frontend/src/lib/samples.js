/**
 * Example images from the ISIC 2019 held-out test split (never used in
 * training), resized to 640 px. ISIC Archive, CC BY-NC 4.0; see
 * docs/DATASETS.md for the full attribution.
 */
export const SAMPLES = [
  {
    id: "melanoma",
    label: "Melanoma",
    file: "/samples/melanoma.jpg",
    isic: "ISIC_0000140",
  },
  {
    id: "basal-cell-carcinoma",
    label: "Basal cell carcinoma",
    file: "/samples/basal-cell-carcinoma.jpg",
    isic: "ISIC_0024885",
  },
  {
    id: "nevus",
    label: "Benign mole (nevus)",
    file: "/samples/nevus.jpg",
    isic: "ISIC_0000082",
  },
];

/** Fetch a sample image and wrap it as a File, like a user upload. */
export async function loadSample(sample, { signal } = {}) {
  const response = await fetch(sample.file, { signal });
  if (!response.ok) throw new Error(`Could not load the sample image (${response.status}).`);
  return new File([await response.arrayBuffer()], `${sample.id}.jpg`, { type: "image/jpeg" });
}
