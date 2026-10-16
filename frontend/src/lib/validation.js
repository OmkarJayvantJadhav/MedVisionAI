export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const DEFAULT_MAX_MB = 8;

/**
 * Check a file before uploading. The server validates again (by content, not
 * by type), so this only gives faster feedback.
 *
 * @returns {string | null} an error message, or null when the file looks fine
 */
export function validateFile(file, maxMb = DEFAULT_MAX_MB) {
  if (!file) return "Please choose an image.";
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return "Only JPEG, PNG and WebP images are supported.";
  }
  if (file.size > maxMb * 1024 * 1024) {
    return `The image is larger than ${maxMb} MB. Please use a smaller photo.`;
  }
  return null;
}
