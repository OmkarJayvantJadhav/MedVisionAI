/**
 * Minimal client for the MedVision API (`/api/v1`).
 *
 * Every failure is raised as an ApiError carrying the backend's stable error
 * `code`, so the UI can explain each case. The response shapes are documented
 * in `openapi.json` (generated from the backend; see docs/api.md).
 */

export const API_BASE = "/api/v1";

export class ApiError extends Error {
  constructor(code, message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError("network_error", "Could not reach the server.");
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON response (for example a proxy error page); handled below.
  }

  if (!response.ok) {
    const code = body?.error?.code ?? "http_error";
    const message = body?.error?.message ?? `Request failed with status ${response.status}.`;
    throw new ApiError(code, message, response.status);
  }
  if (body === null) {
    throw new ApiError("invalid_response", "The server returned an unexpected response.");
  }
  return body;
}

/** Model version, classes, thresholds and the upload limit. */
export function getMeta({ signal } = {}) {
  return request("/meta", { signal });
}

/**
 * Classify an image and return the prediction plus Grad-CAM heatmap and overlay
 * (base64 PNGs). One call gives everything the results view needs.
 */
export function explain(file, { className, signal } = {}) {
  const form = new FormData();
  form.append("file", file);
  const query = className ? `?class_name=${encodeURIComponent(className)}` : "";
  return request(`/explain${query}`, { method: "POST", body: form, signal });
}
