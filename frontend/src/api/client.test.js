import { afterEach, describe, expect, it, vi } from "vitest";

import { explainResponse, imageFile, jsonResponse, metaResponse } from "../test/fixtures.js";
import { ApiError, explain, getMeta } from "./client.js";

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("posts the image as multipart form data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(explainResponse));
    vi.stubGlobal("fetch", fetchMock);

    const result = await explain(imageFile(), { className: "Benign Keratosis" });

    expect(result.predicted_class).toBe("Melanoma");
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/explain?class_name=Benign%20Keratosis");
    expect(options.method).toBe("POST");
    expect(options.body.get("file").name).toBe("lesion.png");
  });

  it("returns metadata", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(metaResponse)));
    await expect(getMeta()).resolves.toMatchObject({ max_upload_mb: 8 });
  });

  it("raises the backend error code", async () => {
    const body = { error: { code: "rate_limited", message: "Too many requests." } };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(body, 429)));
    await expect(explain(imageFile())).rejects.toMatchObject({
      name: "ApiError",
      code: "rate_limited",
      status: 429,
    });
  });

  it("handles non-JSON errors and network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Bad gateway", { status: 502 })));
    await expect(getMeta()).rejects.toMatchObject({ code: "http_error", status: 502 });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>", { status: 200 })));
    await expect(getMeta()).rejects.toMatchObject({ code: "invalid_response" });

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await getMeta().catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe("network_error");
  });

  it("lets aborts propagate unchanged", async () => {
    const abort = new DOMException("Aborted", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abort));
    await expect(getMeta()).rejects.toBe(abort);
  });
});
