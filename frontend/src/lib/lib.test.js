import { describe, expect, it } from "vitest";

import { imageFile } from "../test/fixtures.js";
import { confidence, errorMessage, formatPercent, hasAlert, rejection } from "./labels.js";
import { validateFile } from "./validation.js";

describe("labels", () => {
  it("explains error codes, falling back to the server message", () => {
    expect(errorMessage({ code: "image_too_large" })).toMatch(/too large/);
    expect(errorMessage({ code: "brand_new", message: "Server says no." })).toBe("Server says no.");
    expect(errorMessage(null)).toBe("Something went wrong.");
  });

  it("raises the alert from the cancer flag whatever the confidence, never for a rejected image", () => {
    expect(hasAlert({ status: "ok", risk_flag: "malignant_suspected" })).toBe(true);
    expect(hasAlert({ status: "low_confidence", risk_flag: "malignant_suspected" })).toBe(true);
    expect(hasAlert({ status: "ok", risk_flag: "benign" })).toBe(false);
    expect(hasAlert({ status: "low_confidence", risk_flag: "uncertain" })).toBe(false);
    expect(hasAlert({ status: "rejected", risk_flag: "malignant_suspected" })).toBe(false);
  });

  it("labels the lesion type as confident or uncertain", () => {
    expect(confidence({ status: "ok" }).label).toBe("Confident");
    expect(confidence({ status: "low_confidence" }).label).toBe("Uncertain");
  });

  it("explains that unfamiliar photos cannot be checked, rather than asking for a retake", () => {
    expect(rejection({ status: "ok" })).toBeNull();
    const blurry = rejection({ status: "rejected", reasons: ["too_blurry"] });
    const unfamiliar = rejection({ status: "rejected", reasons: ["out_of_distribution"] });
    expect(blurry.title).toBe("Not checked: the picture is not clear enough");
    expect(unfamiliar.title).toBe("Not checked: this is not a dermoscopy image");
    expect(unfamiliar.guidance).toMatch(/dermatoscope/);
  });

  it("formats percentages", () => {
    expect(formatPercent(0.714)).toBe("71%");
    expect(formatPercent(0)).toBe("0%");
  });
});

describe("validateFile", () => {
  it("accepts JPEG, PNG and WebP within the limit", () => {
    for (const type of ["image/jpeg", "image/png", "image/webp"]) {
      expect(validateFile(imageFile("x", type))).toBeNull();
    }
  });

  it("rejects missing, unsupported and oversized files", () => {
    expect(validateFile(null)).toMatch(/choose an image/);
    expect(validateFile(imageFile("x.gif", "image/gif"))).toMatch(/Only JPEG/);
    expect(validateFile(imageFile("big.png", "image/png", 3 * 1024 * 1024), 2)).toMatch(/2 MB/);
  });
});
