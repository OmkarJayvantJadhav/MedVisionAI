import { describe, expect, it } from "vitest";

import { imageFile } from "../test/fixtures.js";
import { errorMessage, formatPercent, riskInfo, statusInfo, verdict } from "./labels.js";
import { validateFile } from "./validation.js";

describe("labels", () => {
  it("maps statuses and risk flags to tones", () => {
    expect(statusInfo("ok").tone).toBe("positive");
    expect(statusInfo("rejected").label).toBe("Image not usable");
    expect(riskInfo("malignant_suspected").tone).toBe("danger");
    expect(statusInfo("something_new").label).toBe("Unknown");
    expect(riskInfo(undefined).tone).toBe("neutral");
  });

  it("explains error codes, falling back to the server message", () => {
    expect(errorMessage({ code: "image_too_large" })).toMatch(/too large/);
    expect(errorMessage({ code: "brand_new", message: "Server says no." })).toBe("Server says no.");
    expect(errorMessage(null)).toBe("Something went wrong.");
  });

  it("puts the malignancy flag above every other verdict", () => {
    expect(verdict({ status: "low_confidence", risk_flag: "malignant_suspected" }).tone).toBe(
      "danger",
    );
    expect(verdict({ status: "ok", risk_flag: "malignant_suspected" }).tone).toBe("danger");
    expect(verdict({ status: "low_confidence", risk_flag: "benign" }).tone).toBe("caution");
    expect(verdict({ status: "ok", risk_flag: "benign" }).tone).toBe("positive");
    expect(verdict({ status: "rejected", risk_flag: "uncertain" }).tone).toBe("neutral");
  });

  it("explains that unfamiliar photos cannot be assessed, rather than asking for a retake", () => {
    const blurry = verdict({ status: "rejected", risk_flag: "uncertain", reasons: ["too_blurry"] });
    const unfamiliar = verdict({
      status: "rejected",
      risk_flag: "uncertain",
      reasons: ["out_of_distribution"],
    });
    expect(blurry.title).toBe("This image could not be assessed");
    expect(unfamiliar.title).toBe("Not a dermoscopy image: not assessed");
    expect(unfamiliar.guidance).toMatch(/dermoscopy/);
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
