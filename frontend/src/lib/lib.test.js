import { describe, expect, it } from "vitest";

import { imageFile } from "../test/fixtures.js";
import { errorMessage, formatPercent, verdict } from "./labels.js";
import { validateFile } from "./validation.js";

describe("labels", () => {
  it("explains error codes, falling back to the server message", () => {
    expect(errorMessage({ code: "image_too_large" })).toMatch(/too large/);
    expect(errorMessage({ code: "brand_new", message: "Server says no." })).toBe("Server says no.");
    expect(errorMessage(null)).toBe("Something went wrong.");
  });

  it("puts the cancer flag above every other verdict", () => {
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
    expect(blurry.title).toBe("Not checked: the picture is not clear enough");
    expect(unfamiliar.title).toBe("Not checked: this is not a dermoscopy image");
    expect(unfamiliar.guidance).toMatch(/dermatoscope/);
  });

  it("gives every outcome a short decision label", () => {
    expect(verdict({ status: "ok", risk_flag: "malignant_suspected" }).decision).toBe("Refer");
    expect(verdict({ status: "low_confidence", risk_flag: "uncertain" }).decision).toBe("Unsure");
    expect(verdict({ status: "ok", risk_flag: "benign" }).decision).toBe("No flag");
    expect(verdict({ status: "rejected", risk_flag: "uncertain" }).decision).toBe("Not checked");
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
