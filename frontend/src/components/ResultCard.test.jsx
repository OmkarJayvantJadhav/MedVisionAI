import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { explainResponse } from "../test/fixtures.js";
import ResultCard from "./ResultCard.jsx";

const rejected = {
  ...explainResponse,
  status: "rejected",
  risk_flag: "uncertain",
  reasons: ["too_blurry"],
  calibrated: true,
};

describe("ResultCard", () => {
  it("leads with the referral decision, then the evidence, details and disclaimer", () => {
    render(<ResultCard result={explainResponse} />);

    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading).toHaveTextContent("Refer: possible malignancy");
    expect(screen.getByText("Refer")).toBeInTheDocument();
    expect(screen.getByText(explainResponse.summary)).toBeInTheDocument();
    expect(screen.getByText("Assessment complete")).toBeInTheDocument();

    const gauge = screen.getByRole("meter", { name: "Combined malignancy probability" });
    expect(gauge).toHaveAttribute("aria-valuenow", "74");
    const top = screen.getByRole("list", { name: "Top three model classes" });
    expect(within(top).getAllByRole("meter")).toHaveLength(3);
    expect(within(top).getAllByRole("meter")[0]).toHaveAttribute("aria-valuenow", "71");

    expect(screen.getByText(explainResponse.model_version)).toBeInTheDocument();
    expect(screen.getByText("74 ms")).toBeInTheDocument();
    expect(screen.getByText(/uncalibrated model scores/)).toBeInTheDocument();
    expect(screen.getByText(/Not a medical diagnosis/)).toBeInTheDocument();
  });

  it("shows the referral threshold only when thresholds are fitted", () => {
    const fitted = { ...explainResponse, thresholds_fitted: true };
    const { rerender } = render(
      <ResultCard result={fitted} malignantThreshold={0.19} malignantClasses={["Melanoma"]} />,
    );
    expect(screen.getByText(/Referral threshold 19%/)).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Combined malignancy probability" })).toHaveAttribute(
      "aria-valuetext",
      "74%, at or above the referral threshold of 19%",
    );
    expect(screen.getByText(/Sum of Melanoma/)).toBeInTheDocument();

    rerender(<ResultCard result={explainResponse} malignantThreshold={0.19} />);
    expect(screen.queryByText(/Referral threshold 19%/)).not.toBeInTheDocument();
    expect(screen.getByText(/thresholds have not been fitted/)).toBeInTheDocument();

    rerender(<ResultCard result={fitted} malignantThreshold={null} />);
    expect(screen.getByText(/Referral threshold unavailable/)).toBeInTheDocument();
  });

  it("supports all four outcomes, each with its own decision label", () => {
    const outcomes = [
      [explainResponse, "Refer", "Refer: possible malignancy"],
      [
        { ...explainResponse, status: "low_confidence", risk_flag: "uncertain" },
        "Unsure",
        "Unsure: low confidence",
      ],
      [{ ...explainResponse, status: "ok", risk_flag: "benign" }, "No flag", "No malignancy flag"],
      [rejected, "Not assessed", "Not assessed: image rejected"],
    ];
    for (const [result, decision, title] of outcomes) {
      const { unmount } = render(<ResultCard result={result} />);
      expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(title);
      expect(screen.getByText(decision, { selector: "span" })).toBeInTheDocument();
      unmount();
    }
  });

  it("never calls a no-flag result benign, healthy or cancer-free", () => {
    render(<ResultCard result={{ ...explainResponse, status: "ok", risk_flag: "benign" }} />);
    const decision = screen.getByRole("heading", { level: 2 }).closest("div").parentElement;
    expect(decision).toHaveTextContent(/does not rule out malignancy/);
    expect(decision).not.toHaveTextContent(/healthy|cancer-free|benign/i);
  });

  it("says why a rejected image was not assessed and hides the scores", () => {
    render(<ResultCard result={rejected} />);

    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Not assessed: image rejected",
    );
    expect(screen.getByText("The photo is too blurry.")).toBeInTheDocument();
    expect(screen.queryAllByRole("meter")).toHaveLength(0);
    expect(screen.queryByText(/uncalibrated/)).not.toBeInTheDocument();
  });
});
