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
  it("leads with the suggestion, then the evidence and the disclaimer", () => {
    render(<ResultCard result={explainResponse} />);

    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading).toHaveTextContent("Refer: possible skin cancer");
    expect(screen.getByText("Refer")).toBeInTheDocument();
    expect(screen.getByText(explainResponse.summary)).toBeInTheDocument();

    const gauge = screen.getByRole("meter", { name: "Chance of cancer or pre-cancer" });
    expect(gauge).toHaveAttribute("aria-valuenow", "74");
    const top = screen.getByRole("list", { name: "Top three matches" });
    expect(within(top).getAllByRole("meter")).toHaveLength(3);
    expect(within(top).getAllByRole("meter")[0]).toHaveAttribute("aria-valuenow", "71");

    expect(screen.getByText(/rough scores, not reliable chances/)).toBeInTheDocument();
    expect(screen.getByText(/Not a medical diagnosis/)).toBeInTheDocument();
  });

  it("uses plain words, not technical terms", () => {
    render(
      <ResultCard
        result={{ ...explainResponse, thresholds_fitted: true }}
        malignantThreshold={0.19}
      />,
    );
    const card = screen.getByRole("heading", { level: 2 }).closest("section");
    expect(card).not.toHaveTextContent(
      /threshold|probability|calibrat|model class|risk flag|malignant_suspected|latency|ms\b/i,
    );
    expect(screen.queryByText(explainResponse.model_version)).not.toBeInTheDocument();
  });

  it("shows the referral line only when it has been tuned", () => {
    const fitted = { ...explainResponse, thresholds_fitted: true };
    const { rerender } = render(
      <ResultCard result={fitted} malignantThreshold={0.19} malignantClasses={["Melanoma"]} />,
    );
    expect(screen.getByText(/Referral line at 19%/)).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Chance of cancer or pre-cancer" })).toHaveAttribute(
      "aria-valuetext",
      "74%, above the referral line at 19%",
    );
    expect(screen.getByText(/Adds together melanoma/)).toBeInTheDocument();

    rerender(<ResultCard result={explainResponse} malignantThreshold={0.19} />);
    expect(screen.queryByText(/Referral line at 19%/)).not.toBeInTheDocument();
    expect(screen.getByText(/has not been tuned yet/)).toBeInTheDocument();

    rerender(<ResultCard result={fitted} malignantThreshold={null} />);
    expect(screen.getByText(/referral line could not be loaded/)).toBeInTheDocument();
  });

  it("supports all four outcomes, each with its own decision label", () => {
    const outcomes = [
      [explainResponse, "Refer", "Refer: possible skin cancer"],
      [
        { ...explainResponse, status: "low_confidence", risk_flag: "uncertain" },
        "Unsure",
        "Unsure: the system is not confident",
      ],
      [{ ...explainResponse, status: "ok", risk_flag: "benign" }, "No flag", "No flag raised"],
      [rejected, "Not checked", "Not checked: the picture is not clear enough"],
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
    expect(decision).toHaveTextContent(/does not rule out cancer/);
    expect(decision).not.toHaveTextContent(/healthy|cancer-free|benign/i);
  });

  it("says why a rejected image was not checked and hides the scores", () => {
    render(<ResultCard result={rejected} />);

    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Not checked: the picture is not clear enough",
    );
    expect(screen.getByText("The picture is blurry.")).toBeInTheDocument();
    expect(screen.queryAllByRole("meter")).toHaveLength(0);
    expect(screen.queryByText(/rough scores/)).not.toBeInTheDocument();
  });
});
