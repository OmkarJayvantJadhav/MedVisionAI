import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { explainResponse } from "../test/fixtures.js";
import ResultCard from "./ResultCard.jsx";

describe("ResultCard", () => {
  it("leads with the verdict, then scores, details and disclaimer", () => {
    render(<ResultCard result={explainResponse} />);

    expect(
      screen.getByRole("heading", { name: "Possible malignancy: please see a dermatologist" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Assessment complete")).toBeInTheDocument();
    expect(screen.getByText(explainResponse.summary)).toBeInTheDocument();
    const meters = screen.getAllByRole("meter");
    expect(meters).toHaveLength(3);
    expect(meters[0]).toHaveAttribute("aria-valuenow", "71");
    expect(screen.getByText("74%")).toBeInTheDocument();
    expect(screen.getByText(/uncalibrated model scores/)).toBeInTheDocument();
    expect(screen.getByText(/Not a medical diagnosis/)).toBeInTheDocument();
  });

  it("shows the flag threshold only when thresholds are fitted", () => {
    const fitted = { ...explainResponse, thresholds_fitted: true };
    const { rerender } = render(<ResultCard result={fitted} malignantThreshold={0.19} />);
    expect(screen.getByText(/The flag is raised at 19%/)).toBeInTheDocument();

    rerender(<ResultCard result={explainResponse} malignantThreshold={0.19} />);
    expect(screen.queryByText(/The flag is raised at/)).not.toBeInTheDocument();
  });

  it("says what went wrong for rejected images and hides the scores", () => {
    const rejected = {
      ...explainResponse,
      status: "rejected",
      risk_flag: "uncertain",
      reasons: ["too_blurry"],
      calibrated: true,
    };
    render(<ResultCard result={rejected} />);

    expect(
      screen.getByRole("heading", { name: "This photo could not be assessed" }),
    ).toBeInTheDocument();
    expect(screen.getByText("The photo is too blurry.")).toBeInTheDocument();
    expect(screen.queryAllByRole("meter")).toHaveLength(0);
    expect(screen.queryByText(/uncalibrated/)).not.toBeInTheDocument();
  });
});
