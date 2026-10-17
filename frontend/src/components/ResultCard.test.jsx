import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { explainResponse } from "../test/fixtures.js";
import ResultCard from "./ResultCard.jsx";

describe("ResultCard", () => {
  it("leads with the verdict, then scores, heatmap, details and disclaimer", () => {
    render(<ResultCard result={explainResponse} imageSrc="blob:preview" />);

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
    expect(screen.getByAltText(/Heatmap of regions/)).toHaveAttribute(
      "src",
      expect.stringContaining("data:image/png;base64,"),
    );
    expect(screen.getByText(/Not a medical diagnosis/)).toBeInTheDocument();
  });

  it("shows the flag threshold only when thresholds are fitted", () => {
    const fitted = { ...explainResponse, thresholds_fitted: true };
    const { rerender } = render(
      <ResultCard result={fitted} imageSrc="blob:preview" malignantThreshold={0.19} />,
    );
    expect(screen.getByText(/The flag is raised at 19%/)).toBeInTheDocument();

    rerender(
      <ResultCard result={explainResponse} imageSrc="blob:preview" malignantThreshold={0.19} />,
    );
    expect(screen.queryByText(/The flag is raised at/)).not.toBeInTheDocument();
  });

  it("says what went wrong for rejected images and hides scores and heatmap", () => {
    const rejected = {
      ...explainResponse,
      status: "rejected",
      risk_flag: "uncertain",
      reasons: ["too_blurry"],
      calibrated: true,
    };
    render(<ResultCard result={rejected} imageSrc="blob:preview" />);

    expect(
      screen.getByRole("heading", { name: "This photo could not be assessed" }),
    ).toBeInTheDocument();
    expect(screen.getByText("The photo is too blurry.")).toBeInTheDocument();
    expect(screen.queryAllByRole("meter")).toHaveLength(0);
    expect(screen.queryByAltText(/Heatmap/)).not.toBeInTheDocument();
    expect(screen.queryByText(/uncalibrated/)).not.toBeInTheDocument();
  });

  it("lets the user hide the heatmap and change its opacity", async () => {
    const user = userEvent.setup();
    render(<ResultCard result={explainResponse} imageSrc="blob:preview" />);

    const slider = screen.getByLabelText("Opacity");
    expect(screen.getByAltText(/Heatmap/)).toHaveStyle({ opacity: "0.45" });
    // jsdom does not move range inputs with the keyboard, so set the value directly.
    fireEvent.change(slider, { target: { value: "0.8" } });
    expect(screen.getByAltText(/Heatmap/)).toHaveStyle({ opacity: "0.8" });

    await user.click(screen.getByRole("switch", { name: "Show heatmap" }));
    expect(screen.queryByAltText(/Heatmap/)).not.toBeInTheDocument();
    expect(slider).toBeDisabled();
  });
});
