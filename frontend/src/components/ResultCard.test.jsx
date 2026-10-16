import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { explainResponse } from "../test/fixtures.js";
import ResultCard from "./ResultCard.jsx";

describe("ResultCard", () => {
  it("shows badges, summary, scores, heatmap and disclaimer", () => {
    render(<ResultCard result={explainResponse} imageSrc="blob:preview" />);

    expect(screen.getByText("Assessment complete")).toBeInTheDocument();
    expect(screen.getByText(/Possible malignancy/)).toBeInTheDocument();
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

  it("hides scores and heatmap for rejected images", () => {
    const rejected = {
      ...explainResponse,
      status: "rejected",
      risk_flag: "uncertain",
      calibrated: true,
    };
    render(<ResultCard result={rejected} imageSrc="blob:preview" />);

    expect(screen.getByText("Image not usable")).toBeInTheDocument();
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

    await user.click(screen.getByLabelText("Show heatmap"));
    expect(screen.queryByAltText(/Heatmap/)).not.toBeInTheDocument();
    expect(slider).toBeDisabled();
  });
});
