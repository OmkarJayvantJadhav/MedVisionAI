import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import HeatmapViewer from "./HeatmapViewer.jsx";

describe("HeatmapViewer", () => {
  it("shows only the image while the model runs", () => {
    render(<HeatmapViewer imageSrc="blob:preview" busy />);
    expect(screen.getByAltText("Selected skin lesion")).toHaveAttribute("src", "blob:preview");
    expect(screen.queryByAltText(/Heatmap/)).not.toBeInTheDocument();
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
  });

  it("lets the user hide the heatmap and change its opacity", async () => {
    const user = userEvent.setup();
    render(
      <HeatmapViewer imageSrc="blob:preview" heatmapPng="aGVhdG1hcA==" explainedClass="Melanoma" />,
    );

    const heatmap = screen.getByAltText("Heatmap of regions that influenced the Melanoma score");
    expect(heatmap).toHaveAttribute("src", "data:image/png;base64,aGVhdG1hcA==");
    expect(heatmap).toHaveStyle({ opacity: "0.45" });
    const slider = screen.getByLabelText("Opacity");
    expect(slider).toHaveAttribute("aria-valuetext", "45%");
    // jsdom does not move range inputs with the keyboard, so set the value directly.
    fireEvent.change(slider, { target: { value: "0.8" } });
    expect(screen.getByAltText(/Heatmap/)).toHaveStyle({ opacity: "0.8" });
    expect(screen.getByText("80%")).toBeInTheDocument();

    await user.click(screen.getByRole("switch", { name: "Show heatmap" }));
    expect(screen.queryByAltText(/Heatmap/)).not.toBeInTheDocument();
    expect(slider).toBeDisabled();
  });

  it("explains that the heatmap is model attention, not a diagnosis", () => {
    render(
      <HeatmapViewer imageSrc="blob:preview" heatmapPng="aGVhdG1hcA==" explainedClass="Melanoma" />,
    );
    expect(screen.getByRole("switch", { name: "Show heatmap" })).toHaveAccessibleDescription(
      /not a diagnosis and not an outline of the lesion or of cancer/,
    );
  });
});
