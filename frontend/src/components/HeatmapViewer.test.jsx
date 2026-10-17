import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import HeatmapViewer from "./HeatmapViewer.jsx";

describe("HeatmapViewer", () => {
  it("shows only the photo while the model runs", () => {
    render(<HeatmapViewer imageSrc="blob:preview" busy onClear={vi.fn()} />);
    expect(screen.getByAltText("Selected skin lesion")).toHaveAttribute("src", "blob:preview");
    expect(screen.queryByAltText(/Heatmap/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose another" })).toBeDisabled();
  });

  it("lets the user hide the heatmap and change its opacity", async () => {
    const user = userEvent.setup();
    const onClear = vi.fn();
    render(
      <HeatmapViewer
        imageSrc="blob:preview"
        heatmapPng="aGVhdG1hcA=="
        explainedClass="Melanoma"
        onClear={onClear}
      />,
    );

    const heatmap = screen.getByAltText("Heatmap of regions that influenced the Melanoma score");
    expect(heatmap).toHaveAttribute("src", "data:image/png;base64,aGVhdG1hcA==");
    expect(heatmap).toHaveStyle({ opacity: "0.45" });
    const slider = screen.getByLabelText("Opacity");
    // jsdom does not move range inputs with the keyboard, so set the value directly.
    fireEvent.change(slider, { target: { value: "0.8" } });
    expect(screen.getByAltText(/Heatmap/)).toHaveStyle({ opacity: "0.8" });

    await user.click(screen.getByRole("switch", { name: "Show heatmap" }));
    expect(screen.queryByAltText(/Heatmap/)).not.toBeInTheDocument();
    expect(slider).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Choose another" }));
    expect(onClear).toHaveBeenCalledOnce();
  });
});
