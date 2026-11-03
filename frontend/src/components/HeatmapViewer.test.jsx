import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import HeatmapViewer from "./HeatmapViewer.jsx";

describe("HeatmapViewer", () => {
  it("shows only the image while the model runs", () => {
    render(<HeatmapViewer imageSrc="blob:preview" busy />);
    expect(screen.getByAltText("Selected skin lesion")).toHaveAttribute("src", "blob:preview");
    expect(screen.queryByAltText(/Coloured highlight/)).not.toBeInTheDocument();
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
  });

  it("lets the user hide the highlight and change its strength", async () => {
    const user = userEvent.setup();
    render(
      <HeatmapViewer imageSrc="blob:preview" heatmapPng="aGVhdG1hcA==" explainedClass="Melanoma" />,
    );

    const heatmap = screen.getByAltText(
      "Coloured highlight of the areas that mattered most for the Melanoma match",
    );
    expect(heatmap).toHaveAttribute("src", "data:image/png;base64,aGVhdG1hcA==");
    expect(heatmap).toHaveStyle({ opacity: "0.45" });
    const slider = screen.getByLabelText("Strength");
    expect(slider).toHaveAttribute("aria-valuetext", "45%");
    // jsdom does not move range inputs with the keyboard, so set the value directly.
    fireEvent.change(slider, { target: { value: "0.8" } });
    expect(screen.getByAltText(/Coloured highlight/)).toHaveStyle({ opacity: "0.8" });
    expect(screen.getByText("80%")).toBeInTheDocument();

    await user.click(screen.getByRole("switch", { name: "Show where it looked" }));
    expect(screen.queryByAltText(/Coloured highlight/)).not.toBeInTheDocument();
    expect(slider).toBeDisabled();
  });

  it("explains that the highlight shows where it looked, not a diagnosis", () => {
    render(
      <HeatmapViewer imageSrc="blob:preview" heatmapPng="aGVhdG1hcA==" explainedClass="Melanoma" />,
    );
    expect(
      screen.getByRole("switch", { name: "Show where it looked" }),
    ).toHaveAccessibleDescription(/not a diagnosis and not an outline of the lesion or of cancer/);
  });
});
