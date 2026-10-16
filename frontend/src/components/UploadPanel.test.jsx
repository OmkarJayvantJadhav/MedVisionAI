import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { imageFile } from "../test/fixtures.js";
import UploadPanel from "./UploadPanel.jsx";

function renderPanel(props = {}) {
  const handlers = { onSelect: vi.fn(), onAnalyze: vi.fn(), onClear: vi.fn() };
  render(<UploadPanel preview={null} busy={false} {...handlers} {...props} />);
  return handlers;
}

describe("UploadPanel", () => {
  it("accepts a dropped file and highlights while dragging", () => {
    const { onSelect } = renderPanel();
    const zone = screen.getByText("Drop an image here").closest("label");
    const file = imageFile();

    fireEvent.dragOver(zone, { dataTransfer: { files: [] } });
    expect(zone.className).toContain("border-teal-600");
    fireEvent.dragLeave(zone);
    expect(zone.className).not.toContain("bg-teal-50");
    fireEvent.drop(zone, { dataTransfer: { files: [file] } });
    expect(onSelect).toHaveBeenCalledWith(file);
  });

  it("accepts a photo from the camera input", async () => {
    const user = userEvent.setup();
    const { onSelect } = renderPanel();
    const file = imageFile("camera.jpg", "image/jpeg");
    await user.upload(screen.getByLabelText("Take a photo"), file);
    expect(onSelect).toHaveBeenCalledWith(file);
  });

  it("ignores an empty selection", () => {
    const { onSelect } = renderPanel();
    fireEvent.change(screen.getByLabelText("Choose an image file"), { target: { files: [] } });
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("shows progress and disables actions while busy", () => {
    renderPanel({ preview: "blob:preview", busy: true });
    expect(screen.getByRole("button", { name: "Analysing…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Choose another" })).toBeDisabled();
  });
});
