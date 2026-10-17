import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SAMPLES } from "../lib/samples.js";
import { imageFile } from "../test/fixtures.js";
import UploadPanel from "./UploadPanel.jsx";

function renderPanel(props = {}) {
  const handlers = { onSelect: vi.fn(), onSample: vi.fn(), onClear: vi.fn() };
  render(<UploadPanel busy={false} {...handlers} {...props} />);
  return handlers;
}

describe("UploadPanel", () => {
  it("accepts a dropped file and highlights while dragging", () => {
    const { onSelect } = renderPanel();
    const zone = screen.getByText("Drop an image here").closest("[data-dragging]");
    const file = imageFile();

    fireEvent.dragOver(zone, { dataTransfer: { files: [] } });
    expect(zone).toHaveAttribute("data-dragging", "true");
    fireEvent.dragLeave(zone);
    expect(zone).toHaveAttribute("data-dragging", "false");
    fireEvent.drop(zone, { dataTransfer: { files: [file] } });
    expect(onSelect).toHaveBeenCalledWith(file);
  });

  it("opens the file picker from the drop zone button", async () => {
    const user = userEvent.setup();
    renderPanel();
    const input = screen.getByLabelText("Choose an image file");
    const click = vi.spyOn(input, "click");
    await user.click(screen.getByRole("button", { name: /Drop an image here/ }));
    expect(click).toHaveBeenCalled();
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

  it("offers every sample", async () => {
    const user = userEvent.setup();
    const { onSample } = renderPanel();
    for (const sample of SAMPLES) {
      await user.click(screen.getByRole("button", { name: sample.label }));
    }
    expect(onSample.mock.calls.map(([sample]) => sample.id)).toEqual(SAMPLES.map((s) => s.id));
  });

  it("disables the samples while busy", () => {
    renderPanel({ busy: true });
    for (const sample of SAMPLES) {
      expect(screen.getByRole("button", { name: sample.label })).toBeDisabled();
    }
  });
});
