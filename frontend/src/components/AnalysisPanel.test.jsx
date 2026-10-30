import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AnalysisProgress, SLOW_AFTER_S } from "./AnalysisPanel.jsx";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("AnalysisProgress", () => {
  it("lists the steps without claiming any of them has finished", () => {
    render(<AnalysisProgress />);
    expect(screen.getByRole("progressbar", { name: "Analysis in progress" })).not.toHaveAttribute(
      "aria-valuenow",
    );
    for (const step of ["Image quality validation", "AI model inference", "Heatmap generation"]) {
      expect(screen.getByText(step)).toBeInTheDocument();
    }
    act(() => vi.advanceTimersByTime(3000));
    expect(screen.getByText("3 s")).toBeInTheDocument();
    expect(screen.queryByText(/complete|done/i)).not.toBeInTheDocument();
  });

  it("explains a slow first request", () => {
    render(<AnalysisProgress />);
    expect(screen.queryByText(/Taking longer than usual/)).not.toBeInTheDocument();
    act(() => vi.advanceTimersByTime(SLOW_AFTER_S * 1000));
    expect(screen.getByText(/Taking longer than usual/)).toBeInTheDocument();
  });
});
