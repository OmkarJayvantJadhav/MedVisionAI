import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { metaResponse } from "../test/fixtures.js";
import InfoSections from "./InfoSections.jsx";

describe("InfoSections", () => {
  it("offers the five collapsible sections", () => {
    render(<InfoSections meta={metaResponse} />);
    for (const title of [
      "How MedVision AI works",
      "What the system knows",
      "Privacy",
      "How well it works",
      "Limitations",
    ]) {
      expect(screen.getByText(title).closest("details")).not.toHaveAttribute("open");
    }
  });

  it("shows published results only for the exact served model version", () => {
    const { rerender } = render(<InfoSections meta={metaResponse} />);
    expect(screen.getByText(/Test results for each version/)).toBeInTheDocument();
    expect(screen.queryByText("Everyday photos turned away")).not.toBeInTheDocument();

    rerender(<InfoSections meta={{ ...metaResponse, model_version: "skin-effb0-v2.2.0" }} />);
    expect(screen.getByText("Everyday photos turned away").nextSibling).toHaveTextContent("99.6%");
    expect(screen.getByText(/pre-cancers that raised the alert/).nextSibling).toHaveTextContent(
      "90.4%",
    );
    expect(screen.getByText("Melanomas named as melanoma").nextSibling).toHaveTextContent("43%");
    expect(screen.getByText(/percentages were off/).nextSibling).toHaveTextContent(
      "About 3 percentage points",
    );
  });

  it("states that a no-flag answer never rules out rarer cancers", () => {
    render(<InfoSections meta={null} />);
    expect(screen.getByText(/never rules them out/)).toBeInTheDocument();
    expect(screen.getByText(/These details appear once the system is ready/)).toBeInTheDocument();
  });

  it("avoids technical terms", () => {
    const { container } = render(
      <InfoSections meta={{ ...metaResponse, model_version: "skin-effb0-v2.2.0" }} />,
    );
    expect(container).not.toHaveTextContent(
      /ONNX|CPU|inference|temperature|calibrat|threshold|ECE|F1|held-out|validation|architecture|EfficientNet/i,
    );
  });
});
