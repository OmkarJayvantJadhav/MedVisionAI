import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { metaResponse } from "../test/fixtures.js";
import InfoSections from "./InfoSections.jsx";

describe("InfoSections", () => {
  it("offers the five collapsible sections", () => {
    render(<InfoSections meta={metaResponse} />);
    for (const title of [
      "How MedVision AI works",
      "About the AI model",
      "Privacy and data handling",
      "Model performance",
      "Known limitations",
    ]) {
      expect(screen.getByText(title).closest("details")).not.toHaveAttribute("open");
    }
  });

  it("shows published results only for the exact served model version", () => {
    const { rerender } = render(<InfoSections meta={metaResponse} />);
    expect(screen.getByText(/Measured results for each model version/)).toBeInTheDocument();
    expect(screen.queryByText("Everyday photos rejected")).not.toBeInTheDocument();

    rerender(<InfoSections meta={{ ...metaResponse, model_version: "skin-effb0-v2.2.0" }} />);
    expect(screen.getByText("Everyday photos rejected").nextSibling).toHaveTextContent("99.6%");
    expect(screen.getByText(/Malignant lesions flagged/).nextSibling).toHaveTextContent("90.4%");
  });

  it("states that a no-flag answer never rules out rarer cancers", () => {
    render(<InfoSections meta={null} />);
    expect(screen.getByText(/never rules them out/)).toBeInTheDocument();
    expect(screen.getByText(/Model details appear here/)).toBeInTheDocument();
  });
});
