import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { explainResponse } from "../test/fixtures.js";
import ResultCard from "./ResultCard.jsx";

const rejected = {
  ...explainResponse,
  status: "rejected",
  risk_flag: "uncertain",
  reasons: ["too_blurry"],
  calibrated: true,
};
const confidentNoAlert = { ...explainResponse, status: "ok", risk_flag: "benign" };
const uncertainNoAlert = { ...explainResponse, status: "low_confidence", risk_flag: "uncertain" };

describe("ResultCard", () => {
  it("puts the high-risk alert above the lesion type, then the evidence and disclaimer", () => {
    render(<ResultCard result={explainResponse} />);

    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading).toHaveTextContent("High-risk alert: possible skin cancer");
    const lesionType = screen.getByText("Most likely lesion type");
    expect(heading.compareDocumentPosition(lesionType)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(screen.getByText(explainResponse.summary)).toBeInTheDocument();

    const gauge = screen.getByRole("meter", { name: "Chance of cancer or pre-cancer" });
    expect(gauge).toHaveAttribute("aria-valuenow", "74");
    const top = screen.getByRole("list", { name: "Top three matches" });
    expect(within(top).getAllByRole("meter")).toHaveLength(3);
    expect(within(top).getAllByRole("meter")[0]).toHaveAttribute("aria-valuenow", "71");

    expect(screen.queryByText(/No high-risk alert/)).not.toBeInTheDocument();
    expect(screen.getByText(/rough scores, not reliable chances/)).toBeInTheDocument();
    expect(screen.getByText(/Not a medical diagnosis/)).toBeInTheDocument();
  });

  it("leads with the lesion type and how sure the model is when there is no alert", () => {
    const outcomes = [
      [confidentNoAlert, "Confident"],
      [uncertainNoAlert, "Uncertain"],
    ];
    for (const [result, label] of outcomes) {
      const { unmount } = render(<ResultCard result={result} />);
      expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("Melanoma 71%");
      expect(screen.getByText(label, { selector: "span" })).toBeInTheDocument();
      expect(screen.queryByText(/High-risk alert/)).not.toBeInTheDocument();
      unmount();
    }
  });

  it("keeps the alert first even when the model is uncertain about the type", () => {
    render(<ResultCard result={{ ...explainResponse, status: "low_confidence" }} />);
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("High-risk alert");
    expect(screen.getByText("Uncertain", { selector: "span" })).toBeInTheDocument();
  });

  it("never calls a result without an alert benign, healthy or cancer-free", () => {
    render(<ResultCard result={confidentNoAlert} />);
    const note = screen.getByText(/No high-risk alert/);
    expect(note).toHaveTextContent(/does not rule out cancer/);
    expect(note).not.toHaveTextContent(/healthy|cancer-free|benign/i);
  });

  it("shows the alert line only when it has been tuned", () => {
    const fitted = { ...explainResponse, thresholds_fitted: true };
    const { rerender } = render(
      <ResultCard result={fitted} malignantThreshold={0.19} malignantClasses={["Melanoma"]} />,
    );
    expect(screen.getByText(/Alert line at 19%/)).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Chance of cancer or pre-cancer" })).toHaveAttribute(
      "aria-valuetext",
      "74%, above the alert line at 19%",
    );
    expect(screen.getByText(/Adds together melanoma/)).toBeInTheDocument();

    rerender(<ResultCard result={explainResponse} malignantThreshold={0.19} />);
    expect(screen.queryByText(/Alert line at 19%/)).not.toBeInTheDocument();
    expect(screen.getByText(/has not been tuned yet/)).toBeInTheDocument();

    rerender(<ResultCard result={fitted} malignantThreshold={null} />);
    expect(screen.getByText(/alert line could not be loaded/)).toBeInTheDocument();
  });

  it("uses plain words, not technical terms", () => {
    render(
      <ResultCard
        result={{ ...explainResponse, thresholds_fitted: true }}
        malignantThreshold={0.19}
      />,
    );
    const card = screen.getByRole("heading", { level: 2 }).closest("section");
    expect(card).not.toHaveTextContent(
      /threshold|probability|calibrat|model class|risk flag|malignant_suspected|latency|ms\b/i,
    );
    expect(screen.queryByText(explainResponse.model_version)).not.toBeInTheDocument();
  });

  it("says why a rejected image was not checked and hides the type, alert and scores", () => {
    render(<ResultCard result={{ ...rejected, risk_flag: "malignant_suspected" }} />);

    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Not checked: the picture is not clear enough",
    );
    expect(screen.getByText("The picture is blurry.")).toBeInTheDocument();
    expect(screen.queryAllByRole("meter")).toHaveLength(0);
    expect(screen.queryByText(/High-risk alert/)).not.toBeInTheDocument();
    expect(screen.queryByText("Most likely lesion type")).not.toBeInTheDocument();
  });
});
