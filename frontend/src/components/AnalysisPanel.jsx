import { useEffect, useState } from "react";

import { CheckIcon } from "./icons.jsx";

const STEPS = ["Checking image quality", "Running the model", "Drawing the heatmap"];

/** Shown beside the photo while the model runs: the steps and a skeleton of the result. */
export function AnalysisProgress() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 450);
    return () => clearInterval(timer);
  }, []);
  return (
    <section aria-label="Analysis in progress" className="flex flex-col gap-6">
      <p role="status" className="sr-only">
        Analysing the image…
      </p>
      <p className="text-xs font-semibold tracking-[0.12em] text-ink-faint uppercase">Analysing</p>
      <ol className="space-y-3">
        {STEPS.map((label, index) => (
          <li key={label} className="flex items-center gap-3 text-sm">
            {index < step ? (
              <CheckIcon className="h-5 w-5 text-accent" />
            ) : index === step ? (
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-accent/25 border-t-accent" />
            ) : (
              <span className="h-5 w-5 rounded-full border-2 border-line" />
            )}
            <span className={index <= step ? "font-medium" : "text-ink-faint"}>{label}</span>
          </li>
        ))}
      </ol>
      <div className="space-y-3" aria-hidden="true">
        <div className="skeleton h-24 rounded-2xl" />
        <div className="skeleton h-8 w-1/2 rounded-full" />
        <div className="skeleton h-2 rounded-full" />
        <div className="skeleton h-2 w-4/5 rounded-full" />
        <div className="skeleton h-2 w-3/5 rounded-full" />
      </div>
    </section>
  );
}
