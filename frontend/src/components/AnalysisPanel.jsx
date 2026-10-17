import { useEffect, useState } from "react";

import { CheckIcon, GaugeIcon, ShieldIcon, SparkIcon } from "./icons.jsx";

const STEPS = ["Checking image quality", "Running the model", "Drawing the heatmap"];

/** Placeholder shown while the model runs: the steps and a skeleton of the result. */
export function AnalysisProgress() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 450);
    return () => clearInterval(timer);
  }, []);
  return (
    <section
      aria-label="Analysis in progress"
      className="space-y-6 rounded-3xl border border-line bg-surface p-5 shadow-[var(--shadow-card)] sm:p-7"
    >
      <p role="status" className="sr-only">
        Analysing the image…
      </p>
      <ol className="space-y-2.5">
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
        <div className="skeleton h-4 w-2/3 rounded-full" />
        <div className="skeleton h-4 w-1/2 rounded-full" />
        <div className="skeleton aspect-[4/3] rounded-2xl" />
      </div>
    </section>
  );
}

const PROMISES = [
  {
    Icon: GaugeIcon,
    title: "Calibrated scores",
    text: "Top three matches with probabilities tuned to be honest about uncertainty.",
  },
  {
    Icon: ShieldIcon,
    title: "A malignancy safety flag",
    text: "Raised whenever cancer-related classes add up past a threshold, even if a benign class ranks first.",
  },
  {
    Icon: SparkIcon,
    title: "A visual explanation",
    text: "A heatmap of the regions that drove the result, layered over your photo.",
  },
];

/** What the result panel will contain, shown before any photo is chosen. */
export function ResultPlaceholder() {
  return (
    <section
      aria-label="What you will get"
      className="flex h-full flex-col justify-center rounded-3xl border border-dashed border-line p-6 sm:p-8"
    >
      <p className="text-sm font-semibold text-ink-soft">
        The result and heatmap will appear here.
      </p>
      <ul className="mt-5 space-y-5">
        {PROMISES.map(({ Icon, title, text }) => (
          <li key={title} className="flex gap-3.5">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent">
              <Icon />
            </span>
            <span>
              <span className="block text-[15px] font-semibold">{title}</span>
              <span className="mt-0.5 block text-sm leading-relaxed text-ink-soft">{text}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
