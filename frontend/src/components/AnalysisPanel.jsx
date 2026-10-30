import { useEffect, useState } from "react";

const STEPS = ["Image quality validation", "AI model inference", "Heatmap generation"];

/** Seconds after which a slow request is explained by the hosted service waking up. */
export const SLOW_AFTER_S = 8;

/**
 * Shown beside the image while the request runs. The three steps run on the
 * server in one request, so the frontend cannot see each one finish: the panel
 * lists them without ticking them off, shows an indeterminate bar and the real
 * elapsed time, and never invents a percentage.
 */
export function AnalysisProgress() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <section aria-label="Analysis in progress" className="flex flex-col gap-5">
      <p role="status" className="sr-only">
        Analysing the image…
      </p>
      <div>
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="text-base font-semibold">Analysing image</h2>
          <span className="font-mono text-xs text-ink-soft tabular-nums" aria-hidden="true">
            {seconds} s
          </span>
        </div>
        <div
          role="progressbar"
          aria-label="Analysis in progress"
          className="mt-3 h-1.5 overflow-hidden rounded-full bg-sunken"
        >
          <div className="animate-indeterminate h-full w-2/5 rounded-full bg-accent" />
        </div>
      </div>

      <div>
        <p className="text-xs text-ink-soft">The server runs these steps in one request:</p>
        <ol className="mt-2 space-y-2">
          {STEPS.map((label, index) => (
            <li key={label} className="flex items-center gap-3 text-sm">
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-sunken font-mono text-xs text-ink-soft ring-1 ring-line">
                {index + 1}
              </span>
              {label}
            </li>
          ))}
        </ol>
      </div>

      {seconds >= SLOW_AFTER_S && (
        <p className="rounded-lg bg-sunken px-3 py-2 text-xs leading-relaxed text-ink-soft">
          Taking longer than usual. The hosted service sleeps when idle and can take up to a minute
          to start; later analyses take a few seconds.
        </p>
      )}

      <div className="space-y-3" aria-hidden="true">
        <div className="skeleton h-20 rounded-xl" />
        <div className="skeleton h-3 w-2/3 rounded-full" />
        <div className="skeleton h-2 rounded-full" />
        <div className="skeleton h-2 w-4/5 rounded-full" />
      </div>
    </section>
  );
}
