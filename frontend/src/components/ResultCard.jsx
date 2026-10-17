import { REASONS, formatPercent, riskInfo, statusInfo, verdict } from "../lib/labels.js";
import Badge from "./Badge.jsx";
import HeatmapViewer from "./HeatmapViewer.jsx";
import { AlertIcon, CheckIcon, ImageOffIcon, InfoIcon } from "./icons.jsx";

const VERDICT_STYLES = {
  positive: { box: "bg-benign-soft border-benign/25", text: "text-benign", Icon: CheckIcon },
  caution: { box: "bg-caution-soft border-caution/25", text: "text-caution", Icon: InfoIcon },
  danger: { box: "bg-alert-soft border-alert/30", text: "text-alert", Icon: AlertIcon },
  neutral: { box: "bg-sunken border-line", text: "text-ink-soft", Icon: ImageOffIcon },
};

function Verdict({ result }) {
  const { tone, title, guidance } = verdict(result);
  const style = VERDICT_STYLES[tone];
  return (
    <div className={`rounded-2xl border p-5 ${style.box}`}>
      <div className={`flex items-start gap-3 ${style.text}`}>
        <style.Icon className="mt-0.5 h-6 w-6 shrink-0" />
        <h3 className="text-lg leading-snug font-bold tracking-tight">{title}</h3>
      </div>
      <p className="mt-2 text-[15px] leading-relaxed text-ink">{guidance}</p>
    </div>
  );
}

function TopMatches({ items }) {
  return (
    <ul className="space-y-3.5" aria-label="Most likely conditions">
      {items.map((item, index) => (
        <li key={item.class_name}>
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className={index === 0 ? "font-semibold" : "font-medium text-ink-soft"}>
              {item.class_name}
            </span>
            <span className="font-mono text-[13px] tabular-nums">
              {formatPercent(item.probability)}
            </span>
          </div>
          <div
            role="meter"
            aria-label={`${item.class_name} score`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(item.probability * 100)}
            className="mt-1.5 h-2 overflow-hidden rounded-full bg-sunken"
          >
            <div
              className={`h-full rounded-full transition-[width] duration-700 ease-out ${
                index === 0 ? "bg-accent" : "bg-accent/45"
              }`}
              style={{ width: formatPercent(item.probability) }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** The combined malignant score against the threshold that raises the flag. */
function MalignancyGauge({ value, threshold, flagged }) {
  return (
    <div className="rounded-2xl border border-line p-4">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-semibold">Malignant or pre-malignant classes combined</p>
        <p
          className={`font-mono text-sm font-medium tabular-nums ${flagged ? "text-alert" : "text-ink"}`}
        >
          {formatPercent(value)}
        </p>
      </div>
      <div className="relative mt-3 h-2.5 rounded-full bg-sunken">
        <div
          className={`h-full rounded-full transition-[width] duration-700 ease-out ${
            flagged ? "bg-alert" : "bg-benign"
          }`}
          style={{ width: formatPercent(Math.max(value, 0.01)) }}
        />
        {threshold != null && (
          <span
            className="absolute -top-1.5 h-[22px] w-0.5 rounded-full bg-ink"
            style={{ left: formatPercent(threshold) }}
            aria-hidden="true"
          />
        )}
      </div>
      <p className="mt-2.5 text-xs leading-relaxed text-ink-soft">
        Melanoma, basal and squamous cell carcinoma and actinic keratosis added together.
        {threshold != null && (
          <>
            {" "}
            The flag is raised at {formatPercent(threshold)} (marker), a level chosen to catch 9 in
            10 malignant cases in validation.
          </>
        )}
      </p>
    </div>
  );
}

function Details({ result }) {
  const status = statusInfo(result.status);
  const risk = riskInfo(result.risk_flag);
  return (
    <details className="group rounded-2xl border border-line px-4 py-3 text-sm">
      <summary className="cursor-pointer list-none font-semibold marker:hidden">
        <span className="flex items-center justify-between">
          Technical details
          <span className="text-ink-faint transition group-open:rotate-180">▾</span>
        </span>
      </summary>
      <div className="mt-3 space-y-3 text-ink-soft">
        <div className="flex flex-wrap gap-2">
          <Badge tone={status.tone}>{status.label}</Badge>
          <Badge tone={risk.tone}>{risk.label}</Badge>
        </div>
        <p className="text-ink">{result.summary}</p>
        {result.reasons?.length > 0 && (
          <ul className="list-disc space-y-1 pl-5">
            {result.reasons.map((reason) => (
              <li key={reason}>{REASONS[reason] ?? reason}</li>
            ))}
          </ul>
        )}
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-xs">
          <dt>Model</dt>
          <dd>{result.model_version}</dd>
          <dt>Calibrated</dt>
          <dd>{result.calibrated ? "yes" : "no"}</dd>
          <dt>Thresholds fitted</dt>
          <dd>{result.thresholds_fitted ? "yes" : "no"}</dd>
          <dt>Processing time</dt>
          <dd>{Math.round(result.latency_ms)} ms</dd>
        </dl>
      </div>
    </details>
  );
}

export default function ResultCard({ result, imageSrc, malignantThreshold = null }) {
  const rejected = result.status === "rejected";
  const top = result.top_k[0];

  return (
    <section
      aria-labelledby="result-heading"
      aria-live="polite"
      className="animate-fade-up space-y-5 rounded-3xl border border-line bg-surface p-5 shadow-[var(--shadow-card)] sm:p-7"
    >
      <div className="flex items-baseline justify-between gap-3">
        <h2 id="result-heading" className="text-lg font-semibold tracking-tight">
          Result
        </h2>
        <span className="font-mono text-xs text-ink-faint">{Math.round(result.latency_ms)} ms</span>
      </div>

      <Verdict result={result} />

      {!rejected && (
        <>
          <div>
            <p className="text-xs font-semibold tracking-[0.08em] text-ink-faint uppercase">
              Closest match
            </p>
            <p className="mt-1 text-2xl font-bold tracking-tight">
              {top.class_name}{" "}
              <span className="font-mono text-lg font-medium text-ink-soft">
                {formatPercent(top.probability)}
              </span>
            </p>
          </div>
          <TopMatches items={result.top_k} />
          <MalignancyGauge
            value={result.malignant_probability}
            threshold={result.thresholds_fitted ? malignantThreshold : null}
            flagged={result.risk_flag === "malignant_suspected"}
          />
        </>
      )}

      {!result.calibrated && (
        <p className="text-xs text-ink-faint">
          Percentages are uncalibrated model scores, not true probabilities.
        </p>
      )}

      {!rejected && result.heatmap_png && (
        <HeatmapViewer
          imageSrc={imageSrc}
          heatmapPng={result.heatmap_png}
          explainedClass={result.explained_class}
        />
      )}

      <Details result={result} />

      <p className="border-t border-line pt-4 text-xs leading-relaxed text-ink-faint">
        {result.disclaimer}
      </p>
    </section>
  );
}
