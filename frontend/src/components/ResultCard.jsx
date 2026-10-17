import { REASONS, formatPercent, riskInfo, statusInfo, verdict } from "../lib/labels.js";
import Badge from "./Badge.jsx";
import { AlertIcon, CheckIcon, ImageOffIcon, InfoIcon } from "./icons.jsx";

const VERDICT_STYLES = {
  positive: { box: "bg-benign-soft", text: "text-benign", Icon: CheckIcon },
  caution: { box: "bg-caution-soft", text: "text-caution", Icon: InfoIcon },
  danger: { box: "bg-alert-soft", text: "text-alert", Icon: AlertIcon },
  neutral: { box: "bg-sunken", text: "text-ink-soft", Icon: ImageOffIcon },
};

function Verdict({ result }) {
  const { tone, title, guidance } = verdict(result);
  const style = VERDICT_STYLES[tone];
  return (
    <div className={`animate-fade-up rounded-2xl p-5 ${style.box}`}>
      <div className={`flex items-start gap-3 ${style.text}`}>
        <style.Icon className="mt-0.5 h-6 w-6 shrink-0" />
        <h3 className="text-lg leading-snug font-bold tracking-tight">{title}</h3>
      </div>
      <p className="mt-2 pl-9 text-[15px] leading-relaxed text-ink">{guidance}</p>
      {result.status === "rejected" && result.reasons?.length > 0 && (
        <ul className="mt-3 space-y-1 pl-9 text-sm text-ink-soft">
          {result.reasons.map((reason) => (
            <li key={reason}>{REASONS[reason] ?? reason}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function TopMatches({ items }) {
  return (
    <ul className="space-y-3" aria-label="Most likely conditions">
      {items.map((item, index) => (
        <li key={item.class_name}>
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className={index === 0 ? "font-semibold" : "text-ink-soft"}>
              {item.class_name}
            </span>
            <span className="font-mono text-[13px] text-ink-soft tabular-nums">
              {formatPercent(item.probability)}
            </span>
          </div>
          <div
            role="meter"
            aria-label={`${item.class_name} score`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(item.probability * 100)}
            className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-sunken"
          >
            <div
              className={`h-full rounded-full transition-[width] duration-1000 ease-out ${
                index === 0 ? "bg-accent" : "bg-accent/35"
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
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-semibold">Cancer-related classes combined</p>
        <p
          className={`font-mono text-sm font-medium tabular-nums ${flagged ? "text-alert" : "text-ink"}`}
        >
          {formatPercent(value)}
        </p>
      </div>
      <div className="relative mt-2.5 h-2 rounded-full bg-sunken">
        <div
          className={`h-full rounded-full transition-[width] duration-1000 ease-out ${
            flagged ? "bg-alert" : "bg-benign"
          }`}
          style={{ width: formatPercent(Math.max(value, 0.01)) }}
        />
        {threshold != null && (
          <span
            className="absolute -top-1.5 h-5 w-0.5 rounded-full bg-ink"
            style={{ left: formatPercent(threshold) }}
            aria-hidden="true"
          />
        )}
      </div>
      <p className="mt-2 text-xs leading-relaxed text-ink-soft">
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
    <details className="group text-sm">
      <summary className="flex cursor-pointer list-none items-center justify-between font-semibold text-ink-soft marker:hidden hover:text-ink">
        Technical details
        <span className="transition group-open:rotate-180">▾</span>
      </summary>
      <div className="mt-3 space-y-3 text-ink-soft">
        <div className="flex flex-wrap gap-2">
          <Badge tone={status.tone}>{status.label}</Badge>
          <Badge tone={risk.tone}>{risk.label}</Badge>
        </div>
        <p className="text-ink">{result.summary}</p>
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

/** The verdict and scores. The photo and heatmap sit beside it in the studio. */
export default function ResultCard({ result, malignantThreshold = null }) {
  const rejected = result.status === "rejected";
  const top = result.top_k[0];

  return (
    <section aria-labelledby="result-heading" aria-live="polite" className="flex flex-col gap-6">
      <div className="flex items-baseline justify-between gap-3">
        <h2
          id="result-heading"
          className="text-xs font-semibold tracking-[0.12em] text-ink-faint uppercase"
        >
          Result
        </h2>
        <span className="font-mono text-xs text-ink-faint">{Math.round(result.latency_ms)} ms</span>
      </div>

      <Verdict result={result} />

      {!rejected && (
        <>
          <div>
            <p className="text-xs font-semibold tracking-[0.12em] text-ink-faint uppercase">
              Closest match
            </p>
            <p className="mt-1.5 text-3xl font-bold tracking-tight">
              {top.class_name}{" "}
              <span className="font-mono text-xl font-medium text-ink-soft">
                {formatPercent(top.probability)}
              </span>
            </p>
          </div>
          <TopMatches items={result.top_k} />
          <div className="h-px bg-line" />
          <MalignancyGauge
            value={result.malignant_probability}
            threshold={result.thresholds_fitted ? malignantThreshold : null}
            flagged={result.risk_flag === "malignant_suspected"}
          />
          {!result.calibrated && (
            <p className="text-xs text-ink-faint">
              Percentages are uncalibrated model scores, not true probabilities.
            </p>
          )}
        </>
      )}

      <div className="h-px bg-line" />
      <Details result={result} />
      <p className="text-xs leading-relaxed text-ink-faint">{result.disclaimer}</p>
    </section>
  );
}
