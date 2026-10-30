import { REASONS, formatPercent, riskInfo, statusInfo, verdict } from "../lib/labels.js";
import Badge from "./Badge.jsx";
import { AlertIcon, ChevronDownIcon, ImageOffIcon, InfoIcon, MinusCircleIcon } from "./icons.jsx";

// "No flag" deliberately uses a neutral "minus" icon, not a check mark: it means
// the referral threshold was not reached, never that the lesion is benign.
const DECISION_STYLES = {
  danger: {
    panel: "border-alert/40 bg-alert-soft",
    accent: "text-alert",
    Icon: AlertIcon,
  },
  caution: {
    panel: "border-caution/40 bg-caution-soft",
    accent: "text-caution",
    Icon: InfoIcon,
  },
  positive: {
    panel: "border-benign/35 bg-benign-soft",
    accent: "text-benign",
    Icon: MinusCircleIcon,
  },
  neutral: {
    panel: "border-line-strong bg-sunken",
    accent: "text-ink-soft",
    Icon: ImageOffIcon,
  },
};

function Decision({ result }) {
  const { tone, decision, title, guidance } = verdict(result);
  const style = DECISION_STYLES[tone];
  return (
    <div className={`animate-fade-in rounded-xl border-2 p-5 ${style.panel}`}>
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-semibold tracking-[0.1em] text-ink-soft uppercase">
          Referral decision
        </p>
        <span
          className={`rounded-md bg-surface px-2 py-0.5 text-xs font-bold tracking-wide uppercase ring-1 ring-current ${style.accent}`}
        >
          {decision}
        </span>
      </div>
      <div className={`mt-3 flex items-start gap-3 ${style.accent}`}>
        <style.Icon className="mt-0.5 h-7 w-7 shrink-0" />
        <h2
          id="result-heading"
          className="text-xl leading-snug font-bold tracking-tight sm:text-2xl"
        >
          {title}
        </h2>
      </div>
      <p className="mt-3 text-[15px] leading-relaxed text-ink">{guidance}</p>
      {result.status === "rejected" && result.reasons?.length > 0 && (
        <div className="mt-3 text-sm">
          <p className="font-semibold">Why it was not assessed</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-ink-soft">
            {result.reasons.map((reason) => (
              <li key={reason}>{REASONS[reason] ?? reason}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h3 className="text-xs font-semibold tracking-[0.1em] text-ink-soft uppercase">{children}</h3>
  );
}

function thresholdNote(threshold, fitted) {
  if (!fitted) return "Provisional: this model's thresholds have not been fitted on data.";
  if (threshold == null) return "Referral threshold unavailable: model information not loaded.";
  return `Referral threshold ${formatPercent(threshold)}, set to catch 9 in 10 malignant lesions in validation.`;
}

/** The combined malignant probability against the referral threshold. */
function MalignancyGauge({ value, threshold, fitted, flagged, malignantClasses }) {
  const shown = fitted ? threshold : null;
  const position =
    shown == null
      ? null
      : value >= shown
        ? "at or above the referral threshold"
        : "below the referral threshold";
  return (
    <div>
      <SectionTitle>Combined malignancy probability</SectionTitle>
      <div className="mt-2 flex items-baseline gap-3">
        <p
          className={`font-mono text-3xl font-medium tabular-nums ${flagged ? "text-alert" : "text-ink"}`}
        >
          {formatPercent(value)}
        </p>
        {position && <p className="text-sm text-ink-soft">{position}</p>}
      </div>
      <div className="relative mt-3">
        <div
          role="meter"
          aria-label="Combined malignancy probability"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(value * 100)}
          aria-valuetext={
            shown == null
              ? formatPercent(value)
              : `${formatPercent(value)}, ${position} of ${formatPercent(shown)}`
          }
          className="h-2.5 overflow-hidden rounded-full bg-sunken ring-1 ring-line"
        >
          <div
            className={`h-full rounded-full transition-[width] duration-700 ease-out ${
              flagged ? "bg-alert" : "bg-accent"
            }`}
            style={{ width: formatPercent(Math.max(value, 0.01)) }}
          />
        </div>
        {shown != null && (
          <span
            aria-hidden="true"
            className="absolute -top-1 h-[18px] w-0.5 -translate-x-1/2 rounded-full bg-ink"
            style={{ left: formatPercent(shown) }}
          />
        )}
      </div>
      <div
        aria-hidden="true"
        className="mt-1 flex justify-between font-mono text-[11px] text-ink-faint"
      >
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-ink-soft">
        {shown != null && (
          <span className="mr-1 inline-block h-2.5 w-0.5 translate-y-px rounded-full bg-ink" />
        )}
        {thresholdNote(threshold, fitted)} Sum of{" "}
        {malignantClasses?.length
          ? malignantClasses.join(", ")
          : "the malignant and pre-malignant classes"}
        .
      </p>
    </div>
  );
}

function TopMatches({ items }) {
  const [top] = items;
  return (
    <div>
      <SectionTitle>Closest model class</SectionTitle>
      <p className="mt-2 text-xl font-bold tracking-tight">
        {top.class_name}{" "}
        <span className="font-mono text-base font-medium text-ink-soft tabular-nums">
          {formatPercent(top.probability)}
        </span>
      </p>
      <ul className="mt-4 space-y-3" aria-label="Top three model classes">
        {items.map((item, index) => (
          <li key={item.class_name}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span
                className={`min-w-0 truncate ${index === 0 ? "font-semibold" : "text-ink-soft"}`}
              >
                {item.class_name}
              </span>
              <span className="font-mono text-[13px] text-ink-soft tabular-nums">
                {formatPercent(item.probability)}
              </span>
            </div>
            <div
              role="meter"
              aria-label={`${item.class_name} probability`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(item.probability * 100)}
              className="mt-1.5 h-2 overflow-hidden rounded-full bg-sunken"
            >
              <div
                className={`h-full rounded-full transition-[width] duration-700 ease-out ${
                  index === 0 ? "bg-accent" : "bg-accent/40"
                }`}
                style={{ width: formatPercent(item.probability) }}
              />
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs leading-relaxed text-ink-soft">
        The class the image most resembles among eight lesion types, not a diagnosis.
      </p>
    </div>
  );
}

function TechnicalDetails({ result }) {
  const status = statusInfo(result.status);
  const risk = riskInfo(result.risk_flag);
  return (
    <details className="group rounded-xl border border-line">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-xl px-4 py-3 text-sm font-semibold marker:hidden hover:bg-sunken/60 [&::-webkit-details-marker]:hidden">
        Technical details
        <ChevronDownIcon className="h-4 w-4 text-ink-soft transition group-open:rotate-180" />
      </summary>
      <div className="space-y-3 border-t border-line px-4 py-3 text-sm">
        <div className="flex flex-wrap gap-2">
          <Badge tone={status.tone}>{status.label}</Badge>
          <Badge tone={risk.tone}>{risk.label}</Badge>
        </div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
          <dt className="text-ink-soft">Status</dt>
          <dd className="font-mono">{result.status}</dd>
          <dt className="text-ink-soft">Risk flag</dt>
          <dd className="font-mono">{result.risk_flag}</dd>
          <dt className="text-ink-soft">Reasons</dt>
          <dd className="font-mono break-words">
            {result.reasons?.length ? result.reasons.join(", ") : "none"}
          </dd>
          {result.explained_class && (
            <>
              <dt className="text-ink-soft">Heatmap class</dt>
              <dd className="font-mono">{result.explained_class}</dd>
            </>
          )}
          <dt className="text-ink-soft">Calibrated</dt>
          <dd className="font-mono">{result.calibrated ? "yes" : "no"}</dd>
          <dt className="text-ink-soft">Thresholds fitted</dt>
          <dd className="font-mono">{result.thresholds_fitted ? "yes" : "no"}</dd>
        </dl>
      </div>
    </details>
  );
}

/**
 * The referral decision first, then the evidence behind it. Every value comes
 * from the API response; the image and heatmap sit beside it in the workspace.
 */
export default function ResultCard({ result, malignantThreshold = null, malignantClasses = null }) {
  const rejected = result.status === "rejected";

  return (
    <section
      aria-labelledby="result-heading"
      aria-live="polite"
      className="flex min-w-0 flex-col gap-6"
    >
      <Decision result={result} />

      <div>
        <SectionTitle>Clinical summary</SectionTitle>
        <p className="mt-2 text-[15px] leading-relaxed">{result.summary}</p>
      </div>

      {!rejected && (
        <>
          <MalignancyGauge
            value={result.malignant_probability}
            threshold={malignantThreshold}
            fitted={result.thresholds_fitted}
            flagged={result.risk_flag === "malignant_suspected"}
            malignantClasses={malignantClasses}
          />
          <TopMatches items={result.top_k} />
          {!result.calibrated && (
            <p className="rounded-lg bg-caution-soft px-3 py-2 text-xs text-ink">
              Percentages are uncalibrated model scores, not true probabilities.
            </p>
          )}
        </>
      )}

      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl bg-sunken px-4 py-3">
          <dt className="text-xs text-ink-soft">Model version</dt>
          <dd className="mt-0.5 font-mono text-[13px] break-all">{result.model_version}</dd>
        </div>
        <div className="rounded-xl bg-sunken px-4 py-3">
          <dt className="text-xs text-ink-soft">Processing time</dt>
          <dd className="mt-0.5 font-mono text-[13px] tabular-nums">
            {Math.round(result.latency_ms)} ms
          </dd>
        </div>
      </dl>

      <TechnicalDetails result={result} />
      <p className="text-xs leading-relaxed text-ink-soft">{result.disclaimer}</p>
    </section>
  );
}
