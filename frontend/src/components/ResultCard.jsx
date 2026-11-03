import { REASONS, formatPercent, verdict } from "../lib/labels.js";
import { AlertIcon, ImageOffIcon, InfoIcon, MinusCircleIcon } from "./icons.jsx";

// "No flag" deliberately uses a neutral "minus" icon, not a check mark: it means
// the referral line was not reached, never that the lesion is benign.
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
        <p className="text-xs font-semibold tracking-[0.1em] text-ink-soft uppercase">Suggestion</p>
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
          <p className="font-semibold">Why it was not checked</p>
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
  if (!fitted)
    return "This version's referral line has not been tuned yet, so treat it as a rough guide.";
  if (threshold == null) return "The referral line could not be loaded.";
  return `Referral line at ${formatPercent(threshold)}: in testing, 9 in 10 cancers and pre-cancers were above it.`;
}

/** The chance of cancer or pre-cancer against the referral line. */
function MalignancyGauge({ value, threshold, fitted, flagged, malignantClasses }) {
  const shown = fitted ? threshold : null;
  const position =
    shown == null ? null : value >= shown ? "above the referral line" : "below the referral line";
  return (
    <div>
      <SectionTitle>Chance of cancer or pre-cancer</SectionTitle>
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
          aria-label="Chance of cancer or pre-cancer"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(value * 100)}
          aria-valuetext={
            shown == null
              ? formatPercent(value)
              : `${formatPercent(value)}, ${position} at ${formatPercent(shown)}`
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
        {thresholdNote(threshold, fitted)} Adds together{" "}
        {malignantClasses?.length
          ? malignantClasses.join(", ").toLowerCase()
          : "the cancer and pre-cancer types"}
        .
      </p>
    </div>
  );
}

function TopMatches({ items }) {
  const [top] = items;
  return (
    <div>
      <SectionTitle>Most similar lesion type</SectionTitle>
      <p className="mt-2 text-xl font-bold tracking-tight">
        {top.class_name}{" "}
        <span className="font-mono text-base font-medium text-ink-soft tabular-nums">
          {formatPercent(top.probability)}
        </span>
      </p>
      <ul className="mt-4 space-y-3" aria-label="Top three matches">
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
              aria-label={`${item.class_name} match`}
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
        The closest of the eight lesion types the system knows. This is a comparison, not a
        diagnosis.
      </p>
    </div>
  );
}

/**
 * The suggestion first, then the evidence behind it in plain words. Every
 * value comes from the API response; the image and heatmap sit beside it in
 * the workspace.
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
        <SectionTitle>Summary</SectionTitle>
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
              These percentages are rough scores, not reliable chances.
            </p>
          )}
        </>
      )}

      <p className="text-xs leading-relaxed text-ink-soft">{result.disclaimer}</p>
    </section>
  );
}
