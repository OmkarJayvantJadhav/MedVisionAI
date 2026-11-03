import {
  ALERT,
  NO_ALERT,
  REASONS,
  confidence,
  formatPercent,
  hasAlert,
  rejection,
} from "../lib/labels.js";
import { AlertIcon, ImageOffIcon, MinusCircleIcon } from "./icons.jsx";

const HEADING = "text-xl leading-snug font-bold tracking-tight sm:text-2xl";

const BADGE_TONES = {
  positive: "text-benign",
  caution: "text-caution",
};

function SectionTitle({ children }) {
  return (
    <h3 className="text-xs font-semibold tracking-[0.1em] text-ink-soft uppercase">{children}</h3>
  );
}

/** A rejected image: what was wrong and how to fix it. */
function NotChecked({ result }) {
  const { title, guidance } = rejection(result);
  return (
    <div className="animate-fade-in rounded-xl border-2 border-line-strong bg-sunken p-5">
      <div className="flex items-start gap-3 text-ink-soft">
        <ImageOffIcon className="mt-0.5 h-7 w-7 shrink-0" />
        <h2 id="result-heading" className={HEADING}>
          {title}
        </h2>
      </div>
      <p className="mt-3 text-[15px] leading-relaxed text-ink">{guidance}</p>
      {result.reasons?.length > 0 && (
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

/** Shown above everything else whenever the alert is raised. */
function HighRiskAlert() {
  return (
    <div className="animate-fade-in rounded-xl border-2 border-alert/40 bg-alert-soft p-5">
      <div className="flex items-start gap-3 text-alert">
        <AlertIcon className="mt-0.5 h-7 w-7 shrink-0" />
        <h2 id="result-heading" className={HEADING}>
          {ALERT.title}
        </h2>
      </div>
      <p className="mt-3 text-[15px] leading-relaxed text-ink">{ALERT.guidance}</p>
    </div>
  );
}

/** The most likely lesion type, how sure the model is, and the top three matches. */
function LesionType({ result, isHeading }) {
  const [top] = result.top_k;
  const sure = confidence(result);
  const Title = isHeading ? "h2" : "p";
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <SectionTitle>Most likely lesion type</SectionTitle>
        <span
          className={`rounded-md bg-surface px-2 py-0.5 text-xs font-bold tracking-wide uppercase ring-1 ring-current ${BADGE_TONES[sure.tone]}`}
        >
          {sure.label}
        </span>
      </div>
      <Title id={isHeading ? "result-heading" : undefined} className={`mt-2 ${HEADING} text-ink`}>
        {top.class_name}{" "}
        <span className="font-mono text-lg font-medium text-ink-soft tabular-nums">
          {formatPercent(top.probability)}
        </span>
      </Title>
      <p className="mt-2 text-sm leading-relaxed text-ink-soft">{sure.note}</p>
      <ul className="mt-4 space-y-3" aria-label="Top three matches">
        {result.top_k.map((item, index) => (
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
    </div>
  );
}

function alertLineNote(threshold, fitted) {
  if (!fitted)
    return "This version's alert line has not been tuned yet, so treat it as a rough guide.";
  if (threshold == null) return "The alert line could not be loaded.";
  return `Alert line at ${formatPercent(threshold)}: in testing, 9 in 10 cancers and pre-cancers were above it.`;
}

/** The chance of cancer or pre-cancer against the alert line. */
function CancerChance({ value, threshold, fitted, flagged, malignantClasses }) {
  const shown = fitted ? threshold : null;
  const position =
    shown == null ? null : value >= shown ? "above the alert line" : "below the alert line";
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
        {alertLineNote(threshold, fitted)} Adds together{" "}
        {malignantClasses?.length
          ? malignantClasses.join(", ").toLowerCase()
          : "the cancer and pre-cancer types"}
        .
      </p>
      {!flagged && (
        // A neutral "minus" icon, not a check mark: no alert never means benign.
        <p className="mt-3 flex items-start gap-2 rounded-lg bg-sunken px-3 py-2 text-sm leading-relaxed">
          <MinusCircleIcon className="mt-0.5 h-4 w-4 shrink-0 text-ink-soft" />
          <span>{NO_ALERT}</span>
        </p>
      )}
    </div>
  );
}

/**
 * The result. A rejected image says what to fix. Otherwise the high-risk
 * alert, when raised, comes first, so a cancer warning is never hidden below
 * the lesion type; then the lesion type with how sure the model is, the
 * chance of cancer or pre-cancer and the summary. Every value comes from the
 * API response; the image and heatmap sit beside it in the workspace.
 */
export default function ResultCard({ result, malignantThreshold = null, malignantClasses = null }) {
  const disclaimer = <p className="text-xs leading-relaxed text-ink-soft">{result.disclaimer}</p>;
  const summary = (
    <div>
      <SectionTitle>Summary</SectionTitle>
      <p className="mt-2 text-[15px] leading-relaxed">{result.summary}</p>
    </div>
  );

  if (rejection(result)) {
    return (
      <section
        aria-labelledby="result-heading"
        aria-live="polite"
        className="flex min-w-0 flex-col gap-6"
      >
        <NotChecked result={result} />
        {summary}
        {disclaimer}
      </section>
    );
  }

  const alert = hasAlert(result);
  return (
    <section
      aria-labelledby="result-heading"
      aria-live="polite"
      className="flex min-w-0 flex-col gap-6"
    >
      {alert && <HighRiskAlert />}
      <LesionType result={result} isHeading={!alert} />
      <CancerChance
        value={result.malignant_probability}
        threshold={malignantThreshold}
        fitted={result.thresholds_fitted}
        flagged={alert}
        malignantClasses={malignantClasses}
      />
      {summary}
      {!result.calibrated && (
        <p className="rounded-lg bg-caution-soft px-3 py-2 text-xs text-ink">
          These percentages are rough scores, not reliable chances.
        </p>
      )}
      {disclaimer}
    </section>
  );
}
