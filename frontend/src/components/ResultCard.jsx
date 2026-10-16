import { formatPercent, riskInfo, statusInfo } from "../lib/labels.js";
import Badge from "./Badge.jsx";
import HeatmapViewer from "./HeatmapViewer.jsx";

function Probabilities({ items }) {
  return (
    <ul className="space-y-3" aria-label="Most likely conditions">
      {items.map((item) => (
        <li key={item.class_name}>
          <div className="flex justify-between text-sm">
            <span className="font-medium">{item.class_name}</span>
            <span className="tabular-nums">{formatPercent(item.probability)}</span>
          </div>
          <div
            role="meter"
            aria-label={`${item.class_name} score`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(item.probability * 100)}
            className="mt-1 h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
          >
            <div
              className="h-full rounded-full bg-teal-600"
              style={{ width: formatPercent(item.probability) }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function ResultCard({ result, imageSrc }) {
  const status = statusInfo(result.status);
  const risk = riskInfo(result.risk_flag);
  const rejected = result.status === "rejected";

  return (
    <section
      aria-labelledby="result-heading"
      aria-live="polite"
      className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200 dark:bg-slate-900 dark:ring-slate-800"
    >
      <h2 id="result-heading" className="text-lg font-semibold">
        2. Result
      </h2>
      <div className="mt-3 flex flex-wrap gap-2">
        <Badge tone={status.tone}>{status.label}</Badge>
        <Badge tone={risk.tone}>{risk.label}</Badge>
      </div>

      <p className="mt-4 leading-relaxed">{result.summary}</p>

      {!rejected && (
        <>
          <h3 className="mt-6 mb-3 text-sm font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
            Top matches
          </h3>
          <Probabilities items={result.top_k} />
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
            Malignant or pre-malignant classes combined:{" "}
            <strong className="tabular-nums">{formatPercent(result.malignant_probability)}</strong>
          </p>
        </>
      )}

      {!result.calibrated && (
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
          Percentages are uncalibrated model scores, not true probabilities.
        </p>
      )}

      {!rejected && result.heatmap_png && (
        <div className="mt-6">
          <HeatmapViewer
            imageSrc={imageSrc}
            heatmapPng={result.heatmap_png}
            explainedClass={result.explained_class}
          />
        </div>
      )}

      <p className="mt-6 border-t border-slate-200 pt-4 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        {result.disclaimer} Model {result.model_version} · {Math.round(result.latency_ms)} ms
      </p>
    </section>
  );
}
