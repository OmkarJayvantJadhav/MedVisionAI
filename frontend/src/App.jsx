import { useCallback, useEffect, useRef, useState } from "react";

import { explain, getMeta } from "./api/client.js";
import { AnalysisProgress } from "./components/AnalysisPanel.jsx";
import Disclaimer from "./components/Disclaimer.jsx";
import ErrorAlert from "./components/ErrorAlert.jsx";
import HeatmapViewer from "./components/HeatmapViewer.jsx";
import Footer from "./components/Footer.jsx";
import { HowItWorks, ModelFacts, Privacy } from "./components/InfoSections.jsx";
import { LockIcon, ShieldIcon, SparkIcon } from "./components/icons.jsx";
import ResultCard from "./components/ResultCard.jsx";
import TopBar from "./components/TopBar.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import { loadSample } from "./lib/samples.js";
import { DEFAULT_MAX_MB, validateFile } from "./lib/validation.js";

const TRUST = [
  { Icon: SparkIcon, text: "Explained with a heatmap" },
  { Icon: ShieldIcon, text: "Calibrated, with a safety flag" },
  { Icon: LockIcon, text: "Photos never stored" },
];

export default function App() {
  const [meta, setMeta] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const request = useRef(null);
  const maxMb = meta?.max_upload_mb ?? DEFAULT_MAX_MB;

  useEffect(() => {
    const controller = new AbortController();
    // The page works without metadata; it only adds the model facts and limits.
    getMeta({ signal: controller.signal })
      .then(setMeta)
      .catch(() => {});
    return () => controller.abort();
  }, []);

  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  const analyse = useCallback(async (target) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await explain(target, { signal: controller.signal }));
    } catch (caught) {
      if (caught?.name !== "AbortError") setError(caught);
    } finally {
      if (request.current === controller) setBusy(false);
    }
  }, []);

  const reset = useCallback(() => {
    request.current?.abort();
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setBusy(false);
  }, []);

  const select = useCallback(
    (selected) => {
      const problem = validateFile(selected, maxMb);
      if (problem) {
        setError(problem);
        return;
      }
      reset();
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      analyse(selected);
    },
    [analyse, maxMb, reset],
  );

  const chooseSample = async (sample) => {
    try {
      select(await loadSample(sample));
    } catch {
      setError("The sample image could not be loaded. Please try again.");
    }
  };

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    const onPaste = (event) => {
      const pasted = [...(event.clipboardData?.files ?? [])].find((f) =>
        f.type.startsWith("image/"),
      );
      if (pasted) select(pasted);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [select]);

  const errorAlert = (
    <ErrorAlert
      error={error}
      onDismiss={() => setError(null)}
      onRetry={file && !busy ? () => analyse(file) : undefined}
    />
  );
  const showHeatmap = result && result.status !== "rejected" && result.heatmap_png;

  return (
    <div id="top" className="flex min-h-screen flex-col">
      <Disclaimer text={meta?.disclaimer} />
      <TopBar />

      <main className="flex-1">
        <section className="mx-auto max-w-3xl px-4 pt-16 pb-12 text-center sm:pt-20">
          <p className="reveal inline-flex items-center gap-2 rounded-full bg-surface px-3.5 py-1.5 text-xs font-semibold text-accent shadow-[var(--shadow-card)] ring-1 ring-line">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            Skin-cancer triage for dermoscopy images
          </p>
          {/* <h1 className="reveal reveal-1 mt-6 text-[2.6rem] leading-[1.04] font-extrabold tracking-[-0.035em] sm:text-6xl">
            See what the model sees.
          </h1> */}
          <p className="reveal reveal-2 mx-auto mt-5 max-w-xl text-base leading-relaxed text-ink-soft sm:text-lg">
            Add a dermoscopy photo of a spot.
          </p>
        </section>

        <section className="reveal reveal-3 mx-auto max-w-5xl px-4">
          <div className="rounded-[32px] bg-surface p-5 shadow-[var(--shadow-studio)] ring-1 ring-line/80 sm:p-8">
            {preview ? (
              <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:gap-10">
                <HeatmapViewer
                  imageSrc={preview}
                  heatmapPng={showHeatmap ? result.heatmap_png : null}
                  explainedClass={result?.explained_class}
                  busy={busy}
                  onClear={reset}
                />
                <div className="order-first lg:order-none lg:pt-1">
                  {busy && <AnalysisProgress />}
                  {!busy && result && (
                    <ResultCard
                      result={result}
                      malignantThreshold={
                        meta?.thresholds?.fitted ? meta.thresholds.malignant : null
                      }
                    />
                  )}
                  {!busy && errorAlert}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <UploadPanel busy={busy} maxMb={maxMb} onSelect={select} onSample={chooseSample} />
                <div className="mx-auto max-w-2xl">{errorAlert}</div>
              </div>
            )}
          </div>
          <ul className="mt-6 flex flex-wrap justify-center gap-x-7 gap-y-2">
            {TRUST.map(({ Icon, text }) => (
              <li key={text} className="flex items-center gap-2 text-[13px] text-ink-soft">
                <Icon className="h-4 w-4 text-accent" />
                {text}
              </li>
            ))}
          </ul>
        </section>

        <HowItWorks />
        <ModelFacts meta={meta} />
        <Privacy />
      </main>

      <Footer modelVersion={meta?.model_version} />
    </div>
  );
}
