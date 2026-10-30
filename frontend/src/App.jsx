import { useCallback, useEffect, useRef, useState } from "react";

import { explain, getMeta } from "./api/client.js";
import { AnalysisProgress } from "./components/AnalysisPanel.jsx";
import Disclaimer from "./components/Disclaimer.jsx";
import ErrorAlert from "./components/ErrorAlert.jsx";
import Footer from "./components/Footer.jsx";
import HeatmapViewer from "./components/HeatmapViewer.jsx";
import InfoSections from "./components/InfoSections.jsx";
import { ImageIcon, ResetIcon } from "./components/icons.jsx";
import ResultCard from "./components/ResultCard.jsx";
import TopBar from "./components/TopBar.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import { loadSample } from "./lib/samples.js";
import { DEFAULT_MAX_MB, validateFile } from "./lib/validation.js";

/** While the model is starting or unreachable, ask `/meta` again this often. */
export const META_RETRY_MS = 15000;
const META_ATTEMPTS = 8;

function workspaceTitle({ busy, result, error }) {
  if (busy) return "Analysing";
  if (result) return "Result";
  if (error) return "Analysis failed";
  return "Image";
}

export default function App() {
  const [meta, setMeta] = useState(null);
  const [service, setService] = useState("connecting");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const request = useRef(null);
  const maxMb = meta?.max_upload_mb ?? DEFAULT_MAX_MB;

  // The page works without metadata; it adds the model facts, the referral
  // threshold and the readiness shown in the header. The backend answers
  // `/meta` only once the model is loaded, so retry while it starts.
  useEffect(() => {
    const controller = new AbortController();
    let timer;
    const load = (attempt) =>
      getMeta({ signal: controller.signal })
        .then((loaded) => {
          setMeta(loaded);
          setService("ready");
        })
        .catch((caught) => {
          if (caught?.name === "AbortError") return;
          setService("unavailable");
          if (attempt < META_ATTEMPTS) timer = setTimeout(() => load(attempt + 1), META_RETRY_MS);
        });
    load(1);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
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
    <div id="top" className="flex min-h-screen flex-col overflow-x-clip">
      <TopBar serviceState={service} modelVersion={meta?.model_version} />

      <main className="flex-1">
        <section className="mx-auto max-w-5xl px-4 pt-10 pb-6 sm:pt-12">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Dermoscopy Analysis</h1>
          <p className="mt-1.5 text-base text-ink-soft">AI-assisted skin lesion referral support</p>
          <div className="mt-5">
            <Disclaimer text={meta?.disclaimer} />
          </div>
        </section>

        <section aria-label="Analysis workspace" className="mx-auto max-w-5xl px-4">
          <div
            aria-busy={busy}
            className="rounded-2xl border border-line bg-surface shadow-[var(--shadow-workspace)]"
          >
            {preview ? (
              <>
                <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3 sm:px-6">
                  <div className="flex min-w-0 items-center gap-2.5">
                    <ImageIcon className="h-5 w-5 shrink-0 text-ink-soft" />
                    <p className="min-w-0 truncate text-sm">
                      <span className="font-semibold">
                        {workspaceTitle({ busy, result, error })}
                      </span>
                      {file?.name && <span className="text-ink-soft"> · {file.name}</span>}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={reset}
                    disabled={busy}
                    className="inline-flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold text-accent ring-1 ring-line transition-colors hover:bg-accent-soft disabled:opacity-50"
                  >
                    <ResetIcon className="h-4 w-4" />
                    New analysis
                  </button>
                </div>
                <div className="grid items-start gap-8 p-4 sm:p-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:gap-10">
                  <HeatmapViewer
                    imageSrc={preview}
                    heatmapPng={showHeatmap ? result.heatmap_png : null}
                    explainedClass={result?.explained_class}
                    busy={busy}
                  />
                  <div className="order-first min-w-0 lg:order-none">
                    {busy && <AnalysisProgress />}
                    {!busy && result && (
                      <ResultCard
                        result={result}
                        malignantThreshold={
                          meta?.thresholds?.fitted ? meta.thresholds.malignant : null
                        }
                        malignantClasses={meta?.malignant_classes}
                      />
                    )}
                    {!busy && errorAlert}
                  </div>
                </div>
              </>
            ) : (
              <div className="space-y-4 p-4 sm:p-6">
                <UploadPanel busy={busy} maxMb={maxMb} onSelect={select} onSample={chooseSample} />
                {errorAlert}
              </div>
            )}
          </div>
        </section>

        <InfoSections meta={meta} />
      </main>

      <Footer modelVersion={meta?.model_version} />
    </div>
  );
}
