import { useEffect, useRef, useState } from "react";

import { explain, getMeta } from "./api/client.js";
import Disclaimer from "./components/Disclaimer.jsx";
import ErrorAlert from "./components/ErrorAlert.jsx";
import ResultCard from "./components/ResultCard.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import { DEFAULT_MAX_MB, validateFile } from "./lib/validation.js";

export default function App() {
  const [meta, setMeta] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const request = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    // The page works without metadata; it only refines the upload limit and footer.
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

  const reset = () => {
    request.current?.abort();
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setBusy(false);
  };

  const select = (selected) => {
    const problem = validateFile(selected, meta?.max_upload_mb ?? DEFAULT_MAX_MB);
    if (problem) {
      setError(problem);
      return;
    }
    reset();
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const analyse = async () => {
    if (!file) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await explain(file, { signal: controller.signal }));
    } catch (caught) {
      if (caught?.name !== "AbortError") setError(caught);
    } finally {
      if (request.current === controller) setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Disclaimer text={meta?.disclaimer} />
      <header className="mx-auto w-full max-w-6xl px-4 pt-8 pb-4">
        <h1 className="text-3xl font-bold tracking-tight">MedVision AI</h1>
        <p className="mt-1 text-slate-600 dark:text-slate-400">
          Explainable skin-lesion classification with Grad-CAM heatmaps.
        </p>
      </header>

      <main className="mx-auto grid w-full max-w-6xl flex-1 gap-6 px-4 pb-10 lg:grid-cols-2">
        <div className="space-y-4">
          <UploadPanel
            preview={preview}
            busy={busy}
            onSelect={select}
            onAnalyze={analyse}
            onClear={reset}
          />
          <ErrorAlert error={error} onDismiss={() => setError(null)} />
        </div>
        <div>
          {result ? (
            <ResultCard result={result} imageSrc={preview} />
          ) : (
            <section className="flex h-full min-h-48 items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 p-6 text-center text-slate-500 dark:border-slate-800 dark:text-slate-400">
              {busy ? (
                <p role="status">Analysing the image…</p>
              ) : (
                <p>The result and heatmap will appear here.</p>
              )}
            </section>
          )}
        </div>
      </main>

      <footer className="border-t border-slate-200 px-4 py-6 text-center text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        {meta && <p>Model {meta.model_version}. </p>}
        <p>
          Images are processed in memory and never stored.{" "}
          <a
            href="https://github.com/OmkarJayvantJadhav/MedVisionAI/blob/main/docs/DATASETS.md"
            className="underline underline-offset-2"
          >
            Training data and licences
          </a>{" "}
          ·{" "}
          <a
            href="https://github.com/OmkarJayvantJadhav/MedVisionAI"
            className="underline underline-offset-2"
          >
            Source code
          </a>
        </p>
      </footer>
    </div>
  );
}
