import { useCallback, useEffect, useRef, useState } from "react";

import { explain, getMeta } from "./api/client.js";
import { AnalysisProgress, ResultPlaceholder } from "./components/AnalysisPanel.jsx";
import Disclaimer from "./components/Disclaimer.jsx";
import ErrorAlert from "./components/ErrorAlert.jsx";
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

  let resultPanel = <ResultPlaceholder />;
  if (busy) resultPanel = <AnalysisProgress />;
  else if (result)
    resultPanel = (
      <ResultCard
        result={result}
        imageSrc={preview}
        malignantThreshold={meta?.thresholds?.fitted ? meta.thresholds.malignant : null}
      />
    );

  return (
    <div id="top" className="flex min-h-screen flex-col">
      <Disclaimer text={meta?.disclaimer} />
      <TopBar />

      <main className="flex-1">
        <section className="mx-auto max-w-6xl px-4 pt-10 pb-14 sm:pt-14">
          <div className="max-w-3xl">
            <p className="text-xs font-semibold tracking-[0.1em] text-accent uppercase">
              Skin-lesion screening research
            </p>
            <h1 className="mt-3 text-4xl leading-[1.08] font-extrabold tracking-tight sm:text-5xl">
              Check a skin-lesion photo and see what the model sees.
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-relaxed text-ink-soft sm:text-lg">
              Upload a close-up of a mole or spot. Within seconds you get the closest matches, a
              calibrated malignancy flag and a heatmap of what drove the result.
            </p>
            <ul className="mt-6 flex flex-wrap gap-x-6 gap-y-2.5">
              {TRUST.map(({ Icon, text }) => (
                <li
                  key={text}
                  className="flex items-center gap-2 text-sm font-medium text-ink-soft"
                >
                  <Icon className="h-[18px] w-[18px] text-accent" />
                  {text}
                </li>
              ))}
            </ul>
          </div>

          <div className="mt-10 grid items-start gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
            <div className="space-y-4">
              <UploadPanel
                preview={preview}
                busy={busy}
                maxMb={maxMb}
                onSelect={select}
                onSample={chooseSample}
                onClear={reset}
              />
              <ErrorAlert
                error={error}
                onDismiss={() => setError(null)}
                onRetry={file && !busy ? () => analyse(file) : undefined}
              />
            </div>
            <div className="lg:sticky lg:top-24">{resultPanel}</div>
          </div>
        </section>

        <HowItWorks />
        <ModelFacts meta={meta} />
        <Privacy />
      </main>

      <Footer modelVersion={meta?.model_version} />
    </div>
  );
}
