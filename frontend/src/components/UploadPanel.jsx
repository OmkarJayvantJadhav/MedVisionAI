import { useId, useState } from "react";

import { ACCEPTED_TYPES } from "../lib/validation.js";

/**
 * Choose an image by drag and drop, file picker or (on phones) the camera.
 * Calls `onSelect(file)`; the parent validates and previews it.
 */
export default function UploadPanel({ preview, busy, onSelect, onAnalyze, onClear }) {
  const inputId = useId();
  const cameraId = useId();
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files) => {
    if (files && files.length > 0) onSelect(files[0]);
  };

  return (
    <section
      aria-labelledby={`${inputId}-heading`}
      className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200 dark:bg-slate-900 dark:ring-slate-800"
    >
      <h2 id={`${inputId}-heading`} className="text-lg font-semibold">
        1. Choose a photo
      </h2>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
        A sharp, well-lit close-up of a single lesion works best. JPEG, PNG or WebP.
      </p>

      {preview ? (
        <img
          src={preview}
          alt="Selected skin lesion"
          className="mt-4 max-h-80 w-full rounded-xl object-contain ring-1 ring-slate-200 dark:ring-slate-700"
        />
      ) : (
        <label
          htmlFor={inputId}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            handleFiles(event.dataTransfer.files);
          }}
          className={`mt-4 flex h-56 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed text-center transition-colors focus-within:ring-2 focus-within:ring-teal-600 ${
            dragging
              ? "border-teal-600 bg-teal-50 dark:bg-teal-950"
              : "border-slate-300 hover:border-teal-600 dark:border-slate-700"
          }`}
        >
          <span className="text-base font-medium">Drop an image here</span>
          <span className="text-sm text-slate-500 dark:text-slate-400">or click to browse</span>
        </label>
      )}

      <input
        id={inputId}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="sr-only"
        aria-label="Choose an image file"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <input
        id={cameraId}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        aria-label="Take a photo"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />

      <div className="mt-4 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onAnalyze}
          disabled={!preview || busy}
          className="rounded-lg bg-teal-700 px-5 py-2.5 font-medium text-white shadow-sm hover:bg-teal-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Analysing…" : "Analyse image"}
        </button>
        <label
          htmlFor={cameraId}
          className="cursor-pointer rounded-lg px-4 py-2.5 font-medium text-teal-800 ring-1 ring-teal-700/40 hover:bg-teal-50 sm:hidden dark:text-teal-200 dark:hover:bg-teal-950"
        >
          Use camera
        </label>
        {preview && (
          <button
            type="button"
            onClick={onClear}
            disabled={busy}
            className="rounded-lg px-4 py-2.5 font-medium text-slate-700 ring-1 ring-slate-300 hover:bg-slate-100 disabled:opacity-50 dark:text-slate-200 dark:ring-slate-700 dark:hover:bg-slate-800"
          >
            Choose another
          </button>
        )}
      </div>
    </section>
  );
}
