import { useId, useRef, useState } from "react";

import { SAMPLES } from "../lib/samples.js";
import { ACCEPTED_TYPES, DEFAULT_MAX_MB } from "../lib/validation.js";
import { CameraIcon, UploadIcon } from "./icons.jsx";

function SampleRow({ busy, onSample }) {
  return (
    <div className="mt-5">
      <p className="text-[13px] font-medium text-ink-soft">No photo to hand? Try a sample:</p>
      <ul className="mt-2.5 grid grid-cols-3 gap-2.5">
        {SAMPLES.map((sample) => (
          <li key={sample.id}>
            <button
              type="button"
              onClick={() => onSample(sample)}
              disabled={busy}
              className="group w-full overflow-hidden rounded-xl border border-line bg-surface text-left transition hover:-translate-y-0.5 hover:border-accent/60 hover:shadow-[var(--shadow-card)] disabled:pointer-events-none disabled:opacity-60"
            >
              <img
                src={sample.file}
                alt=""
                loading="lazy"
                className="aspect-[4/3] w-full object-cover transition group-hover:brightness-105"
              />
              <span className="block truncate px-2.5 py-2 text-xs font-medium">{sample.label}</span>
            </button>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[11px] text-ink-faint">
        Images from the ISIC 2019 test set (CC BY-NC 4.0), never used in training.
      </p>
    </div>
  );
}

/**
 * Choose a photo by drag and drop, the file picker, the camera (phones) or a
 * sample. Calls `onSelect(file)`; the parent validates, previews and analyses it.
 */
export default function UploadPanel({
  preview,
  busy,
  maxMb = DEFAULT_MAX_MB,
  onSelect,
  onSample,
  onClear,
}) {
  const inputId = useId();
  const cameraId = useId();
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files) => {
    if (files && files.length > 0) onSelect(files[0]);
  };

  return (
    <section
      aria-labelledby={`${inputId}-heading`}
      className="rounded-3xl border border-line bg-surface p-5 shadow-[var(--shadow-card)] sm:p-7"
    >
      <div className="flex items-baseline justify-between gap-3">
        <h2 id={`${inputId}-heading`} className="text-lg font-semibold tracking-tight">
          Your photo
        </h2>
        <span className="text-xs text-ink-faint">JPEG, PNG or WebP · up to {maxMb} MB</span>
      </div>

      {preview ? (
        <div className="mt-4">
          <div className="relative overflow-hidden rounded-2xl bg-sunken">
            <img
              src={preview}
              alt="Selected skin lesion"
              className="mx-auto max-h-[22rem] w-full object-contain"
            />
            {busy && (
              <div className="absolute inset-0 grid place-items-center bg-canvas/40 backdrop-blur-[2px]">
                <span className="h-10 w-10 animate-spin rounded-full border-[3px] border-accent/25 border-t-accent" />
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={onClear}
            disabled={busy}
            className="mt-4 w-full rounded-xl border border-line px-4 py-2.5 text-sm font-semibold text-ink transition hover:bg-sunken disabled:opacity-50 sm:w-auto"
          >
            Choose another
          </button>
        </div>
      ) : (
        <>
          <div
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
            data-dragging={dragging}
            className={`mt-4 rounded-2xl border-2 border-dashed transition-colors ${
              dragging ? "border-accent bg-accent-soft" : "border-line hover:border-accent/60"
            }`}
          >
            <button
              type="button"
              onClick={() => input.current?.click()}
              className="flex w-full flex-col items-center justify-center gap-3 rounded-2xl px-6 py-10 text-center sm:py-14"
            >
              <span className="grid h-14 w-14 place-items-center rounded-2xl bg-accent-soft text-accent">
                <UploadIcon className="h-6 w-6" />
              </span>
              <span className="text-base font-semibold">Drop an image here</span>
              <span className="text-sm text-ink-soft">
                or <span className="font-semibold text-accent">browse your files</span>
                <span className="hidden sm:inline"> · you can also paste with Ctrl+V</span>
              </span>
            </button>
          </div>
          <label
            htmlFor={cameraId}
            className="mt-3 flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl bg-accent px-4 py-3 text-sm font-semibold text-white transition hover:bg-accent-strong sm:hidden"
          >
            <CameraIcon className="h-[18px] w-[18px]" />
            Use camera
          </label>
          <SampleRow busy={busy} onSample={onSample} />
        </>
      )}

      <input
        id={inputId}
        ref={input}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="sr-only"
        tabIndex={-1}
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
        tabIndex={-1}
        aria-label="Take a photo"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
    </section>
  );
}
