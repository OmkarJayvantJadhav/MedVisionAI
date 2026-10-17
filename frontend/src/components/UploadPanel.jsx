import { useId, useRef, useState } from "react";

import { SAMPLES } from "../lib/samples.js";
import { ACCEPTED_TYPES, DEFAULT_MAX_MB } from "../lib/validation.js";
import { CameraIcon, UploadIcon } from "./icons.jsx";

function SampleRow({ busy, onSample }) {
  return (
    <div className="mt-8">
      <div className="flex items-center gap-4">
        <span className="h-px flex-1 bg-line" />
        <p className="text-xs font-medium tracking-[0.12em] text-ink-faint uppercase">
          Or try a sample
        </p>
        <span className="h-px flex-1 bg-line" />
      </div>
      <ul className="mt-5 grid grid-cols-3 gap-3 sm:gap-4">
        {SAMPLES.map((sample) => (
          <li key={sample.id}>
            <button
              type="button"
              onClick={() => onSample(sample)}
              disabled={busy}
              className="group w-full text-left disabled:pointer-events-none disabled:opacity-60"
            >
              <span className="block overflow-hidden rounded-2xl ring-1 ring-line transition duration-300 group-hover:-translate-y-0.5 group-hover:shadow-[var(--shadow-card)] group-hover:ring-accent/50">
                <img
                  src={sample.file}
                  alt=""
                  loading="lazy"
                  className="aspect-square w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                />
              </span>
              <span className="mt-2 block truncate text-center text-xs font-medium text-ink-soft group-hover:text-ink sm:text-[13px]">
                {sample.label}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * The empty state of the studio: drop, browse, paste, use the camera (phones)
 * or pick a sample. Calls `onSelect(file)` or `onSample(sample)`.
 */
export default function UploadPanel({ busy = false, maxMb = DEFAULT_MAX_MB, onSelect, onSample }) {
  const inputId = useId();
  const cameraId = useId();
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files) => {
    if (files && files.length > 0) onSelect(files[0]);
  };

  return (
    <section aria-label="Choose a photo" className="mx-auto max-w-2xl">
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
        className={`rounded-[24px] border border-dashed transition duration-300 ${
          dragging
            ? "scale-[1.01] border-accent bg-accent-soft"
            : "border-ink-faint/40 bg-sunken/40 hover:border-accent/60 hover:bg-accent-soft/40"
        }`}
      >
        <button
          type="button"
          onClick={() => input.current?.click()}
          className="group flex w-full flex-col items-center justify-center rounded-[24px] px-6 py-12 text-center sm:py-16"
        >
          <span className="grid h-16 w-16 place-items-center rounded-2xl bg-surface text-accent shadow-[var(--shadow-card)] ring-1 ring-line transition duration-300 group-hover:-translate-y-1">
            <UploadIcon className="h-7 w-7" />
          </span>
          <span className="mt-5 text-lg font-semibold tracking-tight">Drop an image here</span>
          <span className="mt-1.5 text-sm text-ink-soft">
            or <span className="font-semibold text-accent">browse your files</span>
            <span className="hidden sm:inline"> · paste with Ctrl+V</span>
          </span>
          <span className="mt-4 text-xs text-ink-faint">
            Dermoscopy close-up · JPEG, PNG or WebP · up to {maxMb} MB
          </span>
        </button>
      </div>

      <label
        htmlFor={cameraId}
        className="mt-3 flex w-full cursor-pointer items-center justify-center gap-2 rounded-full bg-ink px-4 py-3 text-sm font-semibold text-canvas transition hover:opacity-90 sm:hidden"
      >
        <CameraIcon className="h-[18px] w-[18px]" />
        Use camera
      </label>

      <SampleRow busy={busy} onSample={onSample} />
      <p className="mt-4 text-center text-[11px] text-ink-faint">
        Samples are ISIC 2019 test images (CC BY-NC 4.0), never used in training.
      </p>

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
