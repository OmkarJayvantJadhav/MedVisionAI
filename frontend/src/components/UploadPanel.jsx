import { useId, useRef, useState } from "react";

import { SAMPLES } from "../lib/samples.js";
import { ACCEPTED_TYPES, DEFAULT_MAX_MB } from "../lib/validation.js";
import { InfoIcon, UploadIcon } from "./icons.jsx";

function SampleList({ busy, onSample }) {
  return (
    <div>
      <h3 className="text-sm font-semibold">Or use a sample image</h3>
      <p className="mt-0.5 text-xs text-ink-soft">
        Held-out ISIC 2019 test images (CC BY-NC 4.0), never used in training.
      </p>
      <ul className="mt-3 grid grid-cols-3 gap-3">
        {SAMPLES.map((sample) => (
          <li key={sample.id} className="min-w-0">
            <button
              type="button"
              onClick={() => onSample(sample)}
              disabled={busy}
              className="group flex w-full flex-col overflow-hidden rounded-xl bg-surface text-left ring-1 ring-line transition hover:ring-accent disabled:pointer-events-none disabled:opacity-60"
            >
              <img
                src={sample.file}
                alt=""
                loading="lazy"
                className="aspect-[4/3] w-full object-cover"
              />
              <span className="truncate border-t border-line px-2.5 py-2 text-xs font-medium text-ink-soft group-hover:text-ink sm:text-[13px]">
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
 * The empty state of the workspace: drop, browse or paste a dermoscopy image,
 * or pick a sample. Calls `onSelect(file)` or `onSample(sample)`; the file is
 * validated by the caller.
 */
export default function UploadPanel({ busy = false, maxMb = DEFAULT_MAX_MB, onSelect, onSample }) {
  const inputId = useId();
  const hintId = useId();
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files) => {
    if (files && files.length > 0) onSelect(files[0]);
  };

  return (
    <section aria-label="Choose a dermoscopy image" className="flex flex-col gap-6">
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
        className={`flex flex-col items-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-line-strong bg-sunken/50"
        }`}
      >
        <span className="grid h-12 w-12 place-items-center rounded-xl bg-accent-soft text-accent">
          <UploadIcon className="h-6 w-6" />
        </span>
        <p className="mt-4 text-base font-semibold">Drop a dermoscopy image here</p>
        <p className="mt-1 text-sm text-ink-soft">
          or select a file<span className="hidden sm:inline">, or paste with Ctrl+V</span>
        </p>
        <button
          type="button"
          onClick={() => input.current?.click()}
          disabled={busy}
          aria-describedby={hintId}
          className="mt-5 inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent shadow-sm transition-colors hover:bg-accent-strong disabled:opacity-60"
        >
          Select image
        </button>
        <p id={hintId} className="mt-4 flex flex-wrap justify-center gap-2 text-xs text-ink-soft">
          <span className="rounded-md bg-surface px-2 py-1 ring-1 ring-line">
            JPEG, PNG or WebP
          </span>
          <span className="rounded-md bg-surface px-2 py-1 ring-1 ring-line">Up to {maxMb} MB</span>
          <span className="rounded-md bg-surface px-2 py-1 ring-1 ring-line">One lesion</span>
        </p>
      </div>

      <div className="flex items-start gap-3 rounded-xl bg-accent-soft px-4 py-3 text-sm leading-relaxed">
        <InfoIcon className="mt-0.5 h-5 w-5 shrink-0 text-accent" />
        <p>
          <strong className="font-semibold">Dermoscopy images only.</strong> Use an image of a
          single lesion taken through a dermatoscope. Ordinary clinical or smartphone photos are
          outside what the model was trained on and are rejected rather than assessed.
        </p>
      </div>

      <SampleList busy={busy} onSample={onSample} />

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
    </section>
  );
}
