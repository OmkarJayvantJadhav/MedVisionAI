import { useId, useState } from "react";

/**
 * The selected image, with a scanning line while the request runs and the
 * Grad-CAM heatmap from the response layered on top. The heatmap PNG has the
 * image's aspect ratio (EXIF orientation applied on both sides), so stretching
 * both to the same box keeps them aligned; the toggle and opacity work in the
 * browser without another request.
 */
export default function HeatmapViewer({
  imageSrc,
  heatmapPng = null,
  explainedClass = null,
  busy = false,
}) {
  const sliderId = useId();
  const switchId = useId();
  const noteId = useId();
  const [opacity, setOpacity] = useState(0.45);
  const [visible, setVisible] = useState(true);

  return (
    <figure className="flex min-w-0 flex-col gap-4 lg:sticky lg:top-24">
      <div className="relative mx-auto w-full max-w-xl overflow-hidden rounded-xl bg-sunken ring-1 ring-line">
        <img src={imageSrc} alt="Selected skin lesion" className="block h-auto w-full" />
        {heatmapPng && visible && (
          <img
            src={`data:image/png;base64,${heatmapPng}`}
            alt={`Heatmap of regions that influenced the ${explainedClass} score`}
            className="pointer-events-none absolute inset-0 h-full w-full"
            style={{ opacity }}
          />
        )}
        {busy && (
          <div aria-hidden="true" className="absolute inset-0 bg-canvas/15">
            <span className="animate-scan absolute inset-x-0 h-0.5 bg-accent shadow-[0_0_14px_4px_var(--color-accent)]" />
          </div>
        )}
      </div>

      {heatmapPng && (
        <div className="mx-auto flex w-full max-w-xl flex-col gap-3 rounded-xl border border-line px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
            <label
              htmlFor={switchId}
              className="flex cursor-pointer items-center gap-2.5 text-sm font-medium"
            >
              <input
                id={switchId}
                type="checkbox"
                role="switch"
                checked={visible}
                onChange={(event) => setVisible(event.target.checked)}
                aria-describedby={noteId}
                className="peer sr-only"
              />
              <span
                aria-hidden="true"
                className="relative h-6 w-10 shrink-0 rounded-full bg-line-strong transition peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent after:absolute after:top-0.5 after:left-0.5 after:h-5 after:w-5 after:rounded-full after:bg-white after:shadow after:transition peer-checked:after:translate-x-4"
              />
              <span>Show heatmap</span>
            </label>
            <div className="flex min-w-0 flex-1 items-center justify-end gap-3">
              <label htmlFor={sliderId} className="text-xs text-ink-soft">
                Opacity
              </label>
              <input
                id={sliderId}
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                disabled={!visible}
                aria-valuetext={`${Math.round(opacity * 100)}%`}
                onChange={(event) => setOpacity(Number(event.target.value))}
                className="h-1.5 w-full max-w-40 cursor-pointer accent-[var(--color-accent)] disabled:opacity-40"
              />
              <span className="w-9 text-right font-mono text-xs text-ink-soft tabular-nums">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          </div>
          <div aria-hidden="true" className="flex items-center gap-2 text-[11px] text-ink-soft">
            <span>Lower</span>
            <span className="jet-scale h-1.5 flex-1 rounded-full" />
            <span>Higher influence</span>
          </div>
          <figcaption id={noteId} className="text-xs leading-relaxed text-ink-soft">
            Warmer colours mark the regions that most influenced the{" "}
            <strong className="font-semibold text-ink">{explainedClass}</strong> score. The heatmap
            shows where the model&apos;s attention was; it is not a diagnosis and not an outline of
            the lesion or of cancer.
          </figcaption>
        </div>
      )}
    </figure>
  );
}
