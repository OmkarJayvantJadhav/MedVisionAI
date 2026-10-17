import { useId, useState } from "react";

/**
 * The user's photo, with a scanning line while the model runs and the Grad-CAM
 * heatmap layered on top once it is back. The heatmap PNG has the photo's
 * aspect ratio, so stretching both to the container keeps them aligned;
 * opacity changes in the browser without another request.
 */
export default function HeatmapViewer({
  imageSrc,
  heatmapPng = null,
  explainedClass = null,
  busy = false,
  onClear,
}) {
  const sliderId = useId();
  const switchId = useId();
  const [opacity, setOpacity] = useState(0.45);
  const [visible, setVisible] = useState(true);

  return (
    <figure className="flex flex-col gap-4">
      <div className="relative overflow-hidden rounded-[20px] bg-sunken ring-1 ring-line">
        <img src={imageSrc} alt="Selected skin lesion" className="block w-full" />
        {heatmapPng && visible && (
          <img
            src={`data:image/png;base64,${heatmapPng}`}
            alt={`Heatmap of regions that influenced the ${explainedClass} score`}
            className="pointer-events-none absolute inset-0 h-full w-full transition-opacity duration-300"
            style={{ opacity }}
          />
        )}
        {busy && (
          <div aria-hidden="true" className="absolute inset-0 bg-canvas/20">
            <span className="animate-scan absolute inset-x-0 h-16 -translate-y-1/2 bg-gradient-to-b from-transparent via-accent/35 to-transparent" />
          </div>
        )}
      </div>

      {heatmapPng && (
        <div className="flex flex-col gap-3 rounded-2xl bg-sunken/60 px-4 py-3">
          <div className="flex items-center justify-between gap-3">
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
                className="peer sr-only"
              />
              <span
                aria-hidden="true"
                className="relative h-6 w-10 rounded-full bg-line transition peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent after:absolute after:top-0.5 after:left-0.5 after:h-5 after:w-5 after:rounded-full after:bg-white after:shadow after:transition peer-checked:after:translate-x-4"
              />
              <span>Show heatmap</span>
            </label>
            <div className="flex flex-1 items-center justify-end gap-3">
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
                onChange={(event) => setOpacity(Number(event.target.value))}
                className="h-1.5 w-full max-w-36 cursor-pointer accent-[var(--color-accent)] disabled:opacity-40"
              />
            </div>
          </div>
          <figcaption className="text-xs leading-relaxed text-ink-soft">
            Warmer colours mark the regions that most influenced the{" "}
            <strong className="font-semibold text-ink">{explainedClass}</strong> score. It shows
            where the model looked, not evidence of a diagnosis.
          </figcaption>
        </div>
      )}

      {onClear && (
        <button
          type="button"
          onClick={onClear}
          disabled={busy}
          className="self-start rounded-full px-4 py-2 text-sm font-semibold text-ink-soft ring-1 ring-line transition hover:bg-sunken hover:text-ink disabled:opacity-50"
        >
          Choose another
        </button>
      )}
    </figure>
  );
}
