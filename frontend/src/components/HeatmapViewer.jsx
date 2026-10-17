import { useId, useState } from "react";

/**
 * The uploaded photo with the Grad-CAM heatmap layered on top. The heatmap PNG
 * has the same aspect ratio as the photo, so stretching both to the container
 * width keeps them aligned; opacity is adjusted in the browser.
 */
export default function HeatmapViewer({ imageSrc, heatmapPng, explainedClass }) {
  const sliderId = useId();
  const switchId = useId();
  const [opacity, setOpacity] = useState(0.45);
  const [visible, setVisible] = useState(true);

  return (
    <figure className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold">Where the model looked</p>
        <label htmlFor={switchId} className="flex cursor-pointer items-center gap-2.5 text-sm">
          <span className="text-ink-soft">Show heatmap</span>
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
        </label>
      </div>
      <div className="relative overflow-hidden rounded-2xl bg-sunken">
        <img src={imageSrc} alt="Uploaded skin lesion" className="block w-full" />
        {visible && (
          <img
            src={`data:image/png;base64,${heatmapPng}`}
            alt={`Heatmap of regions that influenced the ${explainedClass} score`}
            className="pointer-events-none absolute inset-0 h-full w-full transition-opacity"
            style={{ opacity }}
          />
        )}
      </div>
      <div className="flex items-center gap-3">
        <label htmlFor={sliderId} className="text-sm text-ink-soft">
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
          className="h-1.5 flex-1 cursor-pointer accent-[var(--color-accent)] disabled:opacity-40"
        />
      </div>
      <figcaption className="text-xs leading-relaxed text-ink-soft">
        Warmer colours mark the regions that most influenced the{" "}
        <strong className="font-semibold text-ink">{explainedClass}</strong> score. The heatmap
        shows what the model looked at; it is not evidence of a diagnosis.
      </figcaption>
    </figure>
  );
}
