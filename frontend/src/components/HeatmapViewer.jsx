import { useId, useState } from "react";

/**
 * The uploaded photo with the Grad-CAM heatmap layered on top. The heatmap PNG
 * has the same aspect ratio as the photo, so stretching both to the container
 * width keeps them aligned; opacity is adjusted in the browser.
 */
export default function HeatmapViewer({ imageSrc, heatmapPng, explainedClass }) {
  const sliderId = useId();
  const [opacity, setOpacity] = useState(0.45);
  const [visible, setVisible] = useState(true);

  return (
    <figure>
      <div className="relative overflow-hidden rounded-xl ring-1 ring-slate-200 dark:ring-slate-700">
        <img src={imageSrc} alt="Uploaded skin lesion" className="block w-full" />
        {visible && (
          <img
            src={`data:image/png;base64,${heatmapPng}`}
            alt={`Heatmap of regions that influenced the ${explainedClass} score`}
            className="pointer-events-none absolute inset-0 h-full w-full"
            style={{ opacity }}
          />
        )}
      </div>
      <figcaption className="mt-2 text-sm text-slate-600 dark:text-slate-400">
        Warmer colours mark regions that most influenced the <strong>{explainedClass}</strong>{" "}
        score. A heatmap shows what the model looked at; it is not evidence of a diagnosis.
      </figcaption>
      <div className="mt-3 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={visible}
            onChange={(event) => setVisible(event.target.checked)}
            className="h-4 w-4 accent-teal-700"
          />
          Show heatmap
        </label>
        <div className="flex items-center gap-2">
          <label htmlFor={sliderId} className="text-sm">
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
            className="w-40 accent-teal-700"
          />
        </div>
      </div>
    </figure>
  );
}
