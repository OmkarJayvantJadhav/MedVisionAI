import { ShieldIcon } from "./icons.jsx";

export const DISCLAIMER =
  "Decision support for clinicians: not a diagnosis or a medical device, and not " +
  "clinically validated. The referral decision rests with the clinician.";

/** The clinical disclaimer. The text comes from `/meta` when the API is reachable. */
export default function Disclaimer({ text = DISCLAIMER }) {
  return (
    <div
      role="note"
      aria-label="Medical disclaimer"
      className="flex items-start gap-3 rounded-xl border border-caution/30 bg-caution-soft px-4 py-3 text-sm leading-relaxed text-ink"
    >
      <ShieldIcon className="mt-0.5 h-5 w-5 shrink-0 text-caution" />
      <p>
        <strong className="font-semibold">Clinical disclaimer. </strong>
        {text}
      </p>
    </div>
  );
}
