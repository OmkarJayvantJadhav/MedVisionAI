import { InfoIcon } from "./icons.jsx";

export const DISCLAIMER =
  "Decision support for clinicians: not a diagnosis or a medical device, and not " +
  "clinically validated. The referral decision rests with the clinician.";

export default function Disclaimer({ text = DISCLAIMER }) {
  return (
    <div
      role="note"
      aria-label="Medical disclaimer"
      className="border-b border-caution/25 bg-caution-soft text-caution"
    >
      <p className="mx-auto flex max-w-6xl items-start justify-center gap-2 px-4 py-2 text-center text-[13px] leading-snug font-medium sm:items-center">
        <InfoIcon className="mt-px h-4 w-4 shrink-0 sm:mt-0" />
        <span>{text}</span>
      </p>
    </div>
  );
}
