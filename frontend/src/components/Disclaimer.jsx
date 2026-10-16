export const DISCLAIMER =
  "Not a medical diagnosis. This tool is for education and screening research only. " +
  "Please consult a qualified dermatologist about any skin concern.";

export default function Disclaimer({ text = DISCLAIMER }) {
  return (
    <div
      role="note"
      aria-label="Medical disclaimer"
      className="border-b border-amber-300 bg-amber-50 px-4 py-2 text-center text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100"
    >
      <strong className="font-semibold">Disclaimer:</strong> {text}
    </div>
  );
}
