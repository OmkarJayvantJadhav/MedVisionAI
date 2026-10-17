const TONES = {
  positive: "bg-benign-soft text-benign ring-benign/25",
  caution: "bg-caution-soft text-caution ring-caution/25",
  danger: "bg-alert-soft text-alert ring-alert/25",
  neutral: "bg-sunken text-ink-soft ring-line",
};

export default function Badge({ tone = "neutral", children }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${TONES[tone] ?? TONES.neutral}`}
    >
      {children}
    </span>
  );
}
