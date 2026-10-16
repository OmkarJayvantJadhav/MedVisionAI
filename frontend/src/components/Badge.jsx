const TONES = {
  positive:
    "bg-emerald-100 text-emerald-900 ring-emerald-600/30 dark:bg-emerald-950 dark:text-emerald-200",
  caution: "bg-amber-100 text-amber-900 ring-amber-600/30 dark:bg-amber-950 dark:text-amber-200",
  danger: "bg-red-100 text-red-900 ring-red-600/30 dark:bg-red-950 dark:text-red-200",
  neutral: "bg-slate-100 text-slate-800 ring-slate-500/30 dark:bg-slate-800 dark:text-slate-200",
};

export default function Badge({ tone = "neutral", children }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset ${TONES[tone] ?? TONES.neutral}`}
    >
      {children}
    </span>
  );
}
