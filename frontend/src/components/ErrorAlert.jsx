import { errorMessage } from "../lib/labels.js";

export default function ErrorAlert({ error, onDismiss }) {
  if (!error) return null;
  return (
    <div
      role="alert"
      className="flex items-start justify-between gap-4 rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100"
    >
      <p>{typeof error === "string" ? error : errorMessage(error)}</p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 font-medium underline underline-offset-2"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
