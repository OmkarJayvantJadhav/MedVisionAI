import { errorMessage } from "../lib/labels.js";
import { AlertIcon } from "./icons.jsx";

export default function ErrorAlert({ error, onDismiss, onRetry }) {
  if (!error) return null;
  return (
    <div
      role="alert"
      className="animate-fade-up flex items-start gap-3 rounded-2xl border border-alert/30 bg-alert-soft p-4 text-sm text-ink"
    >
      <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-alert" />
      <p className="flex-1 leading-relaxed">
        {typeof error === "string" ? error : errorMessage(error)}
      </p>
      <div className="flex shrink-0 gap-3 font-semibold">
        {onRetry && (
          <button type="button" onClick={onRetry} className="text-alert hover:underline">
            Try again
          </button>
        )}
        {onDismiss && (
          <button type="button" onClick={onDismiss} className="text-ink-soft hover:underline">
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
