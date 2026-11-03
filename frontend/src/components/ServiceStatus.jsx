const STATES = {
  connecting: { dot: "bg-caution animate-pulse", label: "Starting up" },
  ready: { dot: "bg-benign", label: "Ready" },
  unavailable: { dot: "bg-alert", label: "Not available right now" },
};

/**
 * Readiness of the API and model, from the `/meta` request: the backend only
 * answers it once the model is loaded and checksum-verified (503 before that).
 */
export default function ServiceStatus({ state }) {
  const info = STATES[state] ?? STATES.connecting;
  return (
    <p
      role="status"
      className="flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium text-ink-soft ring-1 ring-line"
    >
      <span aria-hidden="true" className={`h-2 w-2 rounded-full ${info.dot}`} />
      <span>{info.label}</span>
    </p>
  );
}
