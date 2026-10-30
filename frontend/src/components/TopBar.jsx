import { GitHubIcon } from "./icons.jsx";
import ServiceStatus from "./ServiceStatus.jsx";
import ThemeToggle from "./ThemeToggle.jsx";

export const REPO_URL = "https://github.com/OmkarJayvantJadhav/MedVisionAI";

function Logo() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className="h-8 w-8 shrink-0">
      <rect width="32" height="32" rx="8" className="fill-accent" />
      <circle cx="16" cy="16" r="7" fill="none" className="stroke-on-accent" strokeWidth="2.5" />
      <circle cx="16" cy="16" r="2.5" className="fill-on-accent" />
    </svg>
  );
}

/** Branding, the application name, model readiness and the theme switch. */
export default function TopBar({ serviceState, modelVersion }) {
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-3 px-4">
        <a href="#top" className="flex min-w-0 items-center gap-3 rounded-lg">
          <Logo />
          <span className="flex min-w-0 flex-col leading-tight">
            <span className="text-[15px] font-bold tracking-tight">MedVision AI</span>
            <span className="hidden truncate text-xs text-ink-soft sm:block">
              Dermoscopy referral support
            </span>
          </span>
        </a>
        <div className="flex items-center gap-2">
          <ServiceStatus state={serviceState} modelVersion={modelVersion} />
          <ThemeToggle />
          <a
            href={REPO_URL}
            aria-label="Source code on GitHub"
            title="Source code on GitHub"
            className="hidden h-9 w-9 place-items-center rounded-lg text-ink-soft ring-1 ring-line transition-colors hover:bg-sunken hover:text-ink sm:grid"
          >
            <GitHubIcon className="h-[18px] w-[18px]" />
          </a>
        </div>
      </div>
    </header>
  );
}
