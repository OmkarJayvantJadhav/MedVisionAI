import { GitHubIcon } from "./icons.jsx";

export const REPO_URL = "https://github.com/OmkarJayvantJadhav/MedVisionAI";

const LINKS = [
  { href: "#how-it-works", label: "How it works" },
  { href: "#model", label: "Model" },
  { href: "#privacy", label: "Privacy" },
];

function Logo() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className="h-8 w-8">
      <rect width="32" height="32" rx="9" className="fill-accent" />
      <circle cx="16" cy="16" r="7.5" fill="none" stroke="white" strokeWidth="2" opacity="0.55" />
      <circle cx="16" cy="16" r="3.5" fill="white" />
    </svg>
  );
}

export default function TopBar() {
  return (
    <header className="sticky top-0 z-20 border-b border-line/70 bg-canvas/85 backdrop-blur-md">
      <nav
        aria-label="Main"
        className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-4"
      >
        <a href="#top" className="flex items-center gap-2.5 rounded-lg">
          <Logo />
          <span className="text-[17px] font-bold tracking-tight">
            MedVision <span className="text-accent">AI</span>
          </span>
        </a>
        <div className="flex items-center gap-1 text-sm font-medium text-ink-soft">
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="hidden rounded-lg px-3 py-2 transition-colors hover:bg-sunken hover:text-ink md:block"
            >
              {link.label}
            </a>
          ))}
          <a
            href={REPO_URL}
            className="ml-1 flex items-center gap-2 rounded-lg px-3 py-2 transition-colors hover:bg-sunken hover:text-ink"
          >
            <GitHubIcon className="h-[18px] w-[18px]" />
            <span className="hidden sm:inline">Source</span>
          </a>
        </div>
      </nav>
    </header>
  );
}
