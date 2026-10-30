import { useState } from "react";

import { applyTheme, currentTheme } from "../lib/theme.js";
import { MoonIcon, SunIcon } from "./icons.jsx";

/** Switches between the light and dark theme and remembers the choice. */
export default function ThemeToggle() {
  const [theme, setTheme] = useState(currentTheme);
  const next = theme === "dark" ? "light" : "dark";

  return (
    <button
      type="button"
      onClick={() => {
        applyTheme(next);
        setTheme(next);
      }}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className="grid h-9 w-9 place-items-center rounded-lg text-ink-soft ring-1 ring-line transition-colors hover:bg-sunken hover:text-ink"
    >
      {theme === "dark" ? (
        <SunIcon className="h-[18px] w-[18px]" />
      ) : (
        <MoonIcon className="h-[18px] w-[18px]" />
      )}
    </button>
  );
}
