/**
 * Light/dark theme. With no saved choice the page follows the system setting;
 * a choice is stored per browser and applied as `data-theme` on <html>. The same
 * logic runs as an inline script in index.html before the first paint, so the
 * page never flashes the wrong theme. Storage can be unavailable (private
 * windows, blocked site data), so every access is guarded.
 */

export const THEME_KEY = "medvision-theme";

export function storedTheme() {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function systemTheme() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** The theme currently shown: the saved choice, otherwise the system theme. */
export function currentTheme() {
  return storedTheme() ?? systemTheme();
}

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // The choice still applies for this page view.
  }
}
