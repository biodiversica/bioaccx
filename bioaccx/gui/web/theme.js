/* Light / dark theme selection: follow the OS, or pin one.
 *
 * Nothing here names a colour. The stylesheet holds both palettes in
 * light-dark() tokens and `color-scheme` picks between them; all this does is
 * set `data-theme` on <html> and remember the choice.
 *
 * "system" is a real third state rather than the absence of a choice —
 * without it there is no way back to following the OS once a theme is pinned.
 */

const STORAGE_KEY = "bioaccx.theme";

const MODES = [
  { id: "system", glyph: "◐", title: "Theme: follows your system" },
  { id: "light", glyph: "☀", title: "Theme: light" },
  { id: "dark", glyph: "☾", title: "Theme: dark" },
];

/** The stored choice, tolerating storage that is disabled or holds junk. */
function stored() {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return MODES.some((m) => m.id === value) ? value : "system";
  } catch {
    return "system";
  }
}

function apply(mode) {
  if (mode === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = mode;
  try {
    if (mode === "system") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // A theme that cannot be remembered is still worth applying this session.
  }
}

/* Wire up the toolbar button, which cycles system → light → dark. The inline
 * script in index.html has already applied a pinned theme by now; this only
 * takes over the switching. */
export function initTheme(button) {
  let mode = stored();

  const paint = () => {
    const current = MODES.find((m) => m.id === mode);
    button.textContent = current.glyph;
    button.title = `${current.title} — click to change`;
    button.setAttribute("aria-label", current.title);
  };

  button.addEventListener("click", () => {
    mode = MODES[(MODES.findIndex((m) => m.id === mode) + 1) % MODES.length].id;
    apply(mode);
    paint();
  });

  apply(mode);
  paint();
}
