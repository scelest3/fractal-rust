/**
 * AboutPanel — modal dialog with app name, GitHub link, and UI hints.
 *
 * Open via the "?" button in the stats overlay or by pressing "?".
 * Close via the × button, Escape, or clicking the backdrop.
 */

const REPO_URL = "https://github.com/scelest3/fractal-rust";

const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

const TOUCH_HINTS: { keys: string; desc: string }[] = [
  { keys: "Drag",              desc: "Pan" },
  { keys: "Pinch",             desc: "Zoom in / out" },
  { keys: "Long-press + drag", desc: "Box zoom — draw region to fill viewport" },
  { keys: "URL hash",          desc: "Bookmark · share the current view" },
];

const KEYBOARD_ONLY_HINTS: { keys: string; desc: string }[] = [
  { keys: "+ / −",      desc: "Zoom in / out at center" },
  { keys: "Arrow keys", desc: "Pan (20% of viewport per press)" },
  { keys: "R",          desc: "Reset view to default" },
  { keys: "C",          desc: "Toggle color editor" },
  { keys: "H",          desc: "Toggle HiDPI rendering" },
  { keys: "E",          desc: "Export image as PNG" },
  { keys: "?",          desc: "Toggle this panel" },
];

const KEYBOARD_HINTS: { keys: string; desc: string }[] = [
  { keys: "Scroll wheel", desc: "Zoom in / out at cursor" },
  { keys: "+ / −",        desc: "Zoom in / out at center" },
  { keys: "Left-drag",    desc: "Pan" },
  { keys: "Arrow keys",   desc: "Pan (20% of viewport per press)" },
  { keys: "Right-drag",   desc: "Box zoom — draw region to fill viewport" },
  { keys: "R",            desc: "Reset view to default" },
  { keys: "C",            desc: "Toggle color editor" },
  { keys: "H",            desc: "Toggle HiDPI rendering" },
  { keys: "E",            desc: "Export image as PNG" },
  { keys: "?",            desc: "Toggle this panel" },
  { keys: "URL hash",     desc: "Bookmark · share the current view" },
];

export class AboutPanel {
  private readonly backdrop: HTMLDivElement;
  private readonly dialog: HTMLDivElement;
  private readonly hintsContainer: HTMLDivElement;
  private open = false;

  constructor() {
    this.backdrop = this.makeBackdrop();
    this.dialog   = this.makeDialog();
    this.hintsContainer = this.makeHintsContainer();
    this.dialog.appendChild(this.hintsContainer);
    this.backdrop.appendChild(this.dialog);
    document.body.appendChild(this.backdrop);

    this.backdrop.addEventListener("click", (e) => {
      if (e.target === this.backdrop) this.hide();
    });
    document.addEventListener("keydown", (e) => {
      if (this.open && e.key === "Escape") this.hide();
    });
  }

  show(): void {
    this.open = true;
    this.hintsContainer.replaceChildren(
      ...Array.from(this.makeHintsContainer().childNodes),
    );
    this.backdrop.style.display = "flex";
  }

  hide(): void {
    this.open = false;
    this.backdrop.style.display = "none";
  }

  toggle(): void {
    this.open ? this.hide() : this.show();
  }

  /** Returns a title row — "Fractal Rust" label + clickable "?" icon — for the top of the overlay. */
  makeTitleHeader(): HTMLElement {
    const row = document.createElement("div");
    Object.assign(row.style, {
      display: "flex", alignItems: "center", justifyContent: "space-between",
      marginBottom: "6px",
    });

    const title = document.createElement("span");
    title.textContent = "Fractal Rust";
    Object.assign(title.style, {
      color: "#60e8ff", fontWeight: "bold", fontSize: "13px", letterSpacing: "0.04em",
    });

    const icon = document.createElement("button");
    icon.textContent = "?";
    icon.title = "About / keyboard shortcuts";
    Object.assign(icon.style, {
      background: "transparent", border: "none",
      color: "rgba(255,255,255,0.45)", fontSize: "13px", fontFamily: "monospace",
      cursor: "pointer", padding: "0 0 0 6px", lineHeight: "1",
    });
    icon.addEventListener("mouseover", () => icon.style.color = "rgba(255,255,255,0.85)");
    icon.addEventListener("mouseout",  () => icon.style.color = "rgba(255,255,255,0.45)");
    icon.addEventListener("click", () => this.show());

    row.appendChild(title);
    row.appendChild(icon);
    return row;
  }

  private makeBackdrop(): HTMLDivElement {
    const el = document.createElement("div");
    Object.assign(el.style, {
      display: "none",
      position: "fixed", inset: "0",
      background: "rgba(0,0,0,0.65)",
      zIndex: "1100",
      alignItems: "center", justifyContent: "center",
    });
    return el;
  }

  private makeDialog(): HTMLDivElement {
    const el = document.createElement("div");
    Object.assign(el.style, {
      background: "#0e0e1a",
      color: "#e0e0f0",
      fontFamily: "monospace",
      fontSize: "13px",
      borderRadius: "8px",
      border: "1px solid rgba(255,255,255,0.15)",
      padding: "24px 28px",
      maxWidth: "420px",
      width: "calc(100vw - 48px)",
      lineHeight: "1.6",
      boxShadow: "0 8px 40px rgba(0,0,0,0.7)",
      position: "relative",
      maxHeight: "calc(100vh - 48px)",
      overflowY: "auto",
    });

    // Close button
    const close = document.createElement("button");
    close.textContent = "×";
    Object.assign(close.style, {
      position: "absolute", top: "12px", right: "16px",
      background: "transparent", border: "none",
      color: "rgba(255,255,255,0.5)", fontSize: "20px",
      cursor: "pointer", lineHeight: "1", padding: "0",
    });
    close.addEventListener("click", () => this.hide());
    el.appendChild(close);

    // App name
    const title = document.createElement("div");
    title.textContent = "Fractal Rust";
    Object.assign(title.style, {
      fontSize: "18px", fontWeight: "bold",
      color: "#60e8ff",
      marginBottom: "4px",
      letterSpacing: "0.05em",
    });
    el.appendChild(title);

    // Tagline
    const tagline = document.createElement("div");
    tagline.textContent = "High-resolution fractal explorer · Rust / WASM / WebGL 2";
    Object.assign(tagline.style, {
      fontSize: "11px", color: "rgba(255,255,255,0.45)",
      marginBottom: "16px",
    });
    el.appendChild(tagline);

    // GitHub link
    const linkRow = document.createElement("div");
    linkRow.style.marginBottom = "20px";
    const link = document.createElement("a");
    link.href = REPO_URL;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = REPO_URL.replace("https://", "");
    Object.assign(link.style, {
      color: "#4ab4ff", textDecoration: "none", fontSize: "12px",
    });
    link.addEventListener("mouseover", () => link.style.textDecoration = "underline");
    link.addEventListener("mouseout",  () => link.style.textDecoration = "none");
    linkRow.appendChild(link);
    el.appendChild(linkRow);

    // Divider
    const hr = document.createElement("div");
    Object.assign(hr.style, {
      borderTop: "1px solid rgba(255,255,255,0.12)",
      marginBottom: "14px",
    });
    el.appendChild(hr);

    // Hints header
    const hintsHeader = document.createElement("div");
    hintsHeader.textContent = "Controls";
    Object.assign(hintsHeader.style, {
      fontSize: "11px", fontWeight: "bold",
      color: "rgba(255,255,255,0.45)",
      textTransform: "uppercase", letterSpacing: "0.08em",
      marginBottom: "10px",
    });
    el.appendChild(hintsHeader);

    // Tip
    const tip = document.createElement("div");
    tip.textContent = "Tip: copy the URL to bookmark or share any view.";
    Object.assign(tip.style, {
      marginTop: "16px", fontSize: "11px",
      color: "rgba(255,255,255,0.35)",
      fontStyle: "italic",
    });
    el.appendChild(tip);

    return el;
  }

  private makeHintsContainer(): HTMLDivElement {
    const wrap = document.createElement("div");

    const isJulia = /[#&?]f=julia/.test(decodeURIComponent(window.location.hash));

    if (isTouch) {
      const hints = isJulia
        ? [
            ...TOUCH_HINTS.slice(0, 3),
            { keys: "Tap", desc: "Place Julia c parameter" },
            TOUCH_HINTS[3],
          ]
        : TOUCH_HINTS;

      wrap.appendChild(this.makeTable(hints));

      const details = document.createElement("details");
      Object.assign(details.style, {
        marginTop: "12px", fontSize: "12px",
        color: "rgba(255,255,255,0.45)",
      });
      const summary = document.createElement("summary");
      summary.textContent = "Keyboard shortcuts";
      Object.assign(summary.style, { cursor: "pointer", marginBottom: "8px" });
      details.appendChild(summary);
      details.appendChild(this.makeTable(KEYBOARD_ONLY_HINTS));
      wrap.appendChild(details);
    } else {
      wrap.appendChild(this.makeTable(KEYBOARD_HINTS));
    }

    return wrap;
  }

  private makeTable(hints: { keys: string; desc: string }[]): HTMLTableElement {
    const table = document.createElement("table");
    Object.assign(table.style, {
      width: "100%", borderCollapse: "collapse", fontSize: "12px",
    });
    for (const { keys, desc } of hints) {
      const row = table.insertRow();
      const keyCell = row.insertCell();
      keyCell.textContent = keys;
      Object.assign(keyCell.style, {
        color: "#60e8ff", paddingRight: "16px",
        paddingBottom: "5px", whiteSpace: "nowrap",
        verticalAlign: "top",
      });
      const descCell = row.insertCell();
      descCell.textContent = desc;
      Object.assign(descCell.style, {
        color: "rgba(255,255,255,0.75)", paddingBottom: "5px",
      });
    }
    return table;
  }
}
