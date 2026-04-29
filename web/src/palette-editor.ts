/**
 * PaletteEditor — collapsible palette editing panel.
 *
 * Owns the current Palette state. Any user interaction rebuilds the LUT via
 * buildLut() and calls the onChange callback, which the session uses to
 * uploadLut() + setCycleLen() + blit() with no tile re-render.
 *
 * Usage:
 *   const editor = new PaletteEditor(CLASSIC, (lut, cycleLen) => { ... });
 *   overlay.appendChild(editor.getToggleButton());
 *   document.body.appendChild(editor.getPanel());
 */

import { buildLut, ALL_PRESETS, type ColorStop, type Palette } from "./palette.ts";

// ── Utilities ─────────────────────────────────────────────────────────────────

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) / 255, ((v >> 8) & 0xff) / 255, (v & 0xff) / 255];
}

function rgbToHex(r: number, g: number, b: number): string {
  return "#" + [r, g, b]
    .map(v => Math.round(v * 255).toString(16).padStart(2, "0"))
    .join("");
}

// Log-scale: slider 0–100 ↔ cycleLen 8–512. Midpoint 50 → 64 (default).
function sliderToCycleLen(v: number): number {
  return Math.round(8 * Math.pow(64, v / 100));
}
function cycleLenToSlider(c: number): number {
  return Math.round(Math.log(c / 8) / Math.log(64) * 100);
}

function deepCopyPalette(p: Palette): Palette {
  return { ...p, stops: p.stops.map(s => ({ ...s })) };
}

function drawLutToCanvas(canvas: HTMLCanvasElement, lut: Float32Array): void {
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width, h = canvas.height;
  const img = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    const li = Math.min(Math.floor(x / w * 4096), 4095) * 4;
    const r = Math.round(lut[li] * 255);
    const g = Math.round(lut[li + 1] * 255);
    const b = Math.round(lut[li + 2] * 255);
    for (let y = 0; y < h; y++) {
      const p = (y * w + x) * 4;
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ── PaletteEditor ─────────────────────────────────────────────────────────────

export class PaletteEditor {
  private palette: Palette;
  private selectedStop = 0;
  private readonly onChange: (lut: Float32Array, cycleLen: number) => void;

  private readonly panel: HTMLElement;
  private readonly toggleBtn: HTMLButtonElement;

  // Inner DOM refs populated by buildPanel()
  private readonly barCanvas: HTMLCanvasElement;
  private readonly handleRow: HTMLElement;
  private readonly stopListEl: HTMLElement;
  private readonly cycleLenLabel: HTMLSpanElement;
  private readonly cycleLenSlider: HTMLInputElement;

  constructor(initial: Palette, onChange: (lut: Float32Array, cycleLen: number) => void) {
    this.palette = deepCopyPalette(initial);
    this.onChange = onChange;

    const refs = this.buildPanel();
    this.panel       = refs.panel;
    this.barCanvas   = refs.barCanvas;
    this.handleRow   = refs.handleRow;
    this.stopListEl  = refs.stopListEl;
    this.cycleLenLabel  = refs.cycleLenLabel;
    this.cycleLenSlider = refs.cycleLenSlider;

    this.toggleBtn = this.buildToggleBtn();
    this.notifyChange(); // upload initial LUT
    this.render();
  }

  getToggleButton(): HTMLButtonElement { return this.toggleBtn; }
  getPanel(): HTMLElement { return this.panel; }

  // ── Internal ────────────────────────────────────────────────────────────────

  private notifyChange(): void {
    const lut = buildLut(this.palette);
    this.onChange(lut, this.palette.cycleLen);
  }

  private render(): void {
    this.renderBar();
    this.renderHandles();
    this.renderStopList();
  }

  // ── DOM construction ────────────────────────────────────────────────────────

  private buildToggleBtn(): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.textContent = "Palette ▾";
    Object.assign(btn.style, {
      display: "block", marginTop: "6px", cursor: "pointer",
      background: "rgba(255,255,255,0.15)", color: "white",
      border: "1px solid rgba(255,255,255,0.3)", borderRadius: "3px",
      fontFamily: "monospace", fontSize: "12px", padding: "2px 6px",
      width: "100%", boxSizing: "border-box",
    });
    btn.addEventListener("click", () => {
      const hidden = this.panel.style.display === "none";
      this.panel.style.display = hidden ? "block" : "none";
      btn.textContent = hidden ? "Palette ▴" : "Palette ▾";
    });
    return btn;
  }

  private buildPanel() {
    const panel = document.createElement("div");
    Object.assign(panel.style, {
      position: "fixed", left: "8px", top: "130px",
      background: "rgba(0,0,0,0.75)", color: "white",
      fontFamily: "monospace", fontSize: "12px",
      padding: "8px", borderRadius: "4px",
      width: "316px", zIndex: "998", display: "none",
      lineHeight: "1.5",
    });

    // ── Gradient bar ──────────────────────────────────────────────────────────
    const barCanvas = document.createElement("canvas");
    barCanvas.width = 300; barCanvas.height = 24;
    barCanvas.style.cssText = "display:block; cursor:crosshair; border-radius:2px;";
    barCanvas.addEventListener("click", (e) => this.onBarClick(e));

    const handleRow = document.createElement("div");
    handleRow.style.cssText = "position:relative; height:14px; width:300px; margin-bottom:6px;";

    // ── Stop list ─────────────────────────────────────────────────────────────
    const stopListEl = document.createElement("div");

    // ── Cycle-len row ─────────────────────────────────────────────────────────
    const cycleLenRow = document.createElement("div");
    cycleLenRow.style.cssText = "margin-top:6px; display:flex; align-items:center; gap:6px;";

    const cycleLenSlider = document.createElement("input");
    cycleLenSlider.type = "range"; cycleLenSlider.min = "0"; cycleLenSlider.max = "100";
    cycleLenSlider.value = String(cycleLenToSlider(this.palette.cycleLen));
    cycleLenSlider.style.cssText = "width:140px; vertical-align:middle;";

    const cycleLenLabel = document.createElement("span");
    cycleLenLabel.textContent = String(this.palette.cycleLen);

    cycleLenSlider.addEventListener("input", () => {
      this.palette.cycleLen = sliderToCycleLen(Number(cycleLenSlider.value));
      cycleLenLabel.textContent = String(this.palette.cycleLen);
      this.notifyChange();
    });

    cycleLenRow.append("Band width: ", cycleLenSlider, cycleLenLabel);

    // ── Preset strip ──────────────────────────────────────────────────────────
    const presetRow = document.createElement("div");
    presetRow.style.cssText = "margin-top:8px; display:flex; gap:4px; flex-wrap:wrap;";

    for (const { name, palette } of ALL_PRESETS) {
      const sw = document.createElement("canvas");
      sw.width = 44; sw.height = 20; sw.title = name;
      sw.style.cssText = "cursor:pointer; border-radius:2px; border:2px solid rgba(255,255,255,0.3);";
      sw.addEventListener("click", () => {
        this.palette = deepCopyPalette(palette);
        cycleLenSlider.value = String(cycleLenToSlider(this.palette.cycleLen));
        cycleLenLabel.textContent = String(this.palette.cycleLen);
        this.selectedStop = 0;
        this.notifyChange();
        this.render();
      });
      drawLutToCanvas(sw, buildLut(palette));
      presetRow.appendChild(sw);
    }

    panel.append(barCanvas, handleRow, stopListEl, cycleLenRow, presetRow);

    return { panel, barCanvas, handleRow, stopListEl, cycleLenLabel, cycleLenSlider };
  }

  // ── Render methods ──────────────────────────────────────────────────────────

  private renderBar(): void {
    drawLutToCanvas(this.barCanvas, buildLut(this.palette));
  }

  private renderHandles(): void {
    this.handleRow.innerHTML = "";
    this.palette.stops.forEach((stop, i) => {
      const el = document.createElement("div");
      const left = Math.round(stop.t * 300) - 6;
      el.style.cssText = `
        position:absolute; left:${left}px; top:1px;
        width:12px; height:12px; border-radius:50%;
        background:${rgbToHex(stop.r, stop.g, stop.b)};
        border:2px solid ${i === this.selectedStop ? "#fff" : "rgba(255,255,255,0.5)"};
        box-sizing:border-box; cursor:${i === 0 ? "default" : "grab"};
      `;
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        this.selectedStop = i;
        this.renderHandles();
        this.renderStopList();
      });
      if (i !== 0) this.attachDrag(el, i);
      this.handleRow.appendChild(el);
    });
  }

  private renderStopList(): void {
    this.stopListEl.innerHTML = "";
    this.palette.stops.forEach((stop, i) => {
      const row = document.createElement("div");
      row.style.cssText = `
        display:flex; align-items:center; gap:4px; margin:2px 0; padding:2px 2px;
        border-radius:3px;
        background:${i === this.selectedStop ? "rgba(255,255,255,0.12)" : "transparent"};
      `;

      // t position input
      const tInput = document.createElement("input");
      tInput.type = "number"; tInput.min = "0"; tInput.max = "0.99"; tInput.step = "0.01";
      tInput.value = stop.t.toFixed(2);
      tInput.style.cssText = "width:46px; font-size:11px; font-family:monospace; background:#222; color:white; border:1px solid #555; border-radius:2px;";
      tInput.disabled = i === 0; // first stop is always at t=0
      tInput.addEventListener("change", () => {
        const newT = Math.max(0.005, Math.min(0.995, parseFloat(tInput.value)));
        this.palette.stops[i] = { ...stop, t: newT };
        this.palette.stops.sort((a, b) => a.t - b.t);
        this.selectedStop = this.palette.stops.findIndex(s => s.t === newT);
        this.notifyChange(); this.render();
      });

      // color picker
      const colorInput = document.createElement("input");
      colorInput.type = "color";
      colorInput.value = rgbToHex(stop.r, stop.g, stop.b);
      colorInput.style.cssText = "width:28px; height:22px; padding:1px; cursor:pointer; border:1px solid #555; border-radius:2px; background:none;";
      colorInput.addEventListener("input", () => {
        const [r, g, b] = hexToRgb(colorInput.value);
        this.palette.stops[i] = { ...stop, r, g, b };
        this.notifyChange(); this.renderBar(); this.renderHandles();
      });

      // hueDir toggle
      const dirBtn = document.createElement("button");
      dirBtn.textContent = stop.hueDir === "short" ? "↺" : "↻";
      dirBtn.title = stop.hueDir === "short" ? "Short hue arc" : "Long hue arc";
      dirBtn.style.cssText = "cursor:pointer; font-size:13px; background:none; border:none; color:white; padding:0 2px; line-height:1;";
      dirBtn.addEventListener("click", () => {
        this.palette.stops[i] = { ...stop, hueDir: stop.hueDir === "short" ? "long" : "short" };
        this.notifyChange(); this.renderBar(); this.renderStopList();
      });

      // delete button
      const canDelete = this.palette.stops.length > 2 && i !== 0;
      const delBtn = document.createElement("button");
      delBtn.textContent = "✕";
      delBtn.disabled = !canDelete;
      delBtn.style.cssText = `cursor:${canDelete ? "pointer" : "default"}; background:none; border:none; color:white; padding:0 2px; font-size:11px; opacity:${canDelete ? "1" : "0.3"};`;
      delBtn.addEventListener("click", () => {
        if (!canDelete) return;
        this.palette.stops.splice(i, 1);
        this.selectedStop = Math.min(this.selectedStop, this.palette.stops.length - 1);
        this.notifyChange(); this.render();
      });

      row.addEventListener("click", () => {
        this.selectedStop = i;
        this.renderHandles();
        this.renderStopList();
      });
      row.append(tInput, colorInput, dirBtn, delBtn);
      this.stopListEl.appendChild(row);
    });
  }

  // ── Interactions ────────────────────────────────────────────────────────────

  private onBarClick(e: MouseEvent): void {
    const t = Math.max(0.005, Math.min(0.995, e.offsetX / this.barCanvas.width));
    // Don't add if click is within 8px of an existing handle
    const tPx = t * 300;
    if (this.palette.stops.some(s => Math.abs(s.t * 300 - tPx) < 8)) return;

    const lut = buildLut(this.palette);
    const li = Math.min(Math.floor(t * 4096), 4095) * 4;
    const newStop: ColorStop = { t, r: lut[li], g: lut[li + 1], b: lut[li + 2], hueDir: "short" };
    const pos = this.palette.stops.findIndex(s => s.t > t);
    const insertAt = pos === -1 ? this.palette.stops.length : pos;
    this.palette.stops.splice(insertAt, 0, newStop);
    this.selectedStop = insertAt;
    this.notifyChange(); this.render();
  }

  private attachDrag(el: HTMLElement, i: number): void {
    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const barRect = this.barCanvas.getBoundingClientRect();
      const startClientX = e.clientX;
      const startT = this.palette.stops[i].t;

      const onMove = (me: PointerEvent) => {
        const raw = startT + (me.clientX - startClientX) / barRect.width;
        const lo = i === 1 ? 0.005 : this.palette.stops[i - 1].t + 0.005;
        const hi = i === this.palette.stops.length - 1
          ? 0.995 : this.palette.stops[i + 1].t - 0.005;
        this.palette.stops[i] = { ...this.palette.stops[i], t: Math.max(lo, Math.min(hi, raw)) };
        this.notifyChange();
        this.renderBar();
        this.renderHandles();
        this.updateStopTInput(i);
      };

      el.addEventListener("pointermove", onMove);
      el.addEventListener("pointerup", () => {
        el.removeEventListener("pointermove", onMove);
      }, { once: true });
    });
  }

  private updateStopTInput(i: number): void {
    const row = this.stopListEl.children[i];
    if (!row) return;
    const tInput = row.querySelector("input[type=number]") as HTMLInputElement | null;
    if (tInput) tInput.value = this.palette.stops[i].t.toFixed(2);
  }
}
