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

import { buildLut, makeNewtonPalette, ALL_PRESETS, type ColorStop, type Palette } from "./palette.ts";
import { PANEL_STYLE, SECTION_HEADER_CSS, SELECT_CSS } from "./ui-constants.ts";

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
  private readonly onTrapChange?: (trapColor: [number, number, number], trapRadius: number, trapStrength: number) => void;
  private readonly onInteriorChange?: (
    interiorColor: [number, number, number],
    shadingColor: [number, number, number],
    distanceStrength: number,
    distancePow: number,
    angleStrength: number,
    periodStrength: number,
  ) => void;
  private readonly onDistWeightChange?: (weight: number) => void;
  private readonly onNewtonChange?: (
    colorMode: 0 | 1 | 2,
    phase: number,
    smooth: boolean,
    unresolvedColor: [number, number, number],
  ) => void;

  private readonly panel: HTMLElement;
  private readonly toggleBtn: HTMLButtonElement;

  // Inner DOM refs populated by buildPanel()
  private readonly barCanvas: HTMLCanvasElement;
  private readonly handleRow: HTMLElement;
  private readonly stopListEl: HTMLElement;
  private cycleLenSlider!: HTMLInputElement;
  private cycleLenLabel!: HTMLSpanElement;

  // Section visibility groups for setFractalKind()
  private mandelbrotSections: HTMLElement[] = [];
  private newtonSections: HTMLElement[] = [];
  private currentNewtonDegree = 3;
  // Preset strip ref for Rainbow swatch update
  private presetRow!: HTMLElement;

  constructor(
    initial: Palette,
    onChange: (lut: Float32Array, cycleLen: number) => void,
    onTrapChange?: (trapColor: [number, number, number], trapRadius: number, trapStrength: number) => void,
    onInteriorChange?: (
      interiorColor: [number, number, number],
      shadingColor: [number, number, number],
      distanceStrength: number,
      distancePow: number,
      angleStrength: number,
      periodStrength: number,
    ) => void,
    onDistWeightChange?: (weight: number) => void,
    onNewtonChange?: (
      colorMode: 0 | 1 | 2,
      phase: number,
      smooth: boolean,
      unresolvedColor: [number, number, number],
    ) => void,
  ) {
    this.palette = deepCopyPalette(initial);
    this.onChange = onChange;
    this.onTrapChange = onTrapChange;
    this.onInteriorChange = onInteriorChange;
    this.onDistWeightChange = onDistWeightChange;
    this.onNewtonChange = onNewtonChange;

    const refs = this.buildPanel();
    this.panel      = refs.panel;
    this.barCanvas  = refs.barCanvas;
    this.handleRow  = refs.handleRow;
    this.stopListEl = refs.stopListEl;

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
    Object.assign(panel.style, PANEL_STYLE, {
      position: "fixed", left: "8px", top: "200px", // overridden by ResizeObserver in session.ts
      width: "316px", display: "none",
    });

    // ── Title ─────────────────────────────────────────────────────────────────
    const titleEl = document.createElement("div");
    titleEl.textContent = "Palette";
    Object.assign(titleEl.style, { fontWeight: "bold", marginBottom: "6px" });
    panel.appendChild(titleEl);

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
    this.cycleLenSlider = cycleLenSlider;

    const cycleLenLabel = document.createElement("span");
    cycleLenLabel.textContent = String(this.palette.cycleLen);
    this.cycleLenLabel = cycleLenLabel;

    cycleLenSlider.addEventListener("input", () => {
      this.palette.cycleLen = sliderToCycleLen(Number(cycleLenSlider.value));
      cycleLenLabel.textContent = String(this.palette.cycleLen);
      this.notifyChange();
    });

    cycleLenRow.append("Band width: ", cycleLenSlider, cycleLenLabel);

    // ── Preset strip ──────────────────────────────────────────────────────────
    const presetRow = document.createElement("div");
    presetRow.style.cssText = "margin-top:8px; display:flex; gap:4px; flex-wrap:wrap;";
    for (const { name, palette, isNewton } of ALL_PRESETS) {
      const sw = document.createElement("canvas");
      sw.width = 44; sw.height = 20; sw.title = name;
      sw.style.cssText = "cursor:pointer; border-radius:2px; border:2px solid rgba(255,255,255,0.3);";
      sw.addEventListener("click", () => {
        // Rainbow preset generates stops matched to current Newton degree
        const applied = isNewton ? makeNewtonPalette(this.currentNewtonDegree) : palette;
        this.palette = deepCopyPalette(applied);
        cycleLenSlider.value = String(cycleLenToSlider(this.palette.cycleLen));
        cycleLenLabel.textContent = String(this.palette.cycleLen);
        this.selectedStop = 0;
        this.notifyChange();
        this.render();
      });
      drawLutToCanvas(sw, buildLut(palette));
      presetRow.appendChild(sw);
    }
    this.presetRow = presetRow;

    // ── Orbit Trap ────────────────────────────────────────────────────────────
    const trapSection = document.createElement("div");
    trapSection.style.cssText = "margin-top:8px; border-top:1px solid rgba(255,255,255,0.2); padding-top:6px;";

    const trapHeader = document.createElement("div");
    trapHeader.style.cssText = SECTION_HEADER_CSS;
    const trapHeaderLabel = document.createElement("span");
    trapHeaderLabel.textContent = "Orbit Trap";
    const trapChevron = document.createElement("span");
    trapChevron.textContent = "▾";
    trapHeader.append(trapHeaderLabel, trapChevron);
    trapSection.appendChild(trapHeader);

    const trapBody = document.createElement("div");
    trapSection.appendChild(trapBody);

    const trapColorInput = document.createElement("input");
    trapColorInput.type = "color"; trapColorInput.value = "#ffff00";
    trapColorInput.style.cssText = "width:28px; height:22px; padding:1px; cursor:pointer; border:1px solid #555; border-radius:2px; background:none;";

    const trapRadiusSlider = document.createElement("input");
    trapRadiusSlider.type = "range"; trapRadiusSlider.min = "0"; trapRadiusSlider.max = "2"; trapRadiusSlider.step = "0.01";
    trapRadiusSlider.value = "0.5";
    trapRadiusSlider.style.cssText = "width:120px; vertical-align:middle;";
    const trapRadiusLabel = document.createElement("span");
    trapRadiusLabel.textContent = "0.50";

    const trapStrengthSlider = document.createElement("input");
    trapStrengthSlider.type = "range"; trapStrengthSlider.min = "0"; trapStrengthSlider.max = "1"; trapStrengthSlider.step = "0.01";
    trapStrengthSlider.value = "0";
    trapStrengthSlider.style.cssText = "width:120px; vertical-align:middle;";
    const trapStrengthLabel = document.createElement("span");
    trapStrengthLabel.textContent = "0.00";

    const fireTrap = () => {
      const [r, g, b] = hexToRgb(trapColorInput.value);
      this.onTrapChange?.([r, g, b], parseFloat(trapRadiusSlider.value), parseFloat(trapStrengthSlider.value));
    };

    trapColorInput.addEventListener("input", fireTrap);
    trapRadiusSlider.addEventListener("input", () => {
      trapRadiusLabel.textContent = parseFloat(trapRadiusSlider.value).toFixed(2);
      fireTrap();
    });
    trapStrengthSlider.addEventListener("input", () => {
      trapStrengthLabel.textContent = parseFloat(trapStrengthSlider.value).toFixed(2);
      fireTrap();
    });

    const trapColorRow = document.createElement("div");
    trapColorRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    trapColorRow.append("Color: ", trapColorInput);

    const trapRadiusRow = document.createElement("div");
    trapRadiusRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    trapRadiusRow.append("Radius: ", trapRadiusSlider, trapRadiusLabel);

    const trapStrengthRow = document.createElement("div");
    trapStrengthRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    trapStrengthRow.append("Strength: ", trapStrengthSlider, trapStrengthLabel);

    trapBody.append(trapColorRow, trapRadiusRow, trapStrengthRow);
    trapHeader.addEventListener("click", () => {
      const open = trapBody.style.display !== "none";
      trapBody.style.display = open ? "none" : "";
      trapChevron.textContent = open ? "▾" : "▴";
    });

    // ── Interior ──────────────────────────────────────────────────────────────
    const interiorSection = document.createElement("div");
    interiorSection.style.cssText = "margin-top:8px; border-top:1px solid rgba(255,255,255,0.2); padding-top:6px;";

    const interiorHeader = document.createElement("div");
    interiorHeader.style.cssText = SECTION_HEADER_CSS;
    const interiorHeaderLabel = document.createElement("span");
    interiorHeaderLabel.textContent = "Interior";
    const interiorChevron = document.createElement("span");
    interiorChevron.textContent = "▾";
    interiorHeader.append(interiorHeaderLabel, interiorChevron);
    interiorSection.appendChild(interiorHeader);

    const interiorBody = document.createElement("div");
    interiorSection.appendChild(interiorBody);

    // Base color
    const interiorColorInput = document.createElement("input");
    interiorColorInput.type = "color"; interiorColorInput.value = "#000000";
    interiorColorInput.style.cssText = "width:28px; height:22px; padding:1px; cursor:pointer; border:1px solid #555; border-radius:2px; background:none;";

    // Distance shading
    const distanceToggle = document.createElement("input");
    distanceToggle.type = "checkbox";
    distanceToggle.style.cssText = "cursor:pointer; vertical-align:middle;";

    const distanceStrengthSlider = document.createElement("input");
    distanceStrengthSlider.type = "range"; distanceStrengthSlider.min = "0"; distanceStrengthSlider.max = "1"; distanceStrengthSlider.step = "0.01";
    distanceStrengthSlider.value = "0.5";
    distanceStrengthSlider.style.cssText = "width:90px; vertical-align:middle; display:none;";
    const distanceStrengthLabel = document.createElement("span");
    distanceStrengthLabel.textContent = "0.50"; distanceStrengthLabel.style.display = "none";

    // Shade-toward color (visible when distance on)
    const shadingColorInput = document.createElement("input");
    shadingColorInput.type = "color"; shadingColorInput.value = "#ffffff";
    shadingColorInput.style.cssText = "width:28px; height:22px; padding:1px; cursor:pointer; border:1px solid #555; border-radius:2px; background:none; display:none;";

    // Power curve slider (visible when distance on)
    const distancePowSlider = document.createElement("input");
    distancePowSlider.type = "range"; distancePowSlider.min = "0.1"; distancePowSlider.max = "3"; distancePowSlider.step = "0.05";
    distancePowSlider.value = "1";
    distancePowSlider.style.cssText = "width:90px; vertical-align:middle; display:none;";
    const distancePowLabel = document.createElement("span");
    distancePowLabel.textContent = "1.00"; distancePowLabel.style.display = "none";

    // Angle coloring
    const angleToggle = document.createElement("input");
    angleToggle.type = "checkbox";
    angleToggle.style.cssText = "cursor:pointer; vertical-align:middle;";

    const angleStrengthSlider = document.createElement("input");
    angleStrengthSlider.type = "range"; angleStrengthSlider.min = "0"; angleStrengthSlider.max = "1"; angleStrengthSlider.step = "0.01";
    angleStrengthSlider.value = "1";
    angleStrengthSlider.style.cssText = "width:90px; vertical-align:middle; display:none;";
    const angleStrengthLabel = document.createElement("span");
    angleStrengthLabel.textContent = "1.00"; angleStrengthLabel.style.display = "none";

    // Period coloring
    const periodToggle = document.createElement("input");
    periodToggle.type = "checkbox";
    periodToggle.style.cssText = "cursor:pointer; vertical-align:middle;";

    const periodStrengthSlider = document.createElement("input");
    periodStrengthSlider.type = "range"; periodStrengthSlider.min = "0"; periodStrengthSlider.max = "1"; periodStrengthSlider.step = "0.01";
    periodStrengthSlider.value = "1";
    periodStrengthSlider.style.cssText = "width:90px; vertical-align:middle; display:none;";
    const periodStrengthLabel = document.createElement("span");
    periodStrengthLabel.textContent = "1.00"; periodStrengthLabel.style.display = "none";

    const fireInterior = () => {
      const [r, g, b]     = hexToRgb(interiorColorInput.value);
      const [sr, sg, sb]  = hexToRgb(shadingColorInput.value);
      const distStrength  = distanceToggle.checked ? parseFloat(distanceStrengthSlider.value) : 0.0;
      const distPow       = parseFloat(distancePowSlider.value);
      const angStrength   = angleToggle.checked ? parseFloat(angleStrengthSlider.value) : 0.0;
      const perStrength   = periodToggle.checked ? parseFloat(periodStrengthSlider.value) : 0.0;
      this.onInteriorChange?.([r, g, b], [sr, sg, sb], distStrength, distPow, angStrength, perStrength);
    };

    const setDistanceVisible = (on: boolean) => {
      distanceStrengthSlider.style.display = on ? "" : "none";
      distanceStrengthLabel.style.display  = on ? "" : "none";
      shadingColorInput.style.display      = on ? "" : "none";
      distancePowSlider.style.display      = on ? "" : "none";
      distancePowLabel.style.display       = on ? "" : "none";
    };

    interiorColorInput.addEventListener("input", fireInterior);
    shadingColorInput.addEventListener("input", fireInterior);

    distanceToggle.addEventListener("change", () => { setDistanceVisible(distanceToggle.checked); fireInterior(); });
    distanceStrengthSlider.addEventListener("input", () => {
      distanceStrengthLabel.textContent = parseFloat(distanceStrengthSlider.value).toFixed(2);
      fireInterior();
    });
    distancePowSlider.addEventListener("input", () => {
      distancePowLabel.textContent = parseFloat(distancePowSlider.value).toFixed(2);
      fireInterior();
    });

    angleToggle.addEventListener("change", () => {
      const on = angleToggle.checked;
      angleStrengthSlider.style.display = on ? "" : "none";
      angleStrengthLabel.style.display  = on ? "" : "none";
      fireInterior();
    });
    angleStrengthSlider.addEventListener("input", () => {
      angleStrengthLabel.textContent = parseFloat(angleStrengthSlider.value).toFixed(2);
      fireInterior();
    });

    periodToggle.addEventListener("change", () => {
      const on = periodToggle.checked;
      periodStrengthSlider.style.display = on ? "" : "none";
      periodStrengthLabel.style.display  = on ? "" : "none";
      fireInterior();
    });
    periodStrengthSlider.addEventListener("input", () => {
      periodStrengthLabel.textContent = parseFloat(periodStrengthSlider.value).toFixed(2);
      fireInterior();
    });

    const interiorColorRow = document.createElement("div");
    interiorColorRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    interiorColorRow.append("Color: ", interiorColorInput);

    const distanceRow = document.createElement("div");
    distanceRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0; flex-wrap:wrap;";
    distanceRow.append("Distance: ", distanceToggle, distanceStrengthSlider, distanceStrengthLabel, " → ", shadingColorInput);

    const curveRow = document.createElement("div");
    curveRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    curveRow.append("Curve: ", distancePowSlider, distancePowLabel);

    const angleRow = document.createElement("div");
    angleRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    angleRow.append("Angle: ", angleToggle, angleStrengthSlider, angleStrengthLabel);

    const periodRow = document.createElement("div");
    periodRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    periodRow.append("Period: ", periodToggle, periodStrengthSlider, periodStrengthLabel);

    interiorBody.append(interiorColorRow, distanceRow, curveRow, angleRow, periodRow);
    interiorHeader.addEventListener("click", () => {
      const open = interiorBody.style.display !== "none";
      interiorBody.style.display = open ? "none" : "";
      interiorChevron.textContent = open ? "▾" : "▴";
    });

    // ── Exterior coloring ─────────────────────────────────────────────────────
    const exteriorSection = document.createElement("div");
    exteriorSection.style.cssText = "margin-top:8px; border-top:1px solid rgba(255,255,255,0.2); padding-top:6px;";

    const extHeader = document.createElement("div");
    extHeader.style.cssText = SECTION_HEADER_CSS;
    const extHeaderLabel = document.createElement("span");
    extHeaderLabel.textContent = "Exterior";
    const extChevron = document.createElement("span");
    extChevron.textContent = "▾";
    extHeader.append(extHeaderLabel, extChevron);
    exteriorSection.appendChild(extHeader);

    const exteriorBody = document.createElement("div");
    exteriorSection.appendChild(exteriorBody);

    const distWeightSlider = document.createElement("input");
    distWeightSlider.type = "range"; distWeightSlider.min = "-2"; distWeightSlider.max = "2"; distWeightSlider.step = "0.05";
    distWeightSlider.value = "0";
    distWeightSlider.style.cssText = "width:130px; vertical-align:middle;";
    const distWeightLabel = document.createElement("span");
    distWeightLabel.textContent = "0.00";

    distWeightSlider.addEventListener("input", () => {
      const v = parseFloat(distWeightSlider.value);
      distWeightLabel.textContent = v.toFixed(2);
      this.onDistWeightChange?.(v);
    });

    const distWeightRow = document.createElement("div");
    distWeightRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    distWeightRow.append("Dist: ", distWeightSlider, distWeightLabel);
    exteriorBody.appendChild(distWeightRow);
    extHeader.addEventListener("click", () => {
      const open = exteriorBody.style.display !== "none";
      exteriorBody.style.display = open ? "none" : "";
      extChevron.textContent = open ? "▾" : "▴";
    });

    // ── Newton: Basins ────────────────────────────────────────────────────────
    const basinsSection = document.createElement("div");
    basinsSection.style.cssText = "margin-top:8px; border-top:1px solid rgba(255,255,255,0.2); padding-top:6px; display:none;";

    const basinsHeader = document.createElement("div");
    basinsHeader.style.cssText = SECTION_HEADER_CSS;
    const basinsChevron = document.createElement("span");
    basinsChevron.textContent = "▾";
    basinsHeader.append(Object.assign(document.createElement("span"), { textContent: "Basins" }), basinsChevron);

    const basinsBody = document.createElement("div");
    basinsHeader.addEventListener("click", () => {
      const open = basinsBody.style.display !== "none";
      basinsBody.style.display = open ? "none" : "";
      basinsChevron.textContent = open ? "▾" : "▴";
    });

    const modeSelect = document.createElement("select");
    modeSelect.style.cssText = SELECT_CSS;
    ([ ["Basin + convergence", "0"], ["Convergence only", "1"], ["Basin only", "2"] ] as const)
      .forEach(([label, value]) => {
        const opt = document.createElement("option");
        opt.value = value; opt.textContent = label;
        modeSelect.appendChild(opt);
      });

    const phaseSlider = document.createElement("input");
    phaseSlider.type = "range"; phaseSlider.min = "0"; phaseSlider.max = "1"; phaseSlider.step = "0.01";
    phaseSlider.value = "0";
    phaseSlider.style.cssText = "width:130px; vertical-align:middle;";
    const phaseLabel = document.createElement("span");
    phaseLabel.textContent = "0.00";

    const modeRow = document.createElement("div");
    modeRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    modeRow.append("Mode: ", modeSelect);
    const phaseRow = document.createElement("div");
    phaseRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    phaseRow.append("Phase: ", phaseSlider, phaseLabel);
    basinsBody.append(modeRow, phaseRow);
    basinsSection.append(basinsHeader, basinsBody);

    // ── Newton: Convergence ───────────────────────────────────────────────────
    const convSection = document.createElement("div");
    convSection.style.cssText = "margin-top:8px; border-top:1px solid rgba(255,255,255,0.2); padding-top:6px; display:none;";

    const convHeader = document.createElement("div");
    convHeader.style.cssText = SECTION_HEADER_CSS;
    const convChevron = document.createElement("span");
    convChevron.textContent = "▾";
    convHeader.append(Object.assign(document.createElement("span"), { textContent: "Convergence" }), convChevron);

    const convBody = document.createElement("div");
    convHeader.addEventListener("click", () => {
      const open = convBody.style.display !== "none";
      convBody.style.display = open ? "none" : "";
      convChevron.textContent = open ? "▾" : "▴";
    });

    const smoothToggle = document.createElement("input");
    smoothToggle.type = "checkbox";
    smoothToggle.checked = true;
    smoothToggle.style.cssText = "cursor:pointer; vertical-align:middle;";
    const smoothRow = document.createElement("div");
    smoothRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    smoothRow.append("Smooth: ", smoothToggle);
    convBody.append(smoothRow);
    convSection.append(convHeader, convBody);

    // ── Newton: Unresolved ────────────────────────────────────────────────────
    const unresolvedSection = document.createElement("div");
    unresolvedSection.style.cssText = "margin-top:8px; border-top:1px solid rgba(255,255,255,0.2); padding-top:6px; display:none;";

    const unresolvedHeader = document.createElement("div");
    unresolvedHeader.style.cssText = SECTION_HEADER_CSS;
    const unresolvedChevron = document.createElement("span");
    unresolvedChevron.textContent = "▾";
    unresolvedHeader.append(Object.assign(document.createElement("span"), { textContent: "Unresolved" }), unresolvedChevron);

    const unresolvedBody = document.createElement("div");
    unresolvedHeader.addEventListener("click", () => {
      const open = unresolvedBody.style.display !== "none";
      unresolvedBody.style.display = open ? "none" : "";
      unresolvedChevron.textContent = open ? "▾" : "▴";
    });

    const unresolvedColorInput = document.createElement("input");
    unresolvedColorInput.type = "color";
    unresolvedColorInput.value = "#0d0d0d"; // matches default [0.05, 0.05, 0.05]
    unresolvedColorInput.style.cssText = "width:28px; height:22px; padding:1px; cursor:pointer; border:1px solid #555; border-radius:2px; background:none;";
    const unresolvedRow = document.createElement("div");
    unresolvedRow.style.cssText = "display:flex; align-items:center; gap:6px; margin:3px 0;";
    unresolvedRow.append("Color: ", unresolvedColorInput);
    unresolvedBody.append(unresolvedRow);
    unresolvedSection.append(unresolvedHeader, unresolvedBody);

    // ── Newton: shared fireNewton callback ────────────────────────────────────
    const fireNewton = () => {
      const mode = parseInt(modeSelect.value) as 0 | 1 | 2;
      const phase = parseFloat(phaseSlider.value);
      const smooth = smoothToggle.checked;
      const [r, g, b] = hexToRgb(unresolvedColorInput.value);
      this.onNewtonChange?.(mode, phase, smooth, [r, g, b]);
    };
    modeSelect.addEventListener("change", fireNewton);
    phaseSlider.addEventListener("input", () => {
      phaseLabel.textContent = parseFloat(phaseSlider.value).toFixed(2);
      fireNewton();
    });
    smoothToggle.addEventListener("change", fireNewton);
    unresolvedColorInput.addEventListener("input", fireNewton);

    this.mandelbrotSections = [exteriorSection, trapSection, interiorSection];
    this.newtonSections = [basinsSection, convSection, unresolvedSection];

    panel.append(barCanvas, handleRow, stopListEl, cycleLenRow, presetRow,
      exteriorSection, trapSection, interiorSection,
      basinsSection, convSection, unresolvedSection);

    return { panel, barCanvas, handleRow, stopListEl };
  }

  setFractalKind(kind: "mandelbrot" | "newton"): void {
    const isMandelbrot = kind === "mandelbrot";
    for (const s of this.mandelbrotSections) s.style.display = isMandelbrot ? "" : "none";
    for (const s of this.newtonSections)     s.style.display = isMandelbrot ? "none" : "";
  }

  setNewtonDegree(degree: number): void {
    this.currentNewtonDegree = degree;
    this.palette = deepCopyPalette(makeNewtonPalette(degree));
    this.selectedStop = 0;
    this.notifyChange();
    this.render();
  }

  getPalette(): Palette {
    return deepCopyPalette(this.palette);
  }

  loadPalette(p: Palette): void {
    this.palette = deepCopyPalette(p);
    this.cycleLenSlider.value = String(cycleLenToSlider(this.palette.cycleLen));
    this.cycleLenLabel.textContent = String(this.palette.cycleLen);
    this.selectedStop = 0;
    this.notifyChange();
    this.render();
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
