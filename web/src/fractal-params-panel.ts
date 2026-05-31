import { NEWTON_PRESETS, type NewtonParams } from "./newton.ts";
import { PANEL_STYLE, INPUT_CSS, SELECT_CSS, SECTION_HEADER_CSS, NARROW_PX } from "./ui-constants.ts";

type FractalKind = "mandelbrot" | "newton" | "multibrot" | "julia";

export class FractalParamsPanel {
  private readonly el: HTMLElement;
  private readonly titleEl: HTMLElement;
  private readonly select: HTMLSelectElement;
  private readonly advancedSection: HTMLElement;
  private readonly coeffInputs: HTMLInputElement[] = [];
  private currentDegree = 3;

  // Exponent controls (multibrot + julia)
  private readonly exponentSection: HTMLElement;
  private readonly exponentSlider: HTMLInputElement;
  private readonly exponentNumber: HTMLInputElement;

  // Julia c controls (julia only)
  private readonly juliaCSection: HTMLElement;
  private readonly juliaCReInput: HTMLInputElement;
  private readonly juliaCImInput: HTMLInputElement;
  private readonly miniCanvas: HTMLCanvasElement;
  private readonly miniWrapper: HTMLElement;
  private readonly miniCollapseChevron: HTMLElement;
  private readonly miniCLabel: HTMLElement;
  private miniPixels: ImageData | null = null;
  private miniRendered = false;
  private currentKind: FractalKind = "mandelbrot";
  private miniRenderDebounceId: ReturnType<typeof setTimeout> | null = null;

  // Newton-specific sections
  private readonly newtonPresetSection: HTMLElement;
  private readonly newtonAdvancedSection: HTMLElement;

  constructor(
    private readonly onPolynomialChange: (coeffs: Float64Array, degree: number) => void,
    private readonly onExponentChange: (exponent: number) => void,
    private readonly onJuliaCChange: (re: number, im: number) => void,
    private readonly onMiniRenderRequest: (exponent: number) => Promise<ImageData>,
  ) {
    this.el = document.createElement("div");
    this.el.className = "fractal-params-panel";
    Object.assign(this.el.style, PANEL_STYLE, {
      position: "fixed",
      top: "8px",
      right: "8px",
      minWidth: "180px",
      display: "none",
    });

    // Title
    this.titleEl = document.createElement("div");
    this.titleEl.className = "fractal-params-title";
    Object.assign(this.titleEl.style, { fontWeight: "bold", marginBottom: "6px" });
    this.el.appendChild(this.titleEl);

    // Newton preset picker
    this.newtonPresetSection = document.createElement("div");
    this.newtonPresetSection.style.marginBottom = "6px";
    const presetLabel = document.createElement("label");
    presetLabel.textContent = "Preset: ";
    this.select = document.createElement("select");
    this.select.style.cssText = SELECT_CSS;
    for (const p of NEWTON_PRESETS) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.label;
      this.select.appendChild(opt);
    }
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = "Custom";
    customOpt.style.display = "none";
    this.select.appendChild(customOpt);
    this.select.addEventListener("change", () => {
      if (this.select.value === "custom") return;
      const preset = NEWTON_PRESETS.find(p => p.name === this.select.value)!;
      const coeffs = new Float64Array(preset.coeffs);
      this.rebuildCoeffInputs(preset.degree, coeffs);
      this.onPolynomialChange(coeffs, preset.degree);
    });
    this.newtonPresetSection.appendChild(presetLabel);
    this.newtonPresetSection.appendChild(this.select);
    this.el.appendChild(this.newtonPresetSection);

    // Newton advanced (coefficients)
    this.newtonAdvancedSection = document.createElement("div");
    this.newtonAdvancedSection.style.cssText =
      "margin-top:8px;border-top:1px solid rgba(255,255,255,0.2);padding-top:6px;";
    const advancedHeader = document.createElement("div");
    advancedHeader.style.cssText = SECTION_HEADER_CSS;
    const advancedLabel = document.createElement("span");
    advancedLabel.textContent = "Advanced";
    const chevron = document.createElement("span");
    chevron.textContent = "▾";
    advancedHeader.appendChild(advancedLabel);
    advancedHeader.appendChild(chevron);
    const advanced = document.createElement("div");
    advanced.className = "fractal-params-advanced";
    advanced.style.display = "none";
    advancedHeader.addEventListener("click", () => {
      const open = advanced.style.display !== "none";
      advanced.style.display = open ? "none" : "block";
      chevron.textContent = open ? "▾" : "▴";
    });
    this.advancedSection = advanced;
    this.newtonAdvancedSection.appendChild(advancedHeader);
    this.newtonAdvancedSection.appendChild(advanced);
    this.el.appendChild(this.newtonAdvancedSection);

    // Exponent section (multibrot + julia)
    this.exponentSection = document.createElement("div");
    this.exponentSection.style.cssText = "margin-bottom:6px;";
    const expRow = document.createElement("div");
    expRow.style.cssText = "display:flex;align-items:center;gap:6px;flex-wrap:wrap;";
    const expLabel = document.createElement("label");
    expLabel.textContent = "Exponent:";
    expLabel.style.cssText = "color:#ccc;";
    this.exponentSlider = document.createElement("input");
    this.exponentSlider.type = "range";
    this.exponentSlider.min = "1.0";
    this.exponentSlider.max = "13.0";
    this.exponentSlider.step = "0.01";
    this.exponentSlider.value = "3.0";
    this.exponentSlider.style.cssText = "width:80px;vertical-align:middle;";
    this.exponentNumber = document.createElement("input");
    this.exponentNumber.type = "number";
    this.exponentNumber.min = "1.0";
    this.exponentNumber.max = "13.0";
    this.exponentNumber.step = "0.01";
    this.exponentNumber.value = "3.0";
    this.exponentNumber.style.cssText = INPUT_CSS + "width:52px;";

    this.exponentSlider.addEventListener("input", () => {
      let v = parseFloat(this.exponentSlider.value);
      v = snapToInteger(v);
      this.exponentNumber.value = String(v);
      this.onExponentChange(v);
      if (this.currentKind === "julia") this.requestMini();
    });
    this.exponentNumber.addEventListener("change", () => {
      let v = parseFloat(this.exponentNumber.value);
      if (isNaN(v)) return;
      v = Math.max(1.0, Math.min(13.0, v));
      this.exponentSlider.value = String(v);
      this.onExponentChange(v);
      if (this.currentKind === "julia") this.requestMini();
    });

    expRow.appendChild(expLabel);
    expRow.appendChild(this.exponentSlider);
    expRow.appendChild(this.exponentNumber);
    this.exponentSection.appendChild(expRow);
    this.el.appendChild(this.exponentSection);

    // Julia c section
    this.juliaCSection = document.createElement("div");
    this.juliaCSection.style.cssText = "margin-bottom:6px;";
    const cLabel = document.createElement("div");
    cLabel.textContent = "c (fixed point):";
    cLabel.style.cssText = "color:#ccc;margin-bottom:3px;";
    this.juliaCSection.appendChild(cLabel);

    const reRow = document.createElement("div");
    reRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:3px;";
    const reLabel = document.createElement("label");
    reLabel.textContent = "Re:";
    reLabel.style.cssText = "color:#aaa;width:24px;";
    this.juliaCReInput = document.createElement("input");
    this.juliaCReInput.type = "number";
    this.juliaCReInput.step = "0.001";
    this.juliaCReInput.value = "-0.7";
    this.juliaCReInput.style.cssText = INPUT_CSS + "width:80px;";
    reRow.appendChild(reLabel);
    reRow.appendChild(this.juliaCReInput);
    this.juliaCSection.appendChild(reRow);

    const imRow = document.createElement("div");
    imRow.style.cssText = "display:flex;align-items:center;gap:6px;";
    const imLabel = document.createElement("label");
    imLabel.textContent = "Im:";
    imLabel.style.cssText = "color:#aaa;width:24px;";
    this.juliaCImInput = document.createElement("input");
    this.juliaCImInput.type = "number";
    this.juliaCImInput.step = "0.001";
    this.juliaCImInput.value = "0.27";
    this.juliaCImInput.style.cssText = INPUT_CSS + "width:80px;";
    imRow.appendChild(imLabel);
    imRow.appendChild(this.juliaCImInput);
    this.juliaCSection.appendChild(imRow);

    const commitC = () => {
      const re = parseFloat(this.juliaCReInput.value);
      const im = parseFloat(this.juliaCImInput.value);
      if (!isNaN(re) && !isNaN(im)) this.onJuliaCChange(re, im);
    };
    this.juliaCReInput.addEventListener("blur", commitC);
    this.juliaCImInput.addEventListener("blur", commitC);
    this.juliaCReInput.addEventListener("keydown", (e) => { if (e.key === "Enter") commitC(); });
    this.juliaCImInput.addEventListener("keydown", (e) => { if (e.key === "Enter") commitC(); });

    // Collapse header: chevron + "c = …" label
    const miniHeader = document.createElement("div");
    miniHeader.style.cssText =
      "display:flex;align-items:center;gap:6px;margin-top:8px;cursor:pointer;user-select:none;";
    this.miniCollapseChevron = document.createElement("span");
    this.miniCollapseChevron.textContent = "▼";
    this.miniCLabel = document.createElement("span");
    this.miniCLabel.style.cssText = "color:#aaa;font-size:0.85em;";
    miniHeader.appendChild(this.miniCollapseChevron);
    miniHeader.appendChild(this.miniCLabel);
    miniHeader.addEventListener("click", () => {
      const collapsed = this.miniWrapper.style.display !== "none";
      this.setMiniCollapsed(collapsed);
      sessionStorage.setItem("julia-preview-collapsed", String(collapsed));
    });
    this.juliaCSection.appendChild(miniHeader);

    // Mini Mandelbrot picker — 210×150 matches the 3.5:2.5 (7:5) view aspect ratio
    this.miniWrapper = document.createElement("div");
    this.miniWrapper.style.cssText = "margin-top:4px;";
    this.miniCanvas = document.createElement("canvas");
    this.miniCanvas.width = 210;
    this.miniCanvas.height = 150;
    this.miniCanvas.style.cssText = "display:block;cursor:crosshair;border:1px solid rgba(255,255,255,0.15);touch-action:none;";
    this.miniWrapper.appendChild(this.miniCanvas);
    this.juliaCSection.appendChild(this.miniWrapper);
    this.updateMiniCLabel();

    let miniDragTimer: ReturnType<typeof setTimeout> | null = null;
    const pickC = (e: PointerEvent) => {
      const rect = this.miniCanvas.getBoundingClientRect();
      const px = Math.max(0, Math.min(209, Math.round(e.clientX - rect.left)));
      const py = Math.max(0, Math.min(149, Math.round(e.clientY - rect.top)));
      const step = 3.5 / 210;
      const re = -2.5 + px * step;
      const im = -1.25 + py * step;
      this.juliaCReInput.value = re.toFixed(4);
      this.juliaCImInput.value = im.toFixed(4);
      this.updateMiniCLabel();
      this.drawMiniCanvas();
      if (miniDragTimer !== null) clearTimeout(miniDragTimer);
      miniDragTimer = setTimeout(() => { miniDragTimer = null; this.onJuliaCChange(re, im); }, 50);
    };
    this.miniCanvas.addEventListener("pointerdown", (e) => {
      this.miniCanvas.setPointerCapture(e.pointerId);
      pickC(e);
    });
    this.miniCanvas.addEventListener("pointermove", (e) => { if (e.buttons) pickC(e); });
    this.miniCanvas.addEventListener("pointerup", pickC);

    this.el.appendChild(this.juliaCSection);
  }

  getElement(): HTMLElement { return this.el; }

  setKind(kind: FractalKind): void {
    const isNewton = kind === "newton";
    const isExponent = kind === "multibrot" || kind === "julia";
    const isJulia = kind === "julia";
    const visible = kind !== "mandelbrot";

    this.el.style.display = visible ? "" : "none";
    this.newtonPresetSection.style.display = isNewton ? "" : "none";
    this.newtonAdvancedSection.style.display = isNewton ? "" : "none";
    this.exponentSection.style.display = isExponent ? "" : "none";
    this.juliaCSection.style.display = isJulia ? "" : "none";

    this.currentKind = kind;

    if (isJulia) {
      if (!this.miniRendered) {
        this.miniRendered = true;
        this.requestMini(true);
      }
      const stored = sessionStorage.getItem("julia-preview-collapsed");
      const collapsed = stored !== null ? stored === "true" : window.innerWidth <= NARROW_PX;
      this.setMiniCollapsed(collapsed);
    }

    const titles: Record<string, string> = {
      newton: "Newton", multibrot: "Multibrot", julia: "Julia",
    };
    this.titleEl.textContent = titles[kind] ?? kind;
  }

  /** Update inputs to reflect the given params (e.g. after compute_roots completes). */
  setParams(params: NewtonParams): void {
    this.currentDegree = params.degree;
    this.rebuildCoeffInputs(params.degree, params.coeffs);

    const matchName = NEWTON_PRESETS.find(p =>
      p.degree === params.degree &&
      p.coeffs.every((c, i) => c === params.coeffs[i]),
    )?.name ?? "custom";
    this.select.value = matchName;
  }

  setExponent(e: number): void {
    this.exponentSlider.value = String(e);
    this.exponentNumber.value = String(e);
  }

  setJuliaC(re: number, im: number): void {
    this.juliaCReInput.value = String(re);
    this.juliaCImInput.value = String(im);
    this.updateMiniCLabel();
    if (this.miniPixels) this.drawMiniCanvas();
  }

  private setMiniCollapsed(collapsed: boolean): void {
    this.miniWrapper.style.display = collapsed ? "none" : "";
    this.miniCollapseChevron.textContent = collapsed ? "▶" : "▼";
  }

  private updateMiniCLabel(): void {
    const re = parseFloat(this.juliaCReInput.value);
    const im = parseFloat(this.juliaCImInput.value);
    if (!isNaN(re) && !isNaN(im)) {
      const sign = im >= 0 ? "+" : "";
      this.miniCLabel.textContent = `c = ${re.toPrecision(4)}${sign}${im.toPrecision(4)}i`;
    }
  }

  private requestMini(immediate = false): void {
    if (this.miniRenderDebounceId !== null) clearTimeout(this.miniRenderDebounceId);
    const dispatch = () => {
      this.miniRenderDebounceId = null;
      const exponent = parseFloat(this.exponentNumber.value) || 2.0;
      this.onMiniRenderRequest(exponent).then(img => {
        this.miniPixels = img;
        this.drawMiniCanvas();
      });
    };
    if (immediate) { dispatch(); } else { this.miniRenderDebounceId = setTimeout(dispatch, 150); }
  }

  private drawMiniCanvas(): void {
    const ctx = this.miniCanvas.getContext("2d")!;
    if (this.miniPixels) ctx.putImageData(this.miniPixels, 0, 0);
    this.drawCrosshair(ctx);
  }

  private drawCrosshair(ctx: CanvasRenderingContext2D): void {
    const re = parseFloat(this.juliaCReInput.value);
    const im = parseFloat(this.juliaCImInput.value);
    const step = 3.5 / 210;
    const cx = Math.max(0, Math.min(209, Math.round((re + 2.5) / step)));
    const cy = Math.max(0, Math.min(149, Math.round((im + 1.25) / step)));
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7); ctx.stroke();
    ctx.fillStyle = "#ffe040";
    ctx.fillRect(cx - 1, cy - 1, 3, 3);
  }

  private rebuildCoeffInputs(degree: number, coeffs: Float64Array): void {
    const advanced = this.advancedSection;
    advanced.innerHTML = "";
    this.coeffInputs.length = 0;
    this.currentDegree = degree;

    for (let k = 0; k <= degree; k++) {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;gap:6px;margin:3px 0;";

      const lbl = document.createElement("span");
      lbl.style.cssText = "width:36px;text-align:right;color:#aaa;";
      lbl.textContent = k === 0 ? "z⁰:" : k === 1 ? "z¹:" : `z${superscript(k)}:`;

      const input = document.createElement("input");
      input.type = "text";
      input.value = String(coeffs[k] ?? 0);
      input.style.cssText = INPUT_CSS + "width:70px;";

      const commit = () => this.commitCoeffs();
      input.addEventListener("blur", commit);
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") commit(); });
      input.addEventListener("input", () => {
        this.select.value = "custom";
      });

      row.appendChild(lbl);
      row.appendChild(input);
      advanced.appendChild(row);
      this.coeffInputs.push(input);
    }
  }

  private commitCoeffs(): void {
    const coeffs = new Float64Array(11);
    for (let k = 0; k < this.coeffInputs.length; k++) {
      const v = parseFloat(this.coeffInputs[k].value);
      coeffs[k] = isNaN(v) ? 0 : v;
    }
    const degree = this.currentDegree;
    this.onPolynomialChange(coeffs, degree);
  }
}

function superscript(n: number): string {
  return String(n).replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]);
}

function snapToInteger(v: number): number {
  const nearest = Math.round(v);
  return Math.abs(v - nearest) < 0.05 ? nearest : v;
}
