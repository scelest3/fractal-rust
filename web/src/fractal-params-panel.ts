import { NEWTON_PRESETS, type NewtonParams } from "./newton.ts";
import { PANEL_STYLE, INPUT_CSS, SELECT_CSS, SECTION_HEADER_CSS } from "./ui-constants.ts";

export class FractalParamsPanel {
  private readonly el: HTMLElement;
  private readonly titleEl: HTMLElement;
  private readonly select: HTMLSelectElement;
  private readonly advancedSection: HTMLElement;
  private readonly coeffInputs: HTMLInputElement[] = [];
  private currentDegree = 3;

  constructor(
    private readonly onPolynomialChange: (coeffs: Float64Array, degree: number) => void,
  ) {
    this.el = this.buildPanel();
    this.titleEl = this.el.querySelector(".fractal-params-title")!;
    this.select = this.el.querySelector("select")!;
    this.advancedSection = this.el.querySelector(".fractal-params-advanced")!;
  }

  getElement(): HTMLElement { return this.el; }

  setVisible(visible: boolean): void {
    this.el.style.display = visible ? "" : "none";
  }

  setTitle(text: string): void {
    this.titleEl.textContent = text;
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

  private buildPanel(): HTMLElement {
    const panel = document.createElement("div");
    panel.className = "fractal-params-panel";
    Object.assign(panel.style, PANEL_STYLE, {
      position: "fixed",
      top: "8px",
      right: "8px",
      minWidth: "180px",
    });

    // Title — updated dynamically by setTitle()
    const title = document.createElement("div");
    title.className = "fractal-params-title";
    title.textContent = "Newton";
    Object.assign(title.style, { fontWeight: "bold", marginBottom: "6px" });
    panel.appendChild(title);

    // Preset picker
    const row = document.createElement("div");
    row.style.marginBottom = "6px";
    const label = document.createElement("label");
    label.textContent = "Preset: ";
    const select = document.createElement("select");
    select.style.cssText = SELECT_CSS;

    for (const p of NEWTON_PRESETS) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.label;
      select.appendChild(opt);
    }
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = "Custom";
    customOpt.style.display = "none";
    select.appendChild(customOpt);

    select.addEventListener("change", () => {
      if (select.value === "custom") return;
      const preset = NEWTON_PRESETS.find(p => p.name === select.value)!;
      const coeffs = new Float64Array(preset.coeffs);
      this.rebuildCoeffInputs(preset.degree, coeffs);
      this.onPolynomialChange(coeffs, preset.degree);
    });

    row.appendChild(label);
    row.appendChild(select);
    panel.appendChild(row);

    // Advanced collapsible section — unified pattern: border-top divider + header + chevron
    const advancedSection = document.createElement("div");
    advancedSection.style.cssText =
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

    advancedSection.appendChild(advancedHeader);
    advancedSection.appendChild(advanced);
    panel.appendChild(advancedSection);

    return panel;
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
