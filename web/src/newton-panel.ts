/**
 * Newton fractal UI panel.
 *
 * Renders a preset picker and a collapsible advanced section with per-coefficient
 * text inputs. Calls onPolynomialChange on blur or Enter — no live update while
 * typing.
 */

import { NEWTON_PRESETS, type NewtonParams } from "./newton.ts";

export class NewtonPanel {
  private readonly el: HTMLElement;
  private readonly select: HTMLSelectElement;
  private readonly advancedSection: HTMLElement;
  private readonly coeffInputs: HTMLInputElement[] = [];
  private currentDegree = 3;

  constructor(
    private readonly onPolynomialChange: (coeffs: Float64Array, degree: number) => void,
  ) {
    this.el = this.buildPanel();
    this.select = this.el.querySelector("select")!;
    this.advancedSection = this.el.querySelector(".newton-advanced")!;
  }

  getElement(): HTMLElement { return this.el; }

  /** Update inputs to reflect the given params (e.g. after compute_roots completes). */
  setParams(params: NewtonParams): void {
    this.currentDegree = params.degree;
    this.rebuildCoeffInputs(params.degree, params.coeffs);

    // Update preset picker: match by coefficients or mark as Custom.
    const matchName = NEWTON_PRESETS.find(p =>
      p.degree === params.degree &&
      p.coeffs.every((c, i) => c === params.coeffs[i]),
    )?.name ?? "custom";
    this.select.value = matchName;
  }

  private buildPanel(): HTMLElement {
    const panel = document.createElement("div");
    panel.className = "newton-panel";
    Object.assign(panel.style, {
      position: "fixed",
      top: "8px",
      right: "8px",
      background: "rgba(0,0,0,0.75)",
      color: "#eee",
      fontFamily: "monospace",
      fontSize: "12px",
      padding: "10px 12px",
      borderRadius: "6px",
      zIndex: "998",
      minWidth: "180px",
    });

    // Title
    const title = document.createElement("div");
    title.textContent = "Newton Fractal";
    Object.assign(title.style, { fontWeight: "bold", marginBottom: "6px" });
    panel.appendChild(title);

    // Preset picker
    const row = document.createElement("div");
    row.style.marginBottom = "6px";
    const label = document.createElement("label");
    label.textContent = "Preset: ";
    const select = document.createElement("select");
    select.style.cssText = "background:#222;color:#eee;border:1px solid #555;border-radius:3px;padding:2px 4px;";

    for (const p of NEWTON_PRESETS) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.label;
      select.appendChild(opt);
    }
    // Custom option (hidden by default, shown when user edits coefficients)
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

    // Advanced toggle
    const toggle = document.createElement("button");
    toggle.textContent = "▶ Advanced";
    toggle.style.cssText = "background:none;border:none;color:#aaa;cursor:pointer;font-size:11px;padding:0;margin-bottom:4px;";

    const advanced = document.createElement("div");
    advanced.className = "newton-advanced";
    advanced.style.display = "none";

    toggle.addEventListener("click", () => {
      const open = advanced.style.display !== "none";
      advanced.style.display = open ? "none" : "block";
      toggle.textContent = (open ? "▶" : "▼") + " Advanced";
    });

    panel.appendChild(toggle);
    panel.appendChild(advanced);

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
      input.style.cssText = "width:70px;background:#222;color:#eee;border:1px solid #555;border-radius:3px;padding:2px 4px;font-family:monospace;";

      const commit = () => this.commitCoeffs();
      input.addEventListener("blur", commit);
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") commit(); });
      input.addEventListener("input", () => {
        // Mark picker as Custom while typing.
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
