import { pixelStep } from "./viewport.ts";
import type { GlPipeline } from "./gl-pipeline.ts";

/**
 * Height of each render strip in pixels.
 * Capped by the GPU's maximum renderbuffer size.
 */
export function computeStripHeight(
  exportHeight: number,
  maxRenderbufferSize: number,
): number {
  return Math.min(maxRenderbufferSize, exportHeight);
}

/**
 * Number of strips needed to cover the full export height.
 */
export function computeStripCount(
  exportHeight: number,
  stripHeight: number,
): number {
  return Math.ceil(exportHeight / stripHeight);
}

/**
 * Fractal units per pixel for an export render.
 * Fixes the vertical fractal extent to match the viewport; wider or taller
 * exports reveal more content on the sides or top/bottom respectively.
 */
export function computeExportStep(
  zoomExp: number,
  exportHeight: number,
): number {
  return pixelStep(zoomExp, exportHeight);
}

// ── Export pipeline types ─────────────────────────────────────────────────────

export interface WorkerLease {
  workers: Worker[];
  tileSab: SharedArrayBuffer;
  release(): void;
}

export interface ExportParams {
  width: number;
  height: number;
  format: "png" | "jpeg";
  /** JPEG quality 50–100 (ignored for PNG). */
  quality: number;
  view: { cx: string; cy: string; zoom_exp: number };
  fractalKind: "mandelbrot" | "newton";
  maxIter: number;
  useF64x2: boolean;
  cxRef: number;
  cyRef: number;
  newton?: { degree: number; coeffs: number[]; rootsRe: number[]; rootsIm: number[] };
}

// ── ExportDialog ──────────────────────────────────────────────────────────────

const PRESETS = [
  { label: "1080p", width: 1920,  height: 1080  },
  { label: "4K",    width: 3840,  height: 2160  },
  { label: "8K",    width: 7680,  height: 4320  },
  { label: "16K",   width: 15360, height: 8640  },
] as const;

type BuildParams = (
  width: number,
  height: number,
  format: "png" | "jpeg",
  quality: number,
) => ExportParams;

/**
 * Modal export dialog. Owns its own DOM; call open() to show it.
 * Does not import FractalSession — decoupled via acquireLease and buildParams.
 */
export class ExportDialog {
  private readonly dialog: HTMLDialogElement;
  private readonly widthInput: HTMLInputElement;
  private readonly heightInput: HTMLInputElement;
  private readonly qualitySlider: HTMLInputElement;
  private readonly qualityLabel: HTMLSpanElement;
  private readonly qualityRow: HTMLDivElement;
  private readonly sizeHint: HTMLSpanElement;
  private readonly progressBar: HTMLProgressElement;
  private readonly progressRow: HTMLDivElement;
  private readonly exportBtn: HTMLButtonElement;
  private readonly cancelBtn: HTMLButtonElement;
  private readonly warningEl: HTMLSpanElement;

  private formatRadios: HTMLInputElement[] = [];
  private presetRadios: HTMLInputElement[] = [];
  private activeSession: ExportSession | null = null;
  private maxRbs = 16384;

  constructor(
    private readonly gl: GlPipeline,
    private readonly acquireLease: () => WorkerLease,
    private readonly buildParams: BuildParams,
  ) {
    this.dialog = document.createElement("dialog");
    Object.assign(this.dialog.style, {
      background: "#1a1a1a",
      color: "#eee",
      border: "1px solid #444",
      borderRadius: "8px",
      padding: "20px 24px",
      minWidth: "340px",
      fontFamily: "monospace",
      fontSize: "13px",
    });

    // Title
    const title = document.createElement("h3");
    title.textContent = "Export";
    Object.assign(title.style, { margin: "0 0 14px", fontSize: "15px", fontWeight: "600" });
    this.dialog.appendChild(title);

    // ── Resolution presets ────────────────────────────────────────────────────
    const presetLabel = document.createElement("div");
    presetLabel.textContent = "Resolution";
    presetLabel.style.cssText = "margin-bottom:6px;color:#aaa;";
    this.dialog.appendChild(presetLabel);

    const presetRow = document.createElement("div");
    presetRow.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;";

    this.presetRadios = [];

    for (const p of PRESETS) {
      const { radio, wrapper } = this.makeRadioWidget("preset", p.label, p.label);
      this.presetRadios.push(radio);
      radio.addEventListener("change", () => {
        if (radio.checked) this.applyPreset(p.width, p.height);
      });
      presetRow.appendChild(wrapper);
    }

    const { radio: customR, wrapper: customW } = this.makeRadioWidget("preset", "Custom", "custom");
    customR.addEventListener("change", () => {
      if (customR.checked) { this.widthInput.focus(); }
    });
    presetRow.appendChild(customW);
    this.dialog.appendChild(presetRow);

    // ── Custom width / height inputs ──────────────────────────────────────────
    const dimRow = document.createElement("div");
    dimRow.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:4px;";

    this.widthInput  = this.makeDimInput("Width");
    this.heightInput = this.makeDimInput("Height");

    const xLabel = document.createElement("span");
    xLabel.textContent = "×";
    xLabel.style.color = "#888";

    dimRow.appendChild(this.widthInput);
    dimRow.appendChild(xLabel);
    dimRow.appendChild(this.heightInput);
    this.dialog.appendChild(dimRow);

    this.warningEl = document.createElement("span");
    this.warningEl.style.cssText = "color:#f90;font-size:11px;display:none;margin-bottom:6px;display:block;";
    this.dialog.appendChild(this.warningEl);

    for (const inp of [this.widthInput, this.heightInput]) {
      inp.addEventListener("input", () => {
        customR.checked = true;
        this.clampAndWarn();
        this.updateSizeHint();
      });
    }

    // ── Format ────────────────────────────────────────────────────────────────
    const fmtLabel = document.createElement("div");
    fmtLabel.textContent = "Format";
    fmtLabel.style.cssText = "margin:10px 0 6px;color:#aaa;";
    this.dialog.appendChild(fmtLabel);

    const fmtRow = document.createElement("div");
    fmtRow.style.cssText = "display:flex;gap:12px;margin-bottom:8px;";

    this.formatRadios = [];
    for (const fmt of ["PNG", "JPEG"] as const) {
      const { radio, wrapper } = this.makeRadioWidget("format", fmt, fmt.toLowerCase());
      this.formatRadios.push(radio);
      radio.addEventListener("change", () => {
        this.qualityRow.style.display = radio.value === "jpeg" && radio.checked ? "flex" : "none";
        if (radio.value === "png" && radio.checked) this.qualityRow.style.display = "none";
        this.updateSizeHint();
      });
      fmtRow.appendChild(wrapper);
    }
    this.formatRadios[0].checked = true; // PNG default
    this.dialog.appendChild(fmtRow);

    // ── JPEG quality ──────────────────────────────────────────────────────────
    this.qualityRow = document.createElement("div");
    this.qualityRow.style.cssText = "display:none;align-items:center;gap:8px;margin-bottom:8px;";

    const qLabel = document.createElement("label");
    qLabel.textContent = "Quality";
    qLabel.style.color = "#aaa";

    this.qualitySlider = document.createElement("input");
    this.qualitySlider.type = "range";
    this.qualitySlider.min = "50";
    this.qualitySlider.max = "100";
    this.qualitySlider.value = "90";
    this.qualitySlider.setAttribute("aria-label", "JPEG quality");
    this.qualitySlider.style.flex = "1";
    this.qualitySlider.addEventListener("input", () => {
      this.qualityLabel.textContent = `${this.qualitySlider.value}%`;
      this.updateSizeHint();
    });

    this.qualityLabel = document.createElement("span");
    this.qualityLabel.textContent = "90%";
    this.qualityLabel.style.cssText = "width:36px;text-align:right;";

    this.qualityRow.appendChild(qLabel);
    this.qualityRow.appendChild(this.qualitySlider);
    this.qualityRow.appendChild(this.qualityLabel);
    this.dialog.appendChild(this.qualityRow);

    // ── Size hint ─────────────────────────────────────────────────────────────
    this.sizeHint = document.createElement("span");
    this.sizeHint.style.cssText = "color:#888;font-size:11px;display:block;margin-bottom:12px;";
    this.dialog.appendChild(this.sizeHint);

    // ── Progress bar ──────────────────────────────────────────────────────────
    this.progressRow = document.createElement("div");
    this.progressRow.style.cssText = "display:none;margin-bottom:12px;";

    this.progressBar = document.createElement("progress");
    this.progressBar.setAttribute("aria-label", "Export progress");
    this.progressBar.max = 100;
    this.progressBar.value = 0;
    this.progressBar.style.cssText = "width:100%;height:8px;";
    this.progressRow.appendChild(this.progressBar);
    this.dialog.appendChild(this.progressRow);

    // ── Buttons ───────────────────────────────────────────────────────────────
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:8px;justify-content:flex-end;";

    this.cancelBtn = document.createElement("button");
    this.cancelBtn.textContent = "Cancel";
    this.cancelBtn.setAttribute("aria-label", "Cancel export");
    Object.assign(this.cancelBtn.style, this.btnStyle("#333", "#eee"));
    this.cancelBtn.addEventListener("click", () => this.handleCancel());

    this.exportBtn = document.createElement("button");
    this.exportBtn.textContent = "Export";
    this.exportBtn.setAttribute("aria-label", "Start export");
    Object.assign(this.exportBtn.style, this.btnStyle("#0a84ff", "#fff"));
    this.exportBtn.addEventListener("click", () => void this.handleExport());

    btnRow.appendChild(this.cancelBtn);
    btnRow.appendChild(this.exportBtn);
    this.dialog.appendChild(btnRow);

    // Close on Escape only when not rendering
    this.dialog.addEventListener("cancel", (e) => {
      if (this.activeSession) {
        e.preventDefault();
        this.activeSession.cancel();
      }
    });

    document.body.appendChild(this.dialog);

    // Apply 1080p default once we know maxRbs (set in open())
    this.applyPreset(1920, 1080);
    this.presetRadios[0].checked = true;
  }

  open(): void {
    this.maxRbs = this.gl.getMaxRenderbufferSize();
    // Disable presets that exceed the hardware limit
    for (let i = 0; i < PRESETS.length; i++) {
      const p = PRESETS[i];
      const exceeds = p.width > this.maxRbs || p.height > this.maxRbs;
      this.presetRadios[i].disabled = exceeds;
      (this.presetRadios[i].parentElement as HTMLElement).style.opacity = exceeds ? "0.4" : "1";
    }
    this.setRendering(false);
    this.dialog.showModal();
  }

  private applyPreset(width: number, height: number): void {
    this.widthInput.value  = String(width);
    this.heightInput.value = String(height);
    this.warningEl.style.display = "none";
    this.updateSizeHint();
  }

  private clampAndWarn(): void {
    let warned = false;
    for (const inp of [this.widthInput, this.heightInput]) {
      const v = parseInt(inp.value, 10);
      if (!isNaN(v) && v > this.maxRbs) {
        inp.value = String(this.maxRbs);
        warned = true;
      }
    }
    this.warningEl.textContent = warned
      ? `Clamped to GPU max (${this.maxRbs} px)`
      : "";
    this.warningEl.style.display = warned ? "block" : "none";
  }

  private updateSizeHint(): void {
    const w = parseInt(this.widthInput.value, 10) || 0;
    const h = parseInt(this.heightInput.value, 10) || 0;
    const fmt = this.selectedFormat();
    const bytes = fmt === "jpeg" ? w * h * 3 : w * h * 4;
    const mb = (bytes / 1024 / 1024).toFixed(1);
    this.sizeHint.textContent = `Estimated size: ~${mb} MB`;
  }

  private selectedFormat(): "png" | "jpeg" {
    return (this.formatRadios.find(r => r.checked)?.value ?? "png") as "png" | "jpeg";
  }

  private async handleExport(): Promise<void> {
    const w = Math.max(1, Math.min(this.maxRbs, parseInt(this.widthInput.value, 10) || 1920));
    const h = Math.max(1, Math.min(this.maxRbs, parseInt(this.heightInput.value, 10) || 1080));
    const format = this.selectedFormat();
    const quality = parseInt(this.qualitySlider.value, 10);

    this.setRendering(true);
    this.progressBar.value = 0;

    const lease = this.acquireLease();
    const params = this.buildParams(w, h, format, quality);
    const session = new ExportSession(lease, this.gl, params, (done, total) => {
      this.progressBar.value = Math.round((done / total) * 100);
    });
    this.activeSession = session;

    const blob = await session.run();
    this.activeSession = null;
    this.setRendering(false);

    if (blob) {
      const ext = format === "jpeg" ? "jpg" : "png";
      const cx = parseFloat(params.view.cx).toFixed(6);
      const zoom = params.view.zoom_exp.toFixed(2);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `fractal-${cx}-zoom${zoom}.${ext}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      this.dialog.close();
    }
  }

  private handleCancel(): void {
    if (this.activeSession) {
      this.activeSession.cancel();
    } else {
      this.dialog.close();
    }
  }

  private setRendering(on: boolean): void {
    this.exportBtn.disabled = on;
    this.exportBtn.style.opacity = on ? "0.5" : "1";
    this.progressRow.style.display = on ? "block" : "none";
    this.cancelBtn.textContent = on ? "Cancel" : "Close";
  }

  private makeRadioWidget(
    name: string,
    label: string,
    value: string,
  ): { radio: HTMLInputElement; wrapper: HTMLLabelElement } {
    const wrapper = document.createElement("label");
    Object.assign(wrapper.style, {
      display: "flex", alignItems: "center", gap: "4px",
      cursor: "pointer", padding: "3px 8px",
      background: "#2a2a2a", borderRadius: "4px",
      border: "1px solid #444",
    });
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = name;
    radio.value = value;
    radio.setAttribute("aria-label", label);
    const text = document.createElement("span");
    text.textContent = label;
    wrapper.appendChild(radio);
    wrapper.appendChild(text);
    return { radio, wrapper };
  }

  private makeDimInput(ariaLabel: string): HTMLInputElement {
    const inp = document.createElement("input");
    inp.type = "number";
    inp.min = "1";
    inp.style.cssText = "width:80px;background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:4px;padding:4px 6px;font-family:monospace;font-size:13px;";
    inp.setAttribute("aria-label", ariaLabel);
    return inp;
  }

  private btnStyle(bg: string, color: string): Partial<CSSStyleDeclaration> {
    return {
      background: bg, color, border: "none", borderRadius: "5px",
      padding: "7px 16px", cursor: "pointer", fontFamily: "monospace",
      fontSize: "13px", fontWeight: "600",
    };
  }
}

// ── ExportSession ─────────────────────────────────────────────────────────────

const TILE = 256;
const TILE_SLOT_BYTES = TILE * TILE * 4 * 4; // 1 MiB per slot

function flipRows(pixels: Uint8Array, width: number, height: number): Uint8ClampedArray {
  const rowBytes = width * 4;
  const out = new Uint8ClampedArray(pixels.length);
  for (let y = 0; y < height; y++) {
    out.set(
      pixels.subarray((height - 1 - y) * rowBytes, (height - y) * rowBytes),
      y * rowBytes,
    );
  }
  return out;
}

type TileReadyMsg = { type: "tile_ready"; slotIndex: number; tileX: number; tileY: number; generation: number };

/**
 * Renders an export image using the leased worker pool and GL pipeline.
 * WorkerLease.release() is always called in a finally block.
 */
export class ExportSession {
  private cancelled = false;

  constructor(
    private readonly lease: WorkerLease,
    private readonly gl: GlPipeline,
    private readonly params: ExportParams,
    private readonly onProgress: (done: number, total: number) => void,
  ) {}

  cancel(): void {
    this.cancelled = true;
  }

  async run(): Promise<Blob | null> {
    try {
      return await this.doRun();
    } finally {
      this.lease.release();
    }
  }

  private async doRun(): Promise<Blob | null> {
    const { params, gl } = this;
    const { width, height } = params;

    const maxRbs = gl.getMaxRenderbufferSize();
    const stripHeight = computeStripHeight(height, maxRbs);
    const stripCount = computeStripCount(height, stripHeight);
    const step = computeExportStep(params.view.zoom_exp, height);

    const offscreen = new OffscreenCanvas(width, height);
    const ctx = offscreen.getContext("2d")!;

    gl.initExportFBOs(width, stripHeight);
    gl.setFractalKind(params.fractalKind === "newton" ? 1 : 0);
    if (params.fractalKind === "newton" && params.newton) {
      gl.setNewtonDegree(params.newton.degree);
    }

    const tilesX = Math.ceil(width / TILE);
    let totalTiles = 0;
    for (let s = 0; s < stripCount; s++) {
      const stripY = s * stripHeight;
      const actualH = Math.min(stripHeight, height - stripY);
      totalTiles += tilesX * Math.ceil(actualH / TILE);
    }
    let doneTiles = 0;

    for (let s = 0; s < stripCount; s++) {
      if (this.cancelled) return null;

      const stripY = s * stripHeight;
      const actualH = Math.min(stripHeight, height - stripY);
      const tilesY = Math.ceil(actualH / TILE);

      gl.clearExportFBO();
      await this.renderStrip(stripY, tilesX, tilesY, step, doneTiles, totalTiles);
      doneTiles += tilesX * tilesY;

      if (this.cancelled) return null;
      gl.blitExportStrip();

      if (this.cancelled) return null;
      const pixels = await gl.readbackStrip();

      if (this.cancelled) return null;
      const flipped = flipRows(pixels, width, actualH);
      ctx.putImageData(new ImageData(flipped, width, actualH), 0, stripY);
    }

    return offscreen.convertToBlob(
      params.format === "jpeg"
        ? { type: "image/jpeg", quality: params.quality / 100 }
        : { type: "image/png" },
    );
  }

  private renderStrip(
    stripY: number,
    tilesX: number,
    tilesY: number,
    step: number,
    doneBeforeStrip: number,
    totalTiles: number,
  ): Promise<void> {
    return new Promise((resolve) => {
      const { params, lease, gl } = this;
      const { width, height, useF64x2, cxRef, cyRef } = params;

      // Build pending tile list for this strip
      const pending: { tx: number; ty: number; deltaRe: number; deltaIm: number }[] = [];
      for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
          let deltaRe: number;
          let deltaIm: number;
          if (useF64x2) {
            deltaRe = (tx * TILE - width / 2) * step;
            deltaIm = (stripY + ty * TILE - height / 2) * step;
          } else {
            deltaRe = cxRef + (tx * TILE - width / 2) * step;
            deltaIm = cyRef + (stripY + ty * TILE - height / 2) * step;
          }
          pending.push({ tx, ty, deltaRe, deltaIm });
        }
      }

      const total = pending.length;
      let completedInStrip = 0;
      const busyWorkers = new Set<number>();

      const dispatch = () => {
        for (let i = 0; i < lease.workers.length; i++) {
          if (pending.length === 0) break;
          if (busyWorkers.has(i)) continue;
          const tile = pending.shift()!;
          busyWorkers.add(i);
          lease.workers[i].postMessage({
            type: "render_tile",
            deltaRe: tile.deltaRe,
            deltaIm: tile.deltaIm,
            step,
            maxIter: params.maxIter,
            slotIndex: i,
            tileX: tile.tx,
            tileY: tile.ty,
            generation: 1,
            useF64x2,
            cxRef,
            cyRef,
            fractalKind: params.fractalKind,
            newton: params.newton,
          });
        }
      };

      // Install export handler on each worker
      lease.workers.forEach((w, i) => {
        w.onmessage = (e: MessageEvent) => {
          const msg = e.data as TileReadyMsg;
          if (msg.type !== "tile_ready") return;
          busyWorkers.delete(i);
          gl.uploadExportTile(lease.tileSab, msg.slotIndex, msg.tileX, msg.tileY);
          completedInStrip++;
          this.onProgress(doneBeforeStrip + completedInStrip, totalTiles);
          if (completedInStrip === total) {
            resolve();
          } else {
            dispatch();
          }
        };
      });

      dispatch();
    });
  }
}
