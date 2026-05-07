/**
 * FractalSession — main-thread orchestrator.
 *
 * Responsibilities:
 *  - Own the ZoomPanFSM and wire it to the canvas
 *  - Tile the canvas into 256×256 chunks and dispatch render jobs to workers
 *  - For zoom_exp > 15: trigger reference orbit computation (Orbit Worker),
 *    wait for orbit_ready, then dispatch perturbation tiles
 *  - For zoom_exp ≤ 15: dispatch escape_time tiles directly (no orbit needed)
 *  - Receive completed tiles and upload them to the GL pipeline
 *
 * No WASM runs on the main thread. All computation is in Web Workers.
 */
import { ZoomPanFSM, DEFAULT_VIEW, pixelToFractal, pixelStep } from "./viewport.ts";
import { BoxZoom } from "./box-zoom.ts";
import { GlPipeline } from "./gl-pipeline.ts";
import { wasmBundleUrl } from "./detect-simd.ts";
import { CLASSIC } from "./palette.ts";
import { PaletteEditor } from "./palette-editor.ts";
import {
  newtonParamsZ3, defaultNewtonParams, newtonParamsFromPreset,
  serializeNewtonState, deserializeNewtonState,
  type NewtonParams,
} from "./newton.ts";
import { NewtonPanel } from "./newton-panel.ts";
import { ExportDialog, type WorkerLease, type ColoringState } from "./export.ts";

const BASE_ITER = 256;

const ZOOM_RANGES: Record<"mandelbrot" | "newton", [number, number]> = {
  mandelbrot: [-7, 17],
  newton:     [-7, 14],
};
const ITER_PER_DECADE = 64;
const TILE_SIZE = 256;
const TILE_SLOT_BYTES = TILE_SIZE * TILE_SIZE * 4 * 4; // 1 MiB per slot

// ── Orbit SAB layout (mirrors wasm-bridge memory-layout constants) ──────────
//
// orbitSab holds: [8-byte header] [orbit entries] ... [BLA entries] ...
// The header is currently unused by TypeScript (orbit_len is passed via message).
// PRIMARY_ORBIT_ENTRY_BYTES = 32 (max stride, Phoenix), Mandelbrot uses 16.
// BLA_ENTRY_BYTES = 48.

const ORBIT_HEADER_BYTES = 8;
const PRIMARY_ORBIT_ENTRY_BYTES = 32;
const BLA_ENTRY_BYTES = 48;
const PRIMARY_ORBIT_DATA_OFFSET = ORBIT_HEADER_BYTES; // = 8

/** Covers zoom_exp ≤ 50 with ITER_PER_DECADE=64: max = 256 + 50*64 = 3456. */
const MAX_ORBIT_ITER = 4096;

function blaTableOffset(maxOrbitIter: number): number {
  return ORBIT_HEADER_BYTES + maxOrbitIter * PRIMARY_ORBIT_ENTRY_BYTES;
}
function orbitSabSize(maxOrbitIter: number): number {
  return blaTableOffset(maxOrbitIter) + maxOrbitIter * BLA_ENTRY_BYTES;
}

/**
 * Zoom depth above which F64x2 (double-double) escape-time is used.
 * f64 absolute coordinates lose per-pixel precision around zoom_exp ≈ 14.
 * F64x2 covers zoom_exp up to ≈ 20 cleanly (v1 cap).
 * Perturbation theory (zoom_exp > 20) is v2 — the orbit worker is kept
 * but not triggered in v1.
 */
const F64X2_ZOOM_THRESHOLD = 14;

// Workers: N_TILE_WORKERS tile workers + 1 orbit worker.
const N_WORKERS = Math.max(2, Math.min(8, (navigator.hardwareConcurrency ?? 4) - 2));
const RING_SLOTS = N_WORKERS; // worker k → slot k

interface TileDesc {
  deltaRe: number;
  deltaIm: number;
  step: number;
  maxIter: number;
  tileX: number;
  tileY: number;
  slotIndex: number;
  generation: number;
  useF64x2: boolean;
  cxRef: number;
  cyRef: number;
  fractalKind: "mandelbrot" | "newton";
  newton?: { degree: number; coeffs: number[]; rootsRe: number[]; rootsIm: number[] };
}

type WorkerInMsg =
  | { type: "init_done" }
  | { type: "tile_ready"; slotIndex: number; tileX: number; tileY: number; generation: number };

type OrbitWorkerInMsg =
  | { type: "orbit_worker_ready" }
  | { type: "orbit_ready"; ref_orbit_id: number; orbit_len: number; bla_len: number }
  | { type: "roots_ready"; degree: number; rootsRe: number[]; rootsIm: number[] };

function makeOverlay(): HTMLElement {
  const el = document.createElement("div");
  el.id = "stats-overlay";
  Object.assign(el.style, {
    position: "fixed", top: "8px", left: "8px",
    color: "white", fontFamily: "monospace", fontSize: "12px",
    background: "rgba(0,0,0,0.55)", padding: "6px 10px",
    borderRadius: "4px", pointerEvents: "none", zIndex: "999",
    lineHeight: "1.6",
  });
  document.body.appendChild(el);
  return el;
}

export class FractalSession {
  private readonly fsm: ZoomPanFSM;
  readonly gl: GlPipeline;
  private readonly workers: Worker[];
  private readonly tileSab: SharedArrayBuffer;
  private readonly orbitSab: SharedArrayBuffer;
  private readonly orbitWorker: Worker;
  private readonly overlay: HTMLElement;
  private readonly statsEl: HTMLElement;
  private readonly paletteEditor: PaletteEditor;
  private hiDpi: boolean;

  private readonly busyWorkers = new Set<number>();
  private workerInitCount = 0;
  private paused = false;
  private orbitWorkerReady = false;

  private pendingTiles: TileDesc[] = [];
  private generation = 0;
  private lutReady = false;
  private tilesThisGen = 0;
  private tilesDrawnThisGen = 0;
  private dispatchDebounceId: ReturnType<typeof setTimeout> | null = null;

  // Orbit state
  private currentOrbitId = 0;
  private orbitReady = false;
  private pendingOrbitDispatch = false; // waiting for orbit_ready before dispatching

  private readonly exportDialog: ExportDialog;

  private coloringState: ColoringState = {
    lut:              new Float32Array(4096 * 4),
    cycleLen:         64,
    trapRadius:       0.5,
    trapStrength:     0.0,
    trapColor:        [1.0, 1.0, 0.0],
    interiorColor:    [0.0, 0.0, 0.0],
    shadingColor:     [1.0, 1.0, 1.0],
    distanceStrength: 0.0,
    distancePow:      1.0,
    angleStrength:    0.0,
    periodStrength:   0.0,
    periodCycleLen:   8.0,
    distWeight:       0.0,
    fractalKind:      0,
    newtonDegree:     3,
    unresolvedColor:  [0.05, 0.05, 0.05],
  };

  // Fractal kind + Newton state
  private fractalKind: "mandelbrot" | "newton" = "mandelbrot";
  private newtonParams: NewtonParams = defaultNewtonParams();
  private readonly newtonPanel: NewtonPanel;
  private modeSelectEl!: HTMLSelectElement;

  constructor(canvas: HTMLCanvasElement) {
    this.overlay = makeOverlay();

    // Fractal mode selector — sits above the stats readout.
    const modeRow = document.createElement("div");
    modeRow.style.cssText = "pointer-events:auto;margin-bottom:4px;";
    const modeSelect = document.createElement("select");
    modeSelect.style.cssText = "background:#222;color:#eee;border:1px solid #555;border-radius:3px;padding:2px 6px;font-family:monospace;font-size:12px;cursor:pointer;";
    (["mandelbrot", "newton"] as const).forEach(kind => {
      const opt = document.createElement("option");
      opt.value = kind;
      opt.textContent = kind === "mandelbrot" ? "Mandelbrot" : "Newton";
      modeSelect.appendChild(opt);
    });
    modeSelect.value = this.fractalKind;
    modeSelect.addEventListener("change", () =>
      this.switchFractalKind(modeSelect.value as "mandelbrot" | "newton"),
    );
    modeRow.appendChild(modeSelect);
    this.overlay.appendChild(modeRow);
    this.modeSelectEl = modeSelect;

    this.statsEl = document.createElement("div");
    this.overlay.appendChild(this.statsEl);
    this.tileSab = new SharedArrayBuffer(RING_SLOTS * TILE_SLOT_BYTES);
    this.orbitSab = new SharedArrayBuffer(orbitSabSize(MAX_ORBIT_ITER));
    this.gl = new GlPipeline(canvas);
    this.paletteEditor = new PaletteEditor(
      CLASSIC,
      (lut, cycleLen) => {
        this.gl.uploadLut(lut);
        this.gl.setCycleLen(cycleLen);
        this.gl.blit();
        this.coloringState.lut = lut;
        this.coloringState.cycleLen = cycleLen;
      },
      ([r, g, b], trapRadius, trapStrength) => {
        this.gl.setTrapColor(r, g, b);
        this.gl.setTrapRadius(trapRadius);
        this.gl.setTrapStrength(trapStrength);
        this.gl.blit();
        this.coloringState.trapColor = [r, g, b];
        this.coloringState.trapRadius = trapRadius;
        this.coloringState.trapStrength = trapStrength;
      },
      ([r, g, b], [sr, sg, sb], distanceStrength, distancePow, angleStrength, periodStrength) => {
        this.gl.setInteriorColor(r, g, b);
        this.gl.setShadingColor(sr, sg, sb);
        this.gl.setDistanceStrength(distanceStrength);
        this.gl.setDistancePow(distancePow);
        this.gl.setAngleStrength(angleStrength);
        this.gl.setPeriodStrength(periodStrength);
        this.gl.blit();
        this.coloringState.interiorColor = [r, g, b];
        this.coloringState.shadingColor = [sr, sg, sb];
        this.coloringState.distanceStrength = distanceStrength;
        this.coloringState.distancePow = distancePow;
        this.coloringState.angleStrength = angleStrength;
        this.coloringState.periodStrength = periodStrength;
      },
      (weight) => {
        this.gl.setDistWeight(weight);
        this.gl.blit();
        this.coloringState.distWeight = weight;
      },
    );
    document.body.appendChild(this.paletteEditor.getPanel());

    // Newton panel — preset picker + advanced coefficient inputs.
    this.newtonPanel = new NewtonPanel((coeffs, degree) => {
      this.onPolynomialCommit(coeffs, degree);
    });
    document.body.appendChild(this.newtonPanel.getElement());

    // Restore fractal kind + Newton state from URL hash if present.
    const hashParams = deserializeNewtonState(window.location.hash.slice(1));
    if (hashParams) {
      this.fractalKind = "newton";
      this.newtonParams = hashParams;
      this.modeSelectEl.value = "newton";
    }
    this.newtonPanel.setParams(this.newtonParams);
    this.newtonPanel.setVisible(this.fractalKind === "newton");

    this.hiDpi = (window.devicePixelRatio ?? 1) > 1;
    this.overlay.appendChild(this.buildDprToggle(canvas));
    this.overlay.appendChild(this.paletteEditor.getToggleButton());

    this.exportDialog = new ExportDialog(
      this.gl,
      () => this.pause(),
      (width, height, dpi) => ({
        width, height,
        format: "png",
        dpi,
        coloringState: this.coloringState,
        view:        this.fsm.getView(),
        fractalKind: this.fractalKind,
        maxIter:     this.computeMaxIter(this.fsm.getView().zoom_exp),
        useF64x2:    this.fsm.getView().zoom_exp > F64X2_ZOOM_THRESHOLD,
        cxRef:       parseFloat(this.fsm.getView().cx),
        cyRef:       parseFloat(this.fsm.getView().cy),
        newton: this.fractalKind === "newton" ? {
          degree:  this.newtonParams.degree,
          coeffs:  Array.from(this.newtonParams.coeffs),
          rootsRe: Array.from(this.newtonParams.rootsRe),
          rootsIm: Array.from(this.newtonParams.rootsIm),
        } : undefined,
      }),
    );

    document.addEventListener("keydown", (e: KeyboardEvent) => {
      if ((e.key === "e" || e.key === "E") && !e.metaKey && !e.ctrlKey && !e.altKey) {
        this.exportDialog.open();
      }
    });

    this.fsm = new ZoomPanFSM(DEFAULT_VIEW, {
      onViewChange: () => {
        this.updateOverlay();
        if (this.fsm.getState() === "PANNING") this.debouncedDispatch(150);
      },
      onZoomSettled: () => this.debouncedDispatch(0),
    });
    this.fsm.setCanvasSize(canvas.width, canvas.height);
    this.fsm.setPixelScale(this.hiDpi ? (window.devicePixelRatio || 1) : 1);
    this.fsm.setZoomExpRange(...ZOOM_RANGES[this.fractalKind]);
    this.fsm.attach(canvas);

    new BoxZoom(canvas, () => this.fsm.getView(), (view) => {
      this.fsm.setView(view);
      this.scheduleDispatch();
    }, () => this.hiDpi ? (window.devicePixelRatio || 1) : 1);

    const wasmUrl = wasmBundleUrl();

    // Orbit Worker — one dedicated worker for BigFloat reference orbit computation.
    this.orbitWorker = new Worker(
      new URL("./math-worker.ts", import.meta.url),
      { type: "module" },
    );
    this.orbitWorker.onerror = (e) => console.error("[FractalSession] orbit worker error", e);
    this.orbitWorker.onmessage = (e: MessageEvent<OrbitWorkerInMsg>) =>
      this.onOrbitWorkerMessage(e.data);
    this.orbitWorker.postMessage({
      type: "orbit_init",
      wasmUrl,
      orbitSab: this.orbitSab,
      primaryOrbitDataOffset: PRIMARY_ORBIT_DATA_OFFSET,
      blaTableOffset: blaTableOffset(MAX_ORBIT_ITER),
    });

    // Tile Workers.
    this.workers = Array.from({ length: N_WORKERS }, (_, i) => {
      const w = new Worker(new URL("./tile-worker.ts", import.meta.url), { type: "module" });
      w.onerror = (e) => console.error(`[FractalSession] tile worker ${i} error`, e);
      w.onmessage = (e: MessageEvent<WorkerInMsg>) => this.onWorkerMessage(i, e.data);
      w.postMessage({
        type: "init",
        wasmUrl,
        tileSab: this.tileSab,
        orbitSab: this.orbitSab,
        primaryOrbitDataOffset: PRIMARY_ORBIT_DATA_OFFSET,
        blaTableOffset: blaTableOffset(MAX_ORBIT_ITER),
      });
      return w;
    });
  }

  private switchFractalKind(kind: "mandelbrot" | "newton"): void {
    if (kind === this.fractalKind) return;
    this.fractalKind = kind;
    this.coloringState.fractalKind = kind === "newton" ? 1 : 0;

    // Apply the precision-appropriate zoom range and reset zoom to 0.
    this.fsm.setZoomExpRange(...ZOOM_RANGES[kind]);
    const view = this.fsm.getView();
    this.fsm.setView({ ...view, zoom_exp: 0 });

    this.newtonPanel.setVisible(kind === "newton");
    this.modeSelectEl.value = kind;

    if (kind === "newton") {
      // Compute roots for the current Newton polynomial if needed.
      if (this.newtonParams.rootsRe.every(r => r === 0) && this.orbitWorkerReady) {
        this.orbitWorker.postMessage({
          type: "compute_roots",
          coeffs: Array.from(this.newtonParams.coeffs.subarray(0, this.newtonParams.degree + 1)),
        });
      }
      this.writeUrlHash();
    } else {
      window.history.replaceState(null, "", window.location.pathname);
    }

    this.scheduleDispatch();
  }

  /** Called when the user commits a polynomial change (blur/Enter in the panel). */
  private onPolynomialCommit(coeffs: Float64Array, degree: number): void {
    // Update params with zeroed roots; real roots arrive via roots_ready.
    this.newtonParams = {
      degree,
      coeffs: new Float64Array(coeffs),
      rootsRe: new Float64Array(degree),
      rootsIm: new Float64Array(degree),
    };
    this.coloringState.newtonDegree = degree;
    // Ask the orbit worker to compute roots (it has WASM; main thread doesn't).
    this.orbitWorker.postMessage({
      type: "compute_roots",
      coeffs: Array.from(coeffs.subarray(0, degree + 1)),
    });
  }

  private onOrbitWorkerMessage(msg: OrbitWorkerInMsg): void {
    if (msg.type === "roots_ready") {
      // Roots computed — update params, write URL, re-render.
      this.newtonParams = {
        ...this.newtonParams,
        rootsRe: new Float64Array(msg.rootsRe),
        rootsIm: new Float64Array(msg.rootsIm),
      };
      this.newtonPanel.setParams(this.newtonParams);
      this.writeUrlHash();
      this.scheduleDispatch();
      return;
    }

    if (msg.type === "orbit_worker_ready") {
      console.log("[Session] orbit worker ready");
      this.orbitWorkerReady = true;
      // If Newton params have zeroed roots (e.g. restored from URL), compute them now.
      if (this.fractalKind === "newton" && this.newtonParams.rootsRe.every(r => r === 0)) {
        this.orbitWorker.postMessage({
          type: "compute_roots",
          coeffs: Array.from(this.newtonParams.coeffs.subarray(0, this.newtonParams.degree + 1)),
        });
      }
      // Always retry — scheduleDispatch will request the orbit if needed.
      this.scheduleDispatch();
    } else if (msg.type === "orbit_ready") {
      this.onOrbitReady(msg.ref_orbit_id, msg.orbit_len, msg.bla_len);
    }
  }

  private onOrbitReady(ref_orbit_id: number, orbit_len: number, bla_len: number): void {
    console.log(`[Session] orbit_ready id=${ref_orbit_id} orbit_len=${orbit_len} bla_len=${bla_len}`);
    if (ref_orbit_id !== this.currentOrbitId) {
      console.log(`[Session] stale orbit (current=${this.currentOrbitId}), ignoring`);
      return;
    }

    this.orbitReady = true;
    this.pendingOrbitDispatch = false;

    const view = this.fsm.getView();
    // Forward to all Tile Workers (ordered before subsequent render_tile messages).
    const orbitMsg = {
      type: "orbit_ready",
      primaryOrbitDataOffset: PRIMARY_ORBIT_DATA_OFFSET,
      blaTableOffset: blaTableOffset(MAX_ORBIT_ITER),
      orbit_len,
      bla_len,
      cx_ref: parseFloat(view.cx),
      cy_ref: parseFloat(view.cy),
    };
    for (const w of this.workers) w.postMessage(orbitMsg);

    this.scheduleDispatch();
  }

  /** Write Newton state to window.location.hash without triggering a navigation. */
  private writeUrlHash(): void {
    const fragment = serializeNewtonState(this.newtonParams);
    window.history.replaceState(null, "", `#${fragment}`);
  }

  private onWorkerMessage(workerIndex: number, msg: WorkerInMsg): void {
    if (msg.type === "init_done") {
      this.workerInitCount++;
      if (this.workerInitCount === N_WORKERS) {
        this.lutReady = true;
        this.scheduleDispatch();
      }
    } else if (msg.type === "tile_ready") {
      this.onTileReady(workerIndex, msg);
    }
  }

  private buildDprToggle(canvas: HTMLCanvasElement): HTMLButtonElement {
    const btn = document.createElement("button");
    const update = () => {
      const dpr = window.devicePixelRatio || 1;
      btn.textContent = this.hiDpi ? `HiDPI ${dpr.toFixed(1)}×` : "HiDPI off";
    };
    update();
    Object.assign(btn.style, {
      display: "block", marginTop: "4px", cursor: "pointer",
      background: "rgba(255,255,255,0.15)", color: "white",
      border: "1px solid rgba(255,255,255,0.3)", borderRadius: "3px",
      fontFamily: "monospace", fontSize: "12px", padding: "2px 6px",
      width: "100%", boxSizing: "border-box", pointerEvents: "auto",
    });
    btn.addEventListener("click", () => {
      this.hiDpi = !this.hiDpi;
      const scale = this.hiDpi ? (window.devicePixelRatio || 1) : 1;
      canvas.width  = Math.round(window.innerWidth  * scale);
      canvas.height = Math.round(window.innerHeight * scale);
      canvas.style.width  = window.innerWidth  + "px";
      canvas.style.height = window.innerHeight + "px";
      this.gl.resize(canvas.width, canvas.height);
      this.fsm.setCanvasSize(canvas.width, canvas.height);
      this.fsm.setPixelScale(scale);
      update();
      this.scheduleDispatch();
    });
    return btn;
  }

  private updateOverlay(): void {
    const v = this.fsm.getView();
    const state = this.fsm.getState();
    const mode = v.zoom_exp > F64X2_ZOOM_THRESHOLD ? "f64x2" : "f64";
    this.statsEl.innerHTML =
      `state: ${state} [${mode}]<br>` +
      `cx: ${parseFloat(v.cx).toFixed(8)}<br>` +
      `cy: ${parseFloat(v.cy).toFixed(8)}<br>` +
      `zoom: 10^${v.zoom_exp.toFixed(3)}<br>` +
      `gen: ${this.generation} | workers: ${N_WORKERS}`;
  }

  private debouncedDispatch(delayMs: number): void {
    if (!this.lutReady) return;
    if (this.dispatchDebounceId !== null) clearTimeout(this.dispatchDebounceId);
    this.dispatchDebounceId = setTimeout(() => {
      this.dispatchDebounceId = null;
      this.scheduleDispatch();
    }, delayMs);
  }

  private scheduleDispatch(): void {
    if (!this.lutReady) return;
    const view = this.fsm.getView();
    const useF64x2 = view.zoom_exp > F64X2_ZOOM_THRESHOLD;
    this.dispatchGeneration(view, useF64x2);
  }

  private dispatchGeneration(
    view: { cx: string; cy: string; zoom_exp: number },
    useF64x2: boolean,
  ): void {
    this.generation++;
    const gen = this.generation;
    this.pendingTiles = [];

    // Sync shader with current fractal kind.
    this.gl.setFractalKind(this.fractalKind === "newton" ? 1 : 0);
    if (this.fractalKind === "newton") {
      this.gl.setNewtonDegree(this.newtonParams.degree);
    }

    const canvas = this.gl.canvas;
    const W = canvas.width as number;
    const H = canvas.height as number;
    const step = pixelStep(view.zoom_exp, H);
    const maxIter = this.computeMaxIter(view.zoom_exp);
    const tilesX = Math.ceil(W / TILE_SIZE);
    const tilesY = Math.ceil(H / TILE_SIZE);
    const cxRef = parseFloat(view.cx);
    const cyRef = parseFloat(view.cy);

    this.tilesThisGen = tilesX * tilesY;
    this.tilesDrawnThisGen = 0;
    this.updateOverlay();

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        let deltaRe: number;
        let deltaIm: number;

        if (useF64x2) {
          // F64x2 path: pass offset from viewport center so WASM can split
          // the coordinate as F64x2(cxRef, 0) + F64x2(offset, 0) without
          // catastrophic cancellation.
          deltaRe = (tx * TILE_SIZE - W / 2) * step;
          deltaIm = (ty * TILE_SIZE - H / 2) * step;
        } else {
          // f64 escape-time: absolute fractal coords of tile top-left.
          const { fx, fy } = pixelToFractal(tx * TILE_SIZE, ty * TILE_SIZE, view, W, H);
          deltaRe = fx;
          deltaIm = fy;
        }

        const newton = this.fractalKind === "newton" ? {
          degree:  this.newtonParams.degree,
          coeffs:  Array.from(this.newtonParams.coeffs),
          rootsRe: Array.from(this.newtonParams.rootsRe),
          rootsIm: Array.from(this.newtonParams.rootsIm),
        } : undefined;

        this.pendingTiles.push({
          deltaRe, deltaIm, step, maxIter,
          tileX: tx, tileY: ty,
          slotIndex: 0, // overwritten by dispatchPending
          generation: gen,
          useF64x2,
          cxRef,
          cyRef,
          fractalKind: this.fractalKind,
          newton,
        });
      }
    }

    this.dispatchPending();
  }

  private computeMaxIter(zoom_exp: number): number {
    return Math.max(BASE_ITER, Math.round(BASE_ITER + Math.max(0, zoom_exp) * ITER_PER_DECADE));
  }

  /**
   * Pause viewport tile dispatch and hand the worker pool to the caller.
   * Call WorkerLease.release() (always in a finally block) to resume.
   */
  pause(): WorkerLease {
    this.pendingTiles = [];
    this.paused = true;
    const self = this;
    return {
      workers: this.workers,
      tileSab: this.tileSab,
      replaceWorker(index: number) {
        self.workers[index].terminate();
        const w = new Worker(new URL("./tile-worker.ts", import.meta.url), { type: "module" });
        w.onerror = (e) => console.error(`[FractalSession] tile worker ${index} error`, e);
        w.postMessage({
          type: "init",
          wasmUrl: wasmBundleUrl(),
          tileSab: self.tileSab,
          orbitSab: self.orbitSab,
          primaryOrbitDataOffset: PRIMARY_ORBIT_DATA_OFFSET,
          blaTableOffset: blaTableOffset(MAX_ORBIT_ITER),
        });
        self.workers[index] = w;
      },
      release() {
        self.workers.forEach((w, i) => {
          w.onmessage = (e: MessageEvent<WorkerInMsg>) => self.onWorkerMessage(i, e.data);
        });
        self.paused = false;
        self.scheduleDispatch();
      },
    };
  }

  private dispatchPending(): void {
    for (let i = 0; i < this.workers.length; i++) {
      if (this.pendingTiles.length === 0) break;
      if (this.busyWorkers.has(i)) continue;
      const tile = this.pendingTiles.shift()!;
      tile.slotIndex = i;
      this.busyWorkers.add(i);
      this.workers[i].postMessage({ type: "render_tile", ...tile });
    }
  }

  private onTileReady(
    workerIndex: number,
    msg: { slotIndex: number; tileX: number; tileY: number; generation: number },
  ): void {
    this.busyWorkers.delete(workerIndex);

    // Discard viewport tiles silently while export owns the worker pool.
    if (this.paused) return;

    if (msg.generation !== this.generation) {
      this.dispatchPending();
      return;
    }

    this.gl.uploadAndDrawTile(this.tileSab, workerIndex, msg.tileX, msg.tileY);
    this.tilesDrawnThisGen++;

    if (this.tilesDrawnThisGen === this.tilesThisGen) {
      (this.gl.canvas as HTMLCanvasElement).dataset.rendered = String(msg.generation);
    }

    this.dispatchPending();
  }
}
