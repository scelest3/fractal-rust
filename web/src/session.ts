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

const BASE_ITER = 256;
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

/** Zoom depth at which the perturbation path is engaged. */
const PERTURB_ZOOM_THRESHOLD = 15;

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
  usePerturb: boolean;
}

type WorkerInMsg =
  | { type: "init_done"; lut: Float32Array }
  | { type: "tile_ready"; slotIndex: number; tileX: number; tileY: number; generation: number };

type OrbitWorkerInMsg =
  | { type: "orbit_worker_ready" }
  | { type: "orbit_ready"; ref_orbit_id: number; orbit_len: number; bla_len: number };

function makeOverlay(): HTMLElement {
  const el = document.createElement("div");
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
  private readonly gl: GlPipeline;
  private readonly workers: Worker[];
  private readonly tileSab: SharedArrayBuffer;
  private readonly orbitSab: SharedArrayBuffer;
  private readonly orbitWorker: Worker;
  private readonly overlay: HTMLElement;

  private readonly busyWorkers = new Set<number>();
  private workerInitCount = 0;
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

  constructor(canvas: HTMLCanvasElement) {
    this.overlay = makeOverlay();
    this.tileSab = new SharedArrayBuffer(RING_SLOTS * TILE_SLOT_BYTES);
    this.orbitSab = new SharedArrayBuffer(orbitSabSize(MAX_ORBIT_ITER));
    this.gl = new GlPipeline(canvas);

    this.fsm = new ZoomPanFSM(DEFAULT_VIEW, {
      onViewChange: () => {
        this.updateOverlay();
        if (this.fsm.getState() === "PANNING") this.debouncedDispatch(150);
      },
      onZoomSettled: () => this.debouncedDispatch(0),
    });
    this.fsm.setCanvasSize(canvas.width, canvas.height);
    this.fsm.attach(canvas);

    new BoxZoom(canvas, () => this.fsm.getView(), (view) => {
      this.fsm.setView(view);
      this.scheduleDispatch();
    });

    const wasmUrl = wasmBundleUrl();

    // Orbit Worker — one dedicated worker for BigFloat reference orbit computation.
    this.orbitWorker = new Worker(
      new URL("./orbit-worker.ts", import.meta.url),
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

  private onOrbitWorkerMessage(msg: OrbitWorkerInMsg): void {
    if (msg.type === "orbit_worker_ready") {
      console.log("[Session] orbit worker ready");
      this.orbitWorkerReady = true;
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

    // Forward to all Tile Workers (ordered before subsequent render_tile messages).
    const orbitMsg = {
      type: "orbit_ready",
      primaryOrbitDataOffset: PRIMARY_ORBIT_DATA_OFFSET,
      blaTableOffset: blaTableOffset(MAX_ORBIT_ITER),
      orbit_len,
      bla_len,
    };
    for (const w of this.workers) w.postMessage(orbitMsg);

    this.scheduleDispatch();
  }

  private onWorkerMessage(workerIndex: number, msg: WorkerInMsg): void {
    if (msg.type === "init_done") {
      if (this.workerInitCount === 0) this.gl.uploadLut(msg.lut);
      this.workerInitCount++;
      if (this.workerInitCount === N_WORKERS) {
        this.lutReady = true;
        this.scheduleDispatch();
      }
    } else if (msg.type === "tile_ready") {
      this.onTileReady(workerIndex, msg);
    }
  }

  private updateOverlay(): void {
    const v = this.fsm.getView();
    const state = this.fsm.getState();
    const mode = v.zoom_exp > PERTURB_ZOOM_THRESHOLD ? "perturb" : "escape_time";
    this.overlay.innerHTML =
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
    const usePerturb = view.zoom_exp > PERTURB_ZOOM_THRESHOLD;

    if (usePerturb) {
      if (!this.orbitWorkerReady) {
        // Orbit worker still initialising; onOrbitWorkerMessage will retry.
        return;
      }
      if (!this.orbitReady) {
        // Request a new orbit if we haven't already for this generation.
        if (!this.pendingOrbitDispatch) {
          this.pendingOrbitDispatch = true;
          this.currentOrbitId++;
          const maxIter = this.computeMaxIter(view.zoom_exp);
          console.log(`[Session] requesting orbit id=${this.currentOrbitId} zoom_exp=${view.zoom_exp.toFixed(2)} max_iter=${maxIter}`);
          this.orbitWorker.postMessage({
            type: "compute_orbit",
            cx: view.cx,
            cy: view.cy,
            zoom_exp: view.zoom_exp,
            max_iter: maxIter,
            ref_orbit_id: this.currentOrbitId,
          });
        }
        return; // Tiles dispatched after orbit_ready arrives.
      }
    } else {
      // Escape-time path: no orbit needed; reset orbit state.
      this.orbitReady = false;
      this.pendingOrbitDispatch = false;
    }

    this.dispatchGeneration(view, usePerturb);
  }

  private dispatchGeneration(
    view: { cx: string; cy: string; zoom_exp: number },
    usePerturb: boolean,
  ): void {
    this.generation++;
    const gen = this.generation;
    this.pendingTiles = [];

    const canvas = this.gl.canvas;
    const W = canvas.width as number;
    const H = canvas.height as number;
    const step = pixelStep(view.zoom_exp, H);
    const maxIter = this.computeMaxIter(view.zoom_exp);
    const tilesX = Math.ceil(W / TILE_SIZE);
    const tilesY = Math.ceil(H / TILE_SIZE);

    this.tilesThisGen = tilesX * tilesY;
    this.tilesDrawnThisGen = 0;
    this.updateOverlay();

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        let deltaRe: number;
        let deltaIm: number;

        if (usePerturb) {
          // Perturbation: delta from reference (viewport center).
          // Computed as integer_offset × step — no catastrophic cancellation.
          deltaRe = (tx * TILE_SIZE - W / 2) * step;
          deltaIm = (ty * TILE_SIZE - H / 2) * step;
        } else {
          // Escape-time: absolute fractal coords of tile top-left.
          const { fx, fy } = pixelToFractal(tx * TILE_SIZE, ty * TILE_SIZE, view, W, H);
          deltaRe = fx;
          deltaIm = fy;
        }

        this.pendingTiles.push({
          deltaRe, deltaIm, step, maxIter,
          tileX: tx, tileY: ty,
          slotIndex: 0, // overwritten by dispatchPending
          generation: gen,
          usePerturb,
        });
      }
    }

    this.dispatchPending();
  }

  private computeMaxIter(zoom_exp: number): number {
    return Math.max(BASE_ITER, Math.round(BASE_ITER + Math.max(0, zoom_exp) * ITER_PER_DECADE));
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
