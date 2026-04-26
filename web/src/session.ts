/**
 * FractalSession — main-thread orchestrator for Phase 1.
 *
 * Responsibilities:
 *  - Own the ZoomPanFSM and wire it to the canvas
 *  - Tile the canvas into 256×256 chunks and dispatch render jobs to workers
 *  - Receive completed tiles and upload them to the GL pipeline
 *
 * No WASM runs on the main thread. All computation is in Tile Workers.
 *
 * Worker count: hardwareConcurrency − 2 (min 2, max 8). Each worker owns one
 * SAB slot — worker k always writes to slot k — so N_WORKERS === RING_SLOTS.
 * Workers are independent WASM instances; no shared WASM memory is needed for
 * Phase 1's copy-to-SAB approach. Phase 2 will eliminate the copy.
 */
import { ZoomPanFSM, DEFAULT_VIEW, pixelToFractal, pixelStep } from "./viewport.ts";
import { GlPipeline } from "./gl-pipeline.ts";
import { wasmBundleUrl } from "./detect-simd.ts";

const MAX_ITER = 256;
const TILE_SIZE = 256;
const TILE_SLOT_BYTES = TILE_SIZE * TILE_SIZE * 4 * 4; // 1 MiB per slot

// Reserve 2 threads (main + one headroom) and cap at 8 to control SAB size.
const N_WORKERS = Math.max(2, Math.min(8, (navigator.hardwareConcurrency ?? 4) - 2));
// One slot per worker: worker k always writes to slot k.
const RING_SLOTS = N_WORKERS;

interface TileDesc {
  deltaRe: number;
  deltaIm: number;
  step: number;
  tileX: number;
  tileY: number;
  slotIndex: number;
  generation: number;
}

type WorkerInMsg =
  | { type: "init_done"; lut: Float32Array }
  | { type: "tile_ready"; slotIndex: number; tileX: number; tileY: number; generation: number };

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

  // Tracks workers that have been dispatched a tile and haven't yet responded.
  // A worker is removed from this set when any tile_ready arrives from it,
  // regardless of generation — the worker is genuinely idle at that point.
  private readonly busyWorkers = new Set<number>();

  private workerInitCount = 0;
  private pendingTiles: TileDesc[] = [];
  private generation = 0;
  private lutReady = false;
  private tilesThisGen = 0;
  private tilesDrawnThisGen = 0;
  private dispatchDebounceId: ReturnType<typeof setTimeout> | null = null;
  private readonly overlay: HTMLElement;

  constructor(canvas: HTMLCanvasElement) {
    this.overlay = makeOverlay();
    this.tileSab = new SharedArrayBuffer(RING_SLOTS * TILE_SLOT_BYTES);
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

    const wasmUrl = wasmBundleUrl();
    this.workers = Array.from({ length: N_WORKERS }, (_, i) => {
      const w = new Worker(new URL("./tile-worker.ts", import.meta.url), { type: "module" });
      w.onerror = (e) => console.error(`[FractalSession] worker ${i} error`, e);
      w.onmessage = (e: MessageEvent<WorkerInMsg>) => this.onWorkerMessage(i, e.data);
      w.postMessage({ type: "init", wasmUrl, tileSab: this.tileSab });
      return w;
    });
  }

  private onWorkerMessage(workerIndex: number, msg: WorkerInMsg): void {
    if (msg.type === "init_done") {
      // Upload LUT from the first worker that reports in; all workers produce
      // identical LUTs so subsequent ones are discarded.
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
    this.overlay.innerHTML =
      `state: ${state}<br>` +
      `cx: ${parseFloat(v.cx).toFixed(8)}<br>` +
      `cy: ${parseFloat(v.cy).toFixed(8)}<br>` +
      `zoom: ${v.zoom_exp.toFixed(4)}<br>` +
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

    this.generation++;
    const gen = this.generation;
    this.pendingTiles = [];

    const view = this.fsm.getView();
    const canvas = this.gl.canvas;
    const W = canvas.width as number;
    const H = canvas.height as number;
    const step = pixelStep(view.zoom_exp, H);
    const tilesX = Math.ceil(W / TILE_SIZE);
    const tilesY = Math.ceil(H / TILE_SIZE);

    this.tilesThisGen = tilesX * tilesY;
    this.tilesDrawnThisGen = 0;
    this.updateOverlay();

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const { fx: deltaRe, fy: deltaIm } = pixelToFractal(
          tx * TILE_SIZE,
          ty * TILE_SIZE,
          view,
          W,
          H,
        );
        this.pendingTiles.push({
          deltaRe, deltaIm, step,
          tileX: tx, tileY: ty,
          slotIndex: 0, // overwritten by dispatchPending
          generation: gen,
        });
      }
    }

    this.dispatchPending();
  }

  private dispatchPending(): void {
    for (let i = 0; i < this.workers.length; i++) {
      if (this.pendingTiles.length === 0) break;
      if (this.busyWorkers.has(i)) continue;
      const tile = this.pendingTiles.shift()!;
      tile.slotIndex = i; // worker i owns slot i
      this.busyWorkers.add(i);
      this.workers[i].postMessage({ type: "render_tile", ...tile });
    }
  }

  private onTileReady(
    workerIndex: number,
    msg: { slotIndex: number; tileX: number; tileY: number; generation: number },
  ): void {
    // Worker is idle regardless of generation — unblock it before anything else.
    this.busyWorkers.delete(workerIndex);

    if (msg.generation !== this.generation) {
      // Stale completion from a previous generation; give the worker new work.
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
