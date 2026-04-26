/**
 * FractalSession — main-thread orchestrator for Phase 1.
 *
 * Responsibilities:
 *  - Own the ZoomPanFSM and wire it to the canvas
 *  - Tile the canvas into 256×256 chunks and dispatch render jobs to workers
 *  - Receive completed tiles and upload them to the GL pipeline
 *
 * No WASM runs on the main thread. All computation is in Tile Workers.
 */
import { ZoomPanFSM, DEFAULT_VIEW, pixelToFractal, pixelStep } from "./viewport.ts";
import { GlPipeline } from "./gl-pipeline.ts";
import { wasmBundleUrl } from "./detect-simd.ts";

const MAX_ITER = 256;
const RING_SLOTS = 4;
const TILE_SIZE = 256;
const TILE_SLOT_BYTES = TILE_SIZE * TILE_SIZE * 4 * 4;
const SLOT_EMPTY = 0;

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

export class FractalSession {
  private readonly fsm: ZoomPanFSM;
  private readonly gl: GlPipeline;
  private readonly worker: Worker;
  private readonly slotStateSab: SharedArrayBuffer;
  private readonly tileSab: SharedArrayBuffer;
  private pendingTiles: TileDesc[] = [];
  private readonly activeSlots = new Set<number>();
  private generation = 0;
  private lutReady = false;
  private tilesThisGen = 0;
  private tilesDrawnThisGen = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.slotStateSab = new SharedArrayBuffer(RING_SLOTS * 4);
    this.tileSab = new SharedArrayBuffer(RING_SLOTS * TILE_SLOT_BYTES);
    this.gl = new GlPipeline(canvas);

    this.fsm = new ZoomPanFSM(DEFAULT_VIEW, {
      onViewChange: () => this.scheduleDispatch(),
      onZoomSettled: () => this.scheduleDispatch(),
    });
    this.fsm.setCanvasSize(canvas.width, canvas.height);
    this.fsm.attach(canvas);

    this.worker = new Worker(new URL("./tile-worker.ts", import.meta.url), { type: "module" });
    this.worker.onerror = (e) => console.error("[FractalSession] worker error", e);
    this.worker.onmessage = (e: MessageEvent<WorkerInMsg>) => this.onWorkerMessage(e.data);
    this.worker.postMessage({
      type: "init",
      wasmUrl: wasmBundleUrl(),
      slotStateSab: this.slotStateSab,
      tileSab: this.tileSab,
    });
  }

  private onWorkerMessage(msg: WorkerInMsg): void {
    if (msg.type === "init_done") {
      this.gl.uploadLut(msg.lut);
      this.lutReady = true;
      this.scheduleDispatch();
    } else if (msg.type === "tile_ready") {
      this.onTileReady(msg);
    }
  }

  private scheduleDispatch(): void {
    if (!this.lutReady) return;

    this.generation++;
    const gen = this.generation;
    this.pendingTiles = [];
    this.activeSlots.clear();

    // Reset all slot states so dispatchPending can freely use them.
    const stateArray = new Int32Array(this.slotStateSab);
    for (let i = 0; i < RING_SLOTS; i++) {
      Atomics.store(stateArray, i, SLOT_EMPTY);
    }

    const view = this.fsm.getView();
    const canvas = this.gl.canvas;
    const W = canvas.width as number;
    const H = canvas.height as number;
    const step = pixelStep(view.zoom_exp, H);
    const tilesX = Math.ceil(W / TILE_SIZE);
    const tilesY = Math.ceil(H / TILE_SIZE);

    this.tilesThisGen = tilesX * tilesY;
    this.tilesDrawnThisGen = 0;
    this.gl.clear();

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const { fx: deltaRe, fy: deltaIm } = pixelToFractal(
          tx * TILE_SIZE,
          ty * TILE_SIZE,
          view,
          W,
          H,
        );
        this.pendingTiles.push({ deltaRe, deltaIm, step, tileX: tx, tileY: ty, slotIndex: 0, generation: gen });
      }
    }

    this.dispatchPending();
  }

  private dispatchPending(): void {
    const stateArray = new Int32Array(this.slotStateSab);
    while (this.pendingTiles.length > 0) {
      let slotIndex = -1;
      for (let i = 0; i < RING_SLOTS; i++) {
        if (!this.activeSlots.has(i) && Atomics.load(stateArray, i) === SLOT_EMPTY) {
          slotIndex = i;
          break;
        }
      }
      if (slotIndex === -1) break;

      const tile = this.pendingTiles.shift()!;
      tile.slotIndex = slotIndex;
      this.activeSlots.add(slotIndex);
      this.worker.postMessage({ type: "render_tile", ...tile });
    }
  }

  private onTileReady(msg: {
    slotIndex: number;
    tileX: number;
    tileY: number;
    generation: number;
  }): void {
    const { slotIndex, tileX, tileY, generation } = msg;

    // Discard stale completions from a superseded generation.
    if (generation !== this.generation) {
      const stateArray = new Int32Array(this.slotStateSab);
      Atomics.store(stateArray, slotIndex, SLOT_EMPTY);
      return;
    }

    this.activeSlots.delete(slotIndex);
    this.gl.uploadAndDrawTile(this.tileSab, slotIndex, tileX, tileY);
    this.tilesDrawnThisGen++;

    const stateArray = new Int32Array(this.slotStateSab);
    Atomics.store(stateArray, slotIndex, SLOT_EMPTY);

    // Signal full-frame completion for E2E tests.
    if (this.tilesDrawnThisGen === this.tilesThisGen) {
      (this.gl.canvas as HTMLCanvasElement).dataset.rendered = String(generation);
    }

    this.dispatchPending();
  }
}
