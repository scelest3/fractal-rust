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
  private readonly worker: Worker;
  private readonly slotStateSab: SharedArrayBuffer;
  private readonly tileSab: SharedArrayBuffer;
  private pendingTiles: TileDesc[] = [];
  private readonly activeSlots = new Set<number>();
  private generation = 0;
  private lutReady = false;
  private tilesThisGen = 0;
  private tilesDrawnThisGen = 0;
  private dispatchDebounceId: ReturnType<typeof setTimeout> | null = null;
  private readonly overlay: HTMLElement;

  constructor(canvas: HTMLCanvasElement) {
    this.overlay = makeOverlay();
    this.slotStateSab = new SharedArrayBuffer(RING_SLOTS * 4);
    this.tileSab = new SharedArrayBuffer(RING_SLOTS * TILE_SLOT_BYTES);
    this.gl = new GlPipeline(canvas);

    this.fsm = new ZoomPanFSM(DEFAULT_VIEW, {
      // During pan: debounce 150 ms after the pointer stops moving.
      // During zoom: skip — onZoomSettled handles it with its own 300 ms debounce.
      onViewChange: () => {
        this.updateOverlay();
        if (this.fsm.getState() === "PANNING") this.debouncedDispatch(150);
      },
      // Zoom settled: dispatch (reuses the same debounce slot, replacing any
      // pending pan debounce — zoom settle always wins).
      onZoomSettled: () => this.debouncedDispatch(0),
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

  private updateOverlay(): void {
    const v = this.fsm.getView();
    const state = this.fsm.getState();
    this.overlay.innerHTML =
      `state: ${state}<br>` +
      `cx: ${parseFloat(v.cx).toFixed(8)}<br>` +
      `cy: ${parseFloat(v.cy).toFixed(8)}<br>` +
      `zoom: ${v.zoom_exp.toFixed(4)}<br>` +
      `gen: ${this.generation}`;
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

    // Stale completion: slot state was already reset in scheduleDispatch.
    // Do not touch activeSlots or call dispatchPending — the new generation
    // already owns this slot.
    if (generation !== this.generation) return;

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
