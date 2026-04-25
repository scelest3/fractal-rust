/**
 * Main thread orchestrator — Phase 0 shared-memory hello-world.
 *
 * Memory model for Phase 0:
 *  - slotStateSab: SharedArrayBuffer for Atomic EMPTY/WRITING/READY coordination
 *  - tileSab: SharedArrayBuffer ring buffer for tile pixel data
 *
 * The Tile Worker instantiates WASM, writes a red tile into a WASM heap
 * allocation, copies it once into tileSab, then marks the slot READY.
 * The main thread reads from tileSab for the zero-copy GPU upload.
 *
 * One copy (WASM heap → tileSab) is acceptable for Phase 0. Phase 1 will
 * move to a proper shared-memory build where WASM writes directly into the
 * SAB, eliminating the copy in the hot render loop.
 *
 * No WASM runs on the main thread. session.ts (Phase 1) will replace this.
 */
import { detectSimd, wasmBundleUrl } from "./detect-simd.ts";
import { computeLayout } from "./layout.ts";
import { GlPipeline } from "./gl-pipeline.ts";

const SLOT_READY = 2;
const SLOT_EMPTY = 0;

async function main(): Promise<void> {
  // ── Canvas ───────────────────────────────────────────────────────────────
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  if (typeof SharedArrayBuffer === "undefined") {
    console.error(
      "SharedArrayBuffer unavailable — COOP/COEP headers missing. " +
        "Run the dev server with `npm run dev`.",
    );
    return;
  }

  console.log("SharedArrayBuffer ✓");
  console.log(`SIMD: ${detectSimd() ? "enabled ✓" : "disabled (non-SIMD bundle)"}`);

  // ── Shared buffers ───────────────────────────────────────────────────────
  // Tile layout uses 1 worker and enough iterations to cover the ring.
  const N_WORKERS = 1;
  const MAX_ITER = 1000;
  const layout = computeLayout(N_WORKERS, MAX_ITER);

  // slotStateSab: one Int32 per ring slot (EMPTY=0 / WRITING=1 / READY=2).
  const TOTAL_RING_SLOTS = 4 * N_WORKERS;
  const slotStateSab = new SharedArrayBuffer(TOTAL_RING_SLOTS * 4);

  // tileSab: pixel ring buffer; tile slot 0 is at byte 0.
  const tileSab = new SharedArrayBuffer(TOTAL_RING_SLOTS * layout.tileSlotBytes);

  console.log(
    `Tile ring: ${TOTAL_RING_SLOTS} slots × ` +
      `${(layout.tileSlotBytes / 1024).toFixed(0)} KiB = ` +
      `${((TOTAL_RING_SLOTS * layout.tileSlotBytes) / 1024 / 1024).toFixed(1)} MiB`,
  );

  // ── WebGL pipeline ───────────────────────────────────────────────────────
  const gl = new GlPipeline(canvas);

  // ── Tile Worker ──────────────────────────────────────────────────────────
  const worker = new Worker(new URL("./tile-worker.ts", import.meta.url), {
    type: "module",
  });

  worker.onerror = (e) => console.error("[Worker error]", e);

  worker.onmessage = (event: MessageEvent<{ type: string; slotIndex: number }>) => {
    if (event.data.type !== "tile_ready") return;

    const { slotIndex } = event.data;
    const stateArray = new Int32Array(slotStateSab);

    if (Atomics.load(stateArray, slotIndex) !== SLOT_READY) {
      console.error("Slot not READY after tile_ready message");
      return;
    }

    // Zero-copy GPU upload: Float32Array view directly into tileSab.
    gl.uploadTile(tileSab, slotIndex, layout.tileSlotBytes);
    gl.draw();

    // Release slot back to the ring.
    Atomics.store(stateArray, slotIndex, SLOT_EMPTY);

    console.log("Tile rendered to canvas ✓ — pipeline end-to-end confirmed");
  };

  // Send both SABs to the worker (structured clone — shared, not copied).
  worker.postMessage({
    wasmUrl: wasmBundleUrl(),
    slotStateSab,
    tileSab,
  });
}

main().catch(console.error);
