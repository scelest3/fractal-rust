/**
 * Tile Worker — runs in a Web Worker thread.
 *
 * Uses wasm-bindgen's generated init() via dynamic import so all internal
 * wasm-bindgen imports are satisfied automatically. Writes a solid-color tile
 * into a WASM heap allocation via write_solid_tile(), then copies it once into
 * the shared tileSab. One copy per tile (WASM heap → SAB) is acceptable for
 * Phase 0; Phase 1 will adopt a proper shared-memory build to eliminate it.
 */

const TILE_SLOT_BYTES = 256 * 256 * 4 * 4;
const SLOT_EMPTY = 0;
const SLOT_WRITING = 1;
const SLOT_READY = 2;

interface WorkerInit {
  wasmUrl: string;
  slotStateSab: SharedArrayBuffer;
  tileSab: SharedArrayBuffer;
}

// Raw exports returned by wasm-bindgen's init() — which returns instance.exports.
interface WasmInstanceExports {
  memory: WebAssembly.Memory;
  alloc_tile_buf: () => number;
  free_tile_buf: (ptr: number) => void;
  write_solid_tile: (ptr: number, r: number, g: number, b: number, a: number) => void;
}

self.onmessage = async (event: MessageEvent<WorkerInit>) => {
  const { wasmUrl, slotStateSab, tileSab } = event.data;

  // Derive the wasm-bindgen JS glue URL from the .wasm URL.
  const glueUrl = wasmUrl.replace("_bg.wasm", ".js");

  // Dynamic import of the wasm-bindgen glue. /* @vite-ignore */ tells Vite not
  // to bundle this at build time — it resolves at runtime from /pkg/.
  const glue = await import(/* @vite-ignore */ glueUrl);

  // init() constructs the correct import object, instantiates the WASM module,
  // and returns instance.exports directly (see __wbg_finalize_init in glue).
  // Pass { module_or_path } object — the string overload is deprecated in
  // wasm-bindgen 0.2.118+.
  const exports = (await glue.default({ module_or_path: wasmUrl })) as WasmInstanceExports;

  // ── Ring slot acquisition ─────────────────────────────────────────────
  const stateArray = new Int32Array(slotStateSab);
  const prev = Atomics.compareExchange(stateArray, 0, SLOT_EMPTY, SLOT_WRITING);
  if (prev !== SLOT_EMPTY) {
    console.error("[TileWorker] ring slot 0 not available (state =", prev, ")");
    return;
  }

  // ── Tile write via WASM ───────────────────────────────────────────────
  // alloc_tile_buf() allocates 1 MiB on the WASM heap — a safe region not
  // used by the Rust stack or static data. write_solid_tile() fills it.
  const ptr = exports.alloc_tile_buf();
  exports.write_solid_tile(ptr, 1.0, 0.0, 0.0, 1.0);

  // Copy tile from WASM heap → shared SAB.
  // One copy per tile is the Phase 0 trade-off; Phase 1 writes directly.
  // Re-read memory.buffer after alloc in case WASM grew its linear memory.
  new Uint8Array(tileSab, 0, TILE_SLOT_BYTES).set(
    new Uint8Array(exports.memory.buffer, ptr, TILE_SLOT_BYTES),
  );

  exports.free_tile_buf(ptr);

  // ── Signal main thread ────────────────────────────────────────────────
  Atomics.store(stateArray, 0, SLOT_READY);
  self.postMessage({ type: "tile_ready", slotIndex: 0 });
};
