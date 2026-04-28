// Force ES module scope (prevents global variable collisions with other worker files).
export {};

/**
 * Orbit Worker — runs in a dedicated Web Worker thread.
 *
 * Computes the BigFloat reference orbit and BLA table for deep-zoom rendering.
 * Writes the results into the shared `orbitSab` (SharedArrayBuffer), then posts
 * `orbit_ready` to signal the main thread. The postMessage itself acts as the
 * memory fence — all writes to orbitSab are visible to Tile Workers after they
 * receive the forwarded signal.
 *
 * Memory model: WASM computes orbit + BLA in its own heap, we copy into orbitSab
 * via Uint8Array.set(). This avoids the nightly -Z build-std requirement for
 * injecting a SharedArrayBuffer as WASM linear memory.
 */

/** Bytes per Complex<f64> orbit entry: two f64 = 16 bytes. */
const COMPLEX_F64_BYTES = 16;
/** Bytes per BlaEntry (repr(C), 48 bytes: a 16B + b 16B + skip 4B + pad 4B + valid_radius 8B). */
const BLA_ENTRY_BYTES = 48;

interface OrbitInitMsg {
  type: "orbit_init";
  wasmUrl: string;
  orbitSab: SharedArrayBuffer;
  /** Byte offset in orbitSab where orbit Complex<f64> entries start (after the 8-byte header). */
  primaryOrbitDataOffset: number;
  /** Byte offset in orbitSab where BlaEntry array starts. */
  blaTableOffset: number;
}

interface ComputeOrbitMsg {
  type: "compute_orbit";
  cx: string;
  cy: string;
  zoom_exp: number;
  max_iter: number;
  ref_orbit_id: number;
}

type OrbitWorkerMsg = OrbitInitMsg | ComputeOrbitMsg;

interface WasmExports {
  memory: WebAssembly.Memory;
}

interface WasmGlue {
  compute_reference_orbit: (cx: string, cy: string, zoom_exp: number, max_iter: number) => void;
  orbit_data_ptr: () => number;
  orbit_data_len: () => number;
  bla_data_ptr: () => number;
  bla_data_len: () => number;
}

let wasm: WasmExports | null = null;
let glue: WasmGlue | null = null;
let orbitSab: SharedArrayBuffer | null = null;
let primaryOrbitDataOffset = 0;
let blaTableOffset = 0;

self.onmessage = async (event: MessageEvent<OrbitWorkerMsg>) => {
  const msg = event.data;
  if (msg.type === "orbit_init") {
    await handleInit(msg);
  } else if (msg.type === "compute_orbit") {
    handleComputeOrbit(msg);
  }
};

async function handleInit(msg: OrbitInitMsg): Promise<void> {
  orbitSab = msg.orbitSab;
  primaryOrbitDataOffset = msg.primaryOrbitDataOffset;
  blaTableOffset = msg.blaTableOffset;

  const glueUrl = msg.wasmUrl.replace("_bg.wasm", ".js");
  const glueModule = await import(/* @vite-ignore */ glueUrl);
  wasm = (await glueModule.default({ module_or_path: msg.wasmUrl })) as WasmExports;
  glue = glueModule as WasmGlue;

  (self as unknown as Worker).postMessage({ type: "orbit_worker_ready" });
}

function handleComputeOrbit(msg: ComputeOrbitMsg): void {
  if (!wasm || !glue || !orbitSab) {
    console.error("[OrbitWorker] compute_orbit called before init");
    return;
  }

  const { cx, cy, zoom_exp, max_iter, ref_orbit_id } = msg;

  // Compute in WASM heap (BigFloat<N> precision selected by zoom_exp).
  glue.compute_reference_orbit(cx, cy, zoom_exp, max_iter);

  // orbit_len may be orbit_non_escaped + 1 for exterior references (the extra
  // entry is Z_N, the first escaped value; perturb_pixel needs it).
  const orbit_len = glue.orbit_data_len();
  const bla_len = glue.bla_data_len(); // non-escaped count, ≤ orbit_len
  const orbitPtr = glue.orbit_data_ptr();
  const blaPtr = glue.bla_data_ptr();
  const mem = wasm.memory.buffer;

  // Copy orbit bytes (orbit_len × 16 B) into orbitSab at primaryOrbitDataOffset.
  const orbitByteCount = orbit_len * COMPLEX_F64_BYTES;
  new Uint8Array(orbitSab, primaryOrbitDataOffset, orbitByteCount).set(
    new Uint8Array(mem, orbitPtr, orbitByteCount),
  );

  // Copy BLA bytes (bla_len × 48 B) into orbitSab at blaTableOffset.
  const blaByteCount = bla_len * BLA_ENTRY_BYTES;
  new Uint8Array(orbitSab, blaTableOffset, blaByteCount).set(
    new Uint8Array(mem, blaPtr, blaByteCount),
  );

  (self as unknown as Worker).postMessage({
    type: "orbit_ready",
    ref_orbit_id,
    orbit_len,
    bla_len,
  });
}
