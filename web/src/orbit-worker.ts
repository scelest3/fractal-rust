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

interface ComputeRootsMsg {
  type: "compute_roots";
  /** Polynomial coefficients ascending by degree (up to 11 elements). */
  coeffs: number[];
}

type OrbitWorkerMsg = OrbitInitMsg | ComputeOrbitMsg | ComputeRootsMsg;

interface WasmExports {
  memory: WebAssembly.Memory;
}

interface RootResult {
  degree: number;
  roots_re: () => Float64Array;
  roots_im: () => Float64Array;
}

interface WasmGlue {
  compute_reference_orbit: (cx: string, cy: string, zoom_exp: number, max_iter: number) => void;
  orbit_data_ptr: () => number;
  orbit_data_len: () => number;
  bla_data_ptr: () => number;
  bla_data_len: () => number;
  compute_roots: (coeffs: Float64Array) => RootResult;
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
  } else if (msg.type === "compute_roots") {
    handleComputeRoots(msg);
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

function handleComputeRoots(msg: ComputeRootsMsg): void {
  if (!glue) {
    console.error("[OrbitWorker] compute_roots called before init");
    return;
  }
  try {
    const result = glue.compute_roots(new Float64Array(msg.coeffs));
    (self as unknown as Worker).postMessage({
      type: "roots_ready",
      degree:  result.degree,
      rootsRe: Array.from(result.roots_re()),
      rootsIm: Array.from(result.roots_im()),
    });
  } catch (err) {
    console.error("[OrbitWorker] compute_roots failed:", err);
  }
}

function handleComputeOrbit(msg: ComputeOrbitMsg): void {
  if (!wasm || !glue || !orbitSab) {
    console.error("[OrbitWorker] compute_orbit called before init");
    return;
  }

  const { cx, cy, zoom_exp, max_iter, ref_orbit_id } = msg;
  console.log(`[OrbitWorker] computing orbit cx=${cx} cy=${cy} zoom_exp=${zoom_exp} max_iter=${max_iter}`);

  try {
    // Compute in WASM heap (BigFloat<N> precision selected by zoom_exp).
    glue.compute_reference_orbit(cx, cy, zoom_exp, max_iter);

    // orbit_len may be bla_len + 1 for exterior references (the extra entry is
    // Z_N, the first escaped value; perturb_pixel needs it to detect escape).
    const orbit_len = glue.orbit_data_len();
    const bla_len = glue.bla_data_len();
    const orbitPtr = glue.orbit_data_ptr();
    const blaPtr = glue.bla_data_ptr();
    const mem = wasm.memory.buffer;

    console.log(`[OrbitWorker] orbit_len=${orbit_len} bla_len=${bla_len} orbitPtr=${orbitPtr} blaPtr=${blaPtr}`);

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
  } catch (err) {
    console.error("[OrbitWorker] compute_orbit failed:", err);
  }
}
