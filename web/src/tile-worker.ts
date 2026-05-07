// Force ES module scope (prevents global variable collisions with other worker files).
export {};

/**
 * Tile Worker — runs in a Web Worker thread.
 *
 * Handles three message types:
 *   init        — load WASM, receive SABs, build LUT, signal main thread
 *   orbit_ready — copy orbit + BLA from orbitSab into WASM heap via install_orbit
 *   render_tile — render one 256×256 tile (escape_time or perturbation path),
 *                 copy into the shared tileSab slot, signal main thread
 *
 * Perturbation path (zoom_exp > 15):
 *   delta_re / delta_im are offsets from the reference point (viewport center),
 *   computed as (tile_pixel - W/2) × pixel_step. The WASM uses the pre-installed
 *   orbit to call perturb_pixel per pixel.
 *
 * Phase 2 memory model: WASM writes into its own heap (alloc_tile_buf), then we
 * copy once into the shared tileSab slot. Phase 3 will eliminate the copy when
 * -Z build-std lands and wasm-bindgen can accept an injected SharedArrayBuffer.
 */

const TILE_SLOT_BYTES = 256 * 256 * 4 * 4; // 1 MiB per slot
const COMPLEX_F64_BYTES = 16;               // sizeof(Complex<f64>)
const BLA_ENTRY_BYTES = 48;                 // sizeof(BlaEntry)

interface InitMsg {
  type: "init";
  wasmUrl: string;
  tileSab: SharedArrayBuffer;
  orbitSab: SharedArrayBuffer;
  primaryOrbitDataOffset: number;
  blaTableOffset: number;
}

interface OrbitReadyMsg {
  type: "orbit_ready";
  primaryOrbitDataOffset: number;
  blaTableOffset: number;
  orbit_len: number;
  bla_len: number;
  cx_ref: number;
  cy_ref: number;
}

interface NewtonRenderParams {
  degree: number;
  coeffs: number[];
  rootsRe: number[];
  rootsIm: number[];
}

interface RenderTileMsg {
  type: "render_tile";
  deltaRe: number;
  deltaIm: number;
  step: number;
  maxIter?: number;
  slotIndex: number;
  tileX: number;
  tileY: number;
  generation: number;
  useF64x2: boolean;
  cxRef: number;
  cyRef: number;
  fractalKind: "mandelbrot" | "newton";
  newton?: NewtonRenderParams;
}

interface EncodePngMsg {
  type: "encode_png";
  pixels: Float32Array;
  width: number;
  height: number;
  lut: Float32Array;
  cycleLen: number;
  trapRadius: number;
  trapStrength: number;
  trapColor: [number, number, number];
  interiorColor: [number, number, number];
  shadingColor: [number, number, number];
  distanceStrength: number;
  distancePow: number;
  angleStrength: number;
  periodStrength: number;
  periodCycleLen: number;
  distWeight: number;
  fractalKind: number;
  newtonDegree: number;
  unresolvedColor: [number, number, number];
  dpi: number;
  cx: string;
  cy: string;
  zoomExp: number;
  maxIter: number;
  newtonDegreeMeta: number;
  creationTime: string;
}

type WorkerMsg = InitMsg | OrbitReadyMsg | RenderTileMsg | EncodePngMsg;

interface WasmExports {
  memory: WebAssembly.Memory;
  alloc_tile_buf: () => number;
  free_tile_buf: (ptr: number) => void;
  render_tile_to_ptr: (
    deltaRe: number,
    deltaIm: number,
    pixelStep: number,
    maxIter: number,
    ptr: number,
  ) => void;
  render_tile_f64x2: (
    cxRef: number,
    cyRef: number,
    deltaRe: number,
    deltaIm: number,
    pixelStep: number,
    maxIter: number,
    ptr: number,
  ) => void;
  render_tile_perturb: (
    cxRef: number,
    cyRef: number,
    deltaRe: number,
    deltaIm: number,
    pixelStep: number,
    maxIter: number,
    ptr: number,
  ) => number; // returns glitch count (v2)
}

interface WasmGlue {
  install_orbit: (orbitF64: Float64Array, blaBytes: Uint8Array) => void;
  // Slice parameters (&[f64]) require the JS glue wrapper, not the raw WASM export.
  render_tile_newton_to_ptr: (
    degree: number,
    coeffs: Float64Array,
    rootsRe: Float64Array,
    rootsIm: Float64Array,
    deltaRe: number,
    deltaIm: number,
    pixelStep: number,
    maxIter: number,
    ptr: number,
  ) => void;
  encode_png: (
    pixels: Float32Array,
    width: number,
    height: number,
    stripYOffset: number,
    stripHeight: number,
    lut: Float32Array,
    cycleLen: number,
    trapRadius: number,
    trapStrength: number,
    trapColorR: number, trapColorG: number, trapColorB: number,
    interiorColorR: number, interiorColorG: number, interiorColorB: number,
    shadingColorR: number, shadingColorG: number, shadingColorB: number,
    distanceStrength: number,
    distancePow: number,
    angleStrength: number,
    periodStrength: number,
    periodCycleLen: number,
    distWeight: number,
    fractalKind: number,
    newtonDegree: number,
    unresolvedColorR: number, unresolvedColorG: number, unresolvedColorB: number,
    dpi: number,
    cx: string,
    cy: string,
    zoomExp: number,
    maxIter: number,
    newtonDegreeMeta: number,
    creationTime: string,
  ) => Uint8Array;
}

let wasm: WasmExports | null = null;
let glueModule: WasmGlue | null = null;
let tileSab: SharedArrayBuffer | null = null;
let orbitSab: SharedArrayBuffer | null = null;
let primaryOrbitDataOffset = 0;
let blaTableOffset = 0;
let orbitCxRef = 0;
let orbitCyRef = 0;

self.onmessage = async (event: MessageEvent<WorkerMsg>) => {
  const msg = event.data;
  if (msg.type === "init") {
    await handleInit(msg);
  } else if (msg.type === "orbit_ready") {
    handleOrbitReady(msg);
  } else if (msg.type === "render_tile") {
    handleRenderTile(msg);
  } else if (msg.type === "encode_png") {
    handleEncodePng(msg);
  }
};

async function handleInit(msg: InitMsg): Promise<void> {
  tileSab = msg.tileSab;
  orbitSab = msg.orbitSab;
  primaryOrbitDataOffset = msg.primaryOrbitDataOffset;
  blaTableOffset = msg.blaTableOffset;

  const glueUrl = msg.wasmUrl.replace("_bg.wasm", ".js");
  const glue = await import(/* @vite-ignore */ glueUrl);
  wasm = (await glue.default({ module_or_path: msg.wasmUrl })) as WasmExports;
  glueModule = glue as WasmGlue;

  (self as unknown as Worker).postMessage({ type: "init_done" });
}

function handleOrbitReady(msg: OrbitReadyMsg): void {
  if (!wasm || !glueModule || !orbitSab) {
    console.warn("[TileWorker] orbit_ready before init, ignoring");
    return;
  }

  const { orbit_len, bla_len, cx_ref, cy_ref } = msg;
  orbitCxRef = cx_ref;
  orbitCyRef = cy_ref;
  console.log(`[TileWorker] installing orbit orbit_len=${orbit_len} bla_len=${bla_len}`);

  // Copy orbit from orbitSab (SharedArrayBuffer) into a regular ArrayBuffer,
  // then pass to WASM. wasm-bindgen cannot accept SAB-backed typed arrays.
  // orbit_len may be bla_len + 1 for exterior references (extra escaped entry).
  const orbitByteCount = orbit_len * COMPLEX_F64_BYTES;
  const orbitLocal = new Uint8Array(orbitByteCount);
  orbitLocal.set(new Uint8Array(orbitSab, primaryOrbitDataOffset, orbitByteCount));
  const orbitF64 = new Float64Array(orbitLocal.buffer);

  const blaByteCount = bla_len * BLA_ENTRY_BYTES;
  const blaLocal = new Uint8Array(blaByteCount);
  blaLocal.set(new Uint8Array(orbitSab, blaTableOffset, blaByteCount));

  glueModule.install_orbit(orbitF64, blaLocal);
}

function handleRenderTile(msg: RenderTileMsg): void {
  if (!wasm || !tileSab) {
    console.error("[TileWorker] render_tile before init");
    return;
  }

  const { deltaRe, deltaIm, step, slotIndex, tileX, tileY, generation,
          useF64x2, cxRef, cyRef, fractalKind } = msg;
  const maxIter = msg.maxIter ?? 256;

  const ptr = wasm.alloc_tile_buf();

  if (fractalKind === "newton" && msg.newton && glueModule) {
    const { degree, coeffs, rootsRe, rootsIm } = msg.newton;
    glueModule.render_tile_newton_to_ptr(
      degree,
      new Float64Array(coeffs),
      new Float64Array(rootsRe),
      new Float64Array(rootsIm),
      deltaRe, deltaIm, step, maxIter, ptr,
    );
  } else if (useF64x2) {
    wasm.render_tile_f64x2(cxRef, cyRef, deltaRe, deltaIm, step, maxIter, ptr);
  } else {
    wasm.render_tile_to_ptr(deltaRe, deltaIm, step, maxIter, ptr);
  }

  // Copy WASM heap → tileSab slot. Re-read memory.buffer in case WASM grew.
  const byteOffset = slotIndex * TILE_SLOT_BYTES;
  new Uint8Array(tileSab, byteOffset, TILE_SLOT_BYTES).set(
    new Uint8Array(wasm.memory.buffer, ptr, TILE_SLOT_BYTES),
  );

  wasm.free_tile_buf(ptr);

  (self as unknown as Worker).postMessage({
    type: "tile_ready",
    slotIndex,
    tileX,
    tileY,
    generation,
  });
}

function handleEncodePng(msg: EncodePngMsg): void {
  if (!glueModule) {
    (self as unknown as Worker).postMessage({ type: "encode_error", error: "encode_png called before init" });
    return;
  }
  try {
    const [tr, tg, tb] = msg.trapColor;
    const [ir, ig, ib] = msg.interiorColor;
    const [sr, sg, sb] = msg.shadingColor;
    const [ur, ug, ub] = msg.unresolvedColor;

    const bytes = glueModule.encode_png(
      msg.pixels, msg.width, msg.height,
      0, msg.height,
      msg.lut,
      msg.cycleLen,
      msg.trapRadius, msg.trapStrength,
      tr, tg, tb,
      ir, ig, ib,
      sr, sg, sb,
      msg.distanceStrength, msg.distancePow,
      msg.angleStrength,
      msg.periodStrength, msg.periodCycleLen,
      msg.distWeight,
      msg.fractalKind,
      msg.newtonDegree,
      ur, ug, ub,
      msg.dpi,
      msg.cx, msg.cy, msg.zoomExp,
      msg.maxIter,
      msg.newtonDegreeMeta,
      msg.creationTime,
    );
    (self as unknown as Worker).postMessage({ type: "encode_done", bytes }, [bytes.buffer]);
  } catch (e) {
    console.error("[TileWorker] encode_png failed:", e);
    (self as unknown as Worker).postMessage({ type: "encode_error", error: String(e) });
  }
}
