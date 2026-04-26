/**
 * Tile Worker — runs in a Web Worker thread.
 *
 * Handles two message types:
 *   init        — load WASM, build LUT, signal main thread
 *   render_tile — render one 256×256 tile into a WASM heap buffer,
 *                 copy into the shared SAB slot, signal main thread
 *
 * Phase 1 memory model: WASM writes into its own heap (alloc_tile_buf),
 * then we copy once into the SharedArrayBuffer. Phase 2 will eliminate
 * the copy by compiling with -Z build-std so wasm-bindgen accepts an
 * injected SharedArrayBuffer as WASM linear memory.
 */

const TILE_SLOT_BYTES = 256 * 256 * 4 * 4; // 1 MiB per slot

interface InitMsg {
  type: "init";
  wasmUrl: string;
  tileSab: SharedArrayBuffer;
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
}

type WorkerMsg = InitMsg | RenderTileMsg;

// Raw WASM instance exports (alloc/free/render are primitive-typed, available on InitOutput).
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
}

// build_lut returns a boxed slice and lives on the JS glue module, not on InitOutput.
interface WasmGlue {
  build_lut: (palette: null) => Float32Array;
}

let wasm: WasmExports | null = null;
let glueModule: WasmGlue | null = null;
let tileSab: SharedArrayBuffer | null = null;

self.onmessage = async (event: MessageEvent<WorkerMsg>) => {
  const msg = event.data;
  if (msg.type === "init") {
    await handleInit(msg);
  } else if (msg.type === "render_tile") {
    handleRenderTile(msg);
  }
};

async function handleInit(msg: InitMsg): Promise<void> {
  tileSab = msg.tileSab;

  const glueUrl = msg.wasmUrl.replace("_bg.wasm", ".js");
  const glue = await import(/* @vite-ignore */ glueUrl);
  wasm = (await glue.default({ module_or_path: msg.wasmUrl })) as WasmExports;
  glueModule = glue as WasmGlue;

  // build_lut is a JS glue wrapper (handles boxed-slice memory), not a raw
  // WASM export — call it on the glue module, not on the InitOutput object.
  const lutView = glueModule.build_lut(null);
  const lut = new Float32Array(lutView);

  (self as unknown as Worker).postMessage({ type: "init_done", lut });
}

function handleRenderTile(msg: RenderTileMsg): void {
  if (!wasm || !tileSab) {
    console.error("[TileWorker] render_tile before init");
    return;
  }

  const { deltaRe, deltaIm, step, slotIndex, tileX, tileY, generation } = msg;
  const maxIter = msg.maxIter ?? 256;

  const ptr = wasm.alloc_tile_buf();
  wasm.render_tile_to_ptr(deltaRe, deltaIm, step, maxIter, ptr);

  // Copy WASM heap → SAB slot. Re-read memory.buffer in case WASM grew.
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
