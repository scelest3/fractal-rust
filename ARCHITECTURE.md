# Fractal Explorer — Architecture

> High-resolution fractal renderer · Rust / WASM / WebGL 2  
> Extended precision · Perturbation theory · GPU tile pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Rust Crate Structure](#4-rust-crate-structure)
5. [WASM Bridge](#5-wasm-bridge)
6. [WebGL 2 Rendering Pipeline](#6-webgl-2-rendering-pipeline)
7. [TypeScript UI Shell](#7-typescript-ui-shell)
8. [Extended Precision & Deep Zoom](#8-extended-precision--deep-zoom)
9. [Perturbation Theory](#9-perturbation-theory)
10. [Adding New Fractal Types](#10-adding-new-fractal-types)
11. [Web Worker Topology](#11-web-worker-topology)
12. [Image Export Pipeline](#12-image-export-pipeline)
13. [Build & Dev Tooling](#13-build--dev-tooling)
14. [Testing Strategy](#14-testing-strategy)
15. [Performance Targets](#15-performance-targets)
16. [Phased Roadmap](#16-phased-roadmap)
17. [Key Risks & Mitigations](#17-key-risks--mitigations)
18. [References](#18-references)

---

## 1. Executive Summary

Browser-based, high-resolution fractal explorer targeting sustained 60 fps interaction at viewport resolution and export quality up to 16 384 × 16 384 px. Supports zoom depths of 10⁻³⁰⁰ via perturbation theory with arbitrary-precision reference orbit computation. Zoom beyond 10⁻³⁰⁰ is a v2 goal requiring extended-precision `DeltaC` (see §8).

**Fractal families in scope (v1):** Mandelbrot / filled-Julia, Newton-method fractals over polynomial roots.

| Layer | Language / Target | Responsibility |
|---|---|---|
| Math Kernel | Rust → WASM | Perturbation theory, BLA, extended precision, escape-time, coloring LUT |
| WASM Bridge | wasm-bindgen | Worker-facing API, shared `WebAssembly.Memory` layout, panic marshaling |
| GPU Pipeline | WebGL 2 / GLSL | Tile upload, smooth coloring, orbit-trap compositing, export framebuffer |
| UI Shell | TypeScript | Canvas overlay, zoom/pan FSM, color editor, Worker orchestration, export |

---

## 2. Goals & Non-Goals

### Goals

- Correct deep-zoom rendering at ≥ 1 000 iterations at 10⁻³⁰⁰ depth (v1 cap; see §8)
- Smooth ≥ 60 fps pan / zoom via progressive tile streaming
- PNG / JPEG / EXR image export at user-defined resolution
- Interactive color palette editor (gradient stops, iteration-space mapping, orbit-trap overlay)
- Julia set live preview linked to Mandelbrot pointer position
- Newton fractal with configurable polynomial and root coloring
- Web Worker offload — main thread never blocked

### Non-Goals (v1)

- Zoom beyond 10⁻³⁰⁰ (planned v2 — `DeltaC` opaque type is the upgrade path)
- 3D fractal rendering (Mandelbox, Menger sponge)
- Server-side rendering / compute cluster offload
- Mobile touch (planned v2)
- WebGPU backend (planned v2 — architecture is abstraction-ready)

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser Main Thread                                            │
│  ┌──────────────┐   ┌──────────────────────────────────────┐    │
│  │ TypeScript   │   │ WebGL 2 Pipeline                     │    │
│  │ UI Shell     │──▶│ tileTexArray → accumFBO → screen     │    │
│  └──────┬───────┘   └──────────────────────────────────────┘    │
│         │ postMessage / shared WebAssembly.Memory               │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────┐
│  Web Workers                                                    │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Orbit Worker×1 │  │ Tile Workers×N  │  │ Scheduler ×1    │   │
│  │ BigFloat ref   │  │ Perturbation    │  │ Priority queue  │   │
│  │ orbit + BLA    │  │ loop + BLA      │  │ tile dispatch   │   │
│  └────────┬───────┘  └────────┬────────┘  └────────┬────────┘   │
└───────────┼───────────────────┼────────────────────┼────────────┘
            │                   │                    │
┌───────────▼───────────────────▼────────────────────▼────────────┐
│  Shared WebAssembly.Memory (created on main thread)             │
│  orbit · BLA · secondary orbits · slot states · tile ring       │
└─────────────────────────────────────────────────────────────────┘
            │                   │
┌───────────▼───────────────────▼──────────────────────────────┐
│  WASM Module (per worker instance, same shared Memory)       │
│  arith · kernel · coloring · scheduler                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Rust Crate Structure

```
fractal-workspace/
  Cargo.toml                  # [workspace]
  crates/
    arith/          # Extended-precision arithmetic (no_std)
    kernel/         # Fractal math: escape time, perturbation, BLA
    coloring/       # Coloring algorithms, LUT, orbit traps
    scheduler/      # Tile queue, priority, progressive refinement
    wasm-bridge/    # wasm-bindgen surface — thin API only
  web/
    src/            # TypeScript UI shell
    shaders/        # GLSL vertex + fragment
    pkg/            # wasm-pack output (gitignored)
```

### 4.1 `crate: arith`

Implements numeric types and the `Precision` trait for deep zoom. All types are `no_std + no alloc` — compiles identically on native and WASM targets.

**`arith` is the workspace-wide home for numeric primitives.** `Complex<T>` is defined here and used by every other crate — there is no parallel complex type in the codebase.

| Type / Trait | Notes |
|---|---|
| `Precision` | Trait implemented by `f64`, `F64x2`, `BigFloat<N>`. Provides `zero`, `one`, `from_f64`, `to_f64_lossy`, arithmetic ops, `norm_sqr`. Lets `kernel` write generic orbit code without naming a concrete precision type. |
| `Complex<T: Precision>` | `repr(C)` generic complex number. Used identically in orbit buffers, BLA entries, `EscapeResult`, and perturbation steps. `repr(C)` means `&[Complex<f64>]` can be cast directly from shared WASM memory orbit bytes with zero copy. |
| `F64x2` (double-double) | 106 mantissa bits. Zoom to ~10⁻²⁸. Uses Knuth TwoSum / Veltkamp splitting. |
| `BigFloat<N>` | Fixed-point in `[−8, 8]`, N × 64-bit limbs, carry-chain arithmetic. N chosen by `required_limbs(zoom_exp)`. Fixed-size stack array — no heap. **Not a general-purpose float** — the `[−8, 8]` range is sufficient for all Mandelbrot orbit values and intermediate `z² + c` arithmetic. To generalise to arbitrary reals, add an explicit `exponent: i32` field and adjust the carry-chain to account for it. |
| `DeltaC` | Opaque f64 pair in v1. Per-pixel relative coordinate. `as_complex_f64()` is the only sanctioned extraction — internals can upgrade to rescaled extended precision in v2 without touching any `kernel` call site. |

**Crate boundary rule:** `arith` owns numbers; `kernel` owns fractal algorithms; `wasm-bridge` owns precision dispatch. `kernel` imports only `arith::{Complex, DeltaC, Precision}`. `wasm-bridge` is the only crate that names a concrete `BigFloat<N>` — it holds the single `match required_limbs(zoom_exp)` block that selects N and calls the right `kernel` monomorphization.

### 4.2 `crate: kernel`

Contains the core iteration engines. Each fractal type implements a decomposed set of three traits (see [§10](#10-adding-new-fractal-types) for the full trait design).

`kernel` imports `arith::{Complex, DeltaC, Precision}`. `Complex<f64>` (from `arith`) is the currency for all f64 iteration arithmetic. `kernel` never names `BigFloat<N>` directly — reference orbit computation is generic over `T: Precision`, and `wasm-bridge` provides the concrete N at the call site.

Key `kernel` entry points:

```rust
// Phase 1: plain escape-time render for one tile
pub fn render_tile_escape<M: IterationMap>(map: &M, coords: &[Complex<f64>], max_iter: u32, out: &mut [TilePixel]);

// Phase 2: compute reference orbit at precision T (generic; wasm-bridge monomorphizes for N=2,4,8)
pub fn compute_ref_orbit<T: Precision, M: PerturbationSupport>(c: Complex<T>, max_iter: u32, out: &mut [Complex<f64>]) -> u32;

// Phase 2: per-pixel perturbation hot path
pub fn perturb_pixel<M: PerturbationSupport>(map: &M, orbit: &[Complex<f64>], bla: &[BlaEntry], dc: DeltaC, max_iter: u32) -> EscapeResult;

// Phase 2: build BLA table from f64 orbit (called once by Orbit Worker after compute_ref_orbit)
pub fn build_bla_table(orbit: &[Complex<f64>], out: &mut [BlaEntry]) -> usize;
```

#### 4.2.1 Mandelbrot / Julia Kernel

Standard `z ← z² + c` iteration. The perturbation equation for probe point `z = Z + δ`:

```
δₙ₊₁ = 2·Zₙ·δₙ + δₙ² + Δc
```

The reference orbit is computed once per zoom event at full `BigFloat<N>` precision via `compute_ref_orbit`, stored in the shared `WebAssembly.Memory` orbit region as `Complex<f64>` entries for the perturbation loop. Glitch detection follows the Pauldelbrot criterion: `|δ| > ε·|Z|`. Glitched pixels are re-seeded with a secondary reference orbit (see §9.4).

#### 4.2.2 Newton Fractal Kernel

For polynomial `p(z)`, Newton map is `N(z) = z − p(z) / p'(z)`. The kernel:

- Accepts `Vec<Complex<f64>>` polynomial coefficients
- Auto-differentiates `p'(z)` symbolically at startup
- Identifies roots via Durand-Kerner for root-index coloring
- Iterates until `|p(z)| < ε`, recording convergence speed and attracting root

> Newton fractals do not require perturbation theory — they are not deep-zoom targets. Standard f64 is sufficient.

#### 4.2.3 Bilinear Approximation (BLA)

Precomputes a table mapping `(iteration step, perturbation magnitude) → maximum skippable steps`. Built from the reference orbit in O(N log N) by the Orbit Worker immediately after orbit computation. Skips 80–99% of iterations in smooth interior regions; falls back to full perturbation near the boundary where detail is densest.

The BLA table lives in the shared `WebAssembly.Memory` BLA region alongside the reference orbit. Tile Workers read it directly — zero copy. See §9.3 for the correct merge formula.

### 4.3 `crate: coloring`

Coloring is separated from escape computation. The `EscapeResult` struct carries all raw orbit data:

```rust
pub struct EscapeResult {
    pub iter:          u32,
    pub escaped:       bool,
    pub smooth_t:      f64,        // fractional escape for smooth coloring
    pub orbit_min_r:   f64,        // orbit-trap: min |z| reached
    pub orbit_min_z:   Complex<f64>,
    pub angle_final:   f64,        // arg(z) at escape
}
```

| Algorithm | Formula | Notes |
|---|---|---|
| Smooth iteration | `iter + 1 − log₂(log\|z\|)` | Removes banding; maps to gradient via LUT |
| Orbit trap (circle) | `min \|zₙ\|` over orbit | Reveals internal structure; configurable trap shape |
| Angle (argument) | `arg(z)` at escape | Domain coloring flavor; good for Newton |
| Root index (Newton) | which root attracted orbit | Base hue per root; convergence speed = lightness |
| Stripe average | `∑ sin(arg(zₙ)) / n` | Psychedelic banding; computationally cheap |

The LUT is a 4096-element RGBA f32 array uploaded to a WebGL texture. The GPU fragment shader performs linear interpolation between LUT entries using `smooth_t`.

### 4.4 `crate: scheduler`

Manages the tile work queue dispatched to Workers. Priority ordering:

1. Viewport center tiles (highest)
2. Low-iteration pass (fast feedback — same 256×256 size, low `max_iter`)
3. Full-iteration detail
4. Prefetch tiles just outside viewport

```rust
pub struct TileJob {
    pub tile_x, tile_y:  i32,
    pub iter_level:      u8,        // 0 = low iter (64), 1 = medium (512), 2 = full
    pub max_iter:        u32,
    pub ref_orbit_id:    u64,       // which reference orbit slot to use
    pub priority:        f32,
}
```

Tiles are always 256 × 256 px. At 2 560 × 1 440, that is ~110 tiles for a full repaint. The Scheduler dispatches one job at a time to each Worker via a dedicated `MessageChannel` port (pull model — Workers request the next job on completion).

---

## 5. WASM Bridge

The bridge is the only crate that depends on `wasm-bindgen`. It exposes a small, stable API callable from Web Workers and handles all shared memory concerns at the boundary. **There is no WASM instance on the main thread** — `session.ts` is pure TypeScript.

### 5.1 Worker-Facing API

```rust
/// Called once per worker at startup to get byte offsets for all shared memory regions.
#[wasm_bindgen]
pub fn layout(n_workers: u32, max_iter: u32) -> MemoryLayout;

/// Orbit Worker: compute reference orbit + BLA table into shared memory.
/// Returns { orbit_len, bla_len } on completion.
///
/// wasm-bridge owns the single precision-dispatch match. It calls
/// kernel::compute_ref_orbit::<BigFloat<N>, M> for N = 2, 4, or 8
/// (selected by arith::required_limbs(zoom_exp)), then kernel::build_bla_table.
/// kernel is generic over T: Precision; wasm-bridge supplies the concrete N.
#[wasm_bindgen]
pub fn compute_reference_orbit(
    kind: FractalKind,
    cx: &str,           // BigDecimal string
    cy: &str,           // BigDecimal string
    zoom_exp: f64,      // log₁₀ of magnification
    max_iter: u32,
    orbit_slot: u32,    // 0 = primary, 1–3 = secondary
) -> OrbitResult;

/// Tile Workers: render one tile into the assigned ring slot.
#[wasm_bindgen]
pub fn render_tile(job: JsValue, layout: &MemoryLayout) -> TileResult;

/// Main thread (or any worker): build the LUT from palette parameters.
#[wasm_bindgen]
pub fn build_lut(palette: JsValue) -> Float32Array;
```

`FractalKind` is a `#[wasm_bindgen]` enum. `wasm-pack` automatically emits the corresponding TypeScript union type in the generated `.d.ts` — no custom codegen required.

### 5.2 Shared Memory Model

#### Target architecture (Phase 1+)

The target design uses a single `WebAssembly.Memory({ shared: true })` created on the main thread, structured-cloned to every worker, and injected as `importObject.env.memory` at instantiation. WASM linear memory *is* the shared buffer — tile data written by a Tile Worker is visible to the main thread's `gl.texSubImage2D` call with zero copy.

Producing a wasm-bindgen module that *imports* memory (rather than creating and exporting its own) requires recompiling the Rust standard library with `+atomics` — i.e. `-Z build-std` on nightly. Once the toolchain constraint is lifted this architecture is fully realised.

#### Phase 1 pragmatic model (current)

Until the shared-memory build is in place, Tile Workers use a single `SharedArrayBuffer` (`tileSab`) managed by the main thread. Each worker owns one slot — worker _k_ always writes to slot _k_ — so `RING_SLOTS = N_WORKERS`.

| Buffer | Size | Purpose |
|---|---|---|
| `tileSab` | `N_WORKERS × TILE_SLOT_BYTES` B | Pixel ring data; one slot per worker |

Workers spawn `N_WORKERS = max(2, min(8, hardwareConcurrency − 2))` instances. Each independently calls `alloc_tile_buf()` + `render_tile_to_ptr()` on its own WASM heap, copies the 1 MiB tile into its SAB slot, then signals via `postMessage`. Workers run truly in parallel — no shared WASM memory is needed since each owns its slot.

**One copy per tile (WASM heap → SAB) is accepted.** Phase 2 eliminates it once `-Z build-std` lands and wasm-bindgen can accept an injected `SharedArrayBuffer` as WASM linear memory.

#### Phase 2 orbit SAB

Phase 2 adds a second `SharedArrayBuffer` (`orbitSab`) for the reference orbit and BLA table. It is created on the main thread, sized by `layout()`, and structured-cloned to all workers at startup — the same pattern as `tileSab`. The Orbit Worker writes the orbit and BLA data into `orbitSab`; Tile Workers read from it. The `postMessage` from the Orbit Worker on completion acts as the memory fence. No nightly Rust toolchain is required.

When `-Z build-std` eventually lands, `orbitSab` merges into a single shared `WebAssembly.Memory` with no logic changes — the `layout()` offsets already describe the unified layout.

After each worker calls `layout(n_workers, max_iter)` it obtains byte offsets used for shared orbit and BLA regions (Phase 2+).

#### Memory Map (in order, from byte 0)

| Region | Size | Notes |
|---|---|---|
| **Primary orbit** | 8 B header + `MAX_ITER × 32` B | Header: `{ entry_bytes: u32, orbit_len: u32 }`. Max entry = 32 B (`(Complex<f64>, Complex<f64>)` for Phoenix) |
| **BLA table** | `MAX_ITER × 48` B | `BlaEntry` = `{ a, b: Complex<f64>, skip: u32, _pad: u32, valid_radius: f64 }` — `repr(C)` with 4 B padding before `valid_radius` for f64 alignment |
| **Secondary orbit ×3** | 3 × (8 B header + `MAX_ITER × 16` B) | f64 perturbation-speed orbits; no BigFloat |
| **Slot state array** | `4 × MAX_WORKERS × 4` B | `Int32` per slot: `EMPTY=0 / WRITING=1 / READY=2` |
| **Tile ring** | `4 × MAX_WORKERS × 256 × 256 × 4` B | RGBA f32 per pixel; 4× slots per worker absorbs burst latency |

**Approximate totals** (100 K iter, 8 Tile Workers):
- Primary orbit: ~3.2 MB · BLA: ~4.8 MB · Secondary orbits: ~4.8 MB · Tile ring: ~32 MB
- **Total: ~44.8 MB**

#### Tile slot lifecycle

Workers acquire a ring slot via `Atomics.compareExchange(stateArray, slotIndex, EMPTY, WRITING)` before filling it. On completion the slot is marked `READY` and the worker posts `{ tileId, slotIndex, ref_orbit_id }` to the Scheduler. The main thread marks the slot `EMPTY` after `gl.texSubImage2D` completes.

#### In-flight tile invalidation

When the reference orbit changes (`ref_orbit_id` increments):
1. The Scheduler stops dispatching jobs with the old ID.
2. Each worker checks `Atomics.load(currentOrbitId, 0)` against `job.ref_orbit_id` **before starting** a new tile — stale jobs are discarded immediately and the next job is requested.
3. Completions that arrive with a stale `ref_orbit_id` are discarded by the Scheduler; the ring slot is marked `EMPTY` without GPU upload.

At most one in-progress tile per worker is wasted on orbit switch (~50 ms worst case). No branch is added to the hot perturbation loop.

### 5.3 Build

Shared `WebAssembly.Memory` requires the module to be compiled with atomics support. Add to `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+atomics,+bulk-memory,+mutable-globals"]
```

`wasm-pack` must be invoked with `--no-pack` and `wasm-opt` run separately with `--enable-threads` to preserve atomics:

```sh
wasm-pack build crates/wasm-bridge \
  --target web \
  --no-pack \
  --out-dir ../../web/pkg \
  -- --features simd

wasm-opt web/pkg/wasm_bridge_bg.wasm \
  -O3 --enable-threads --enable-simd \
  -o web/pkg/wasm_bridge_bg.wasm
```

Because `wasm-bindgen`'s generated initializer creates its own `Memory`, it must be bypassed. Workers instantiate manually:

```js
const { instance } = await WebAssembly.instantiateStreaming(
  fetch(wasmUrl),
  { env: { memory } }   // inject the shared Memory
);
```

The `simd` feature activates `wasm-simd128` intrinsics for `F64x2` hot loops (2–4× speedup on Chromium and Firefox).

---

## 6. WebGL 2 Rendering Pipeline

### 6.1 Framebuffer Architecture

| FBO | Format | Purpose |
|---|---|---|
| `tileTexArray` | RGBA32F (2D Array) | 256-slot texture array; each slot = one tile |
| `accumFBO` | RGBA32F | Full-resolution accumulation; smooth-color pass composites tiles here |
| `lut1D` | RGBA32F (1D) | 4 096-element LUT; rebuilt when palette changes |
| `exportFBO` | RGBA16F | Off-screen render target for high-res export |

### 6.2 Render Passes

**Pass 1 — Tile Upload**

When the Scheduler forwards a `READY` tile, the main thread calls `gl.texSubImage3D` to write the raw iteration-count tile from the shared `WebAssembly.Memory` ring slot into the texture array layer. Raw data only — no color mapping here. Channels: `r = smooth_t`, `g = orbit_min_r`, `b = angle`, `a = escaped flag`.

**Pass 2 — Smooth-Color Accumulation**

Fullscreen quad reads from `tileTexArray` and `lut1D`:

```glsl
float t = mod(smooth_t * palette_speed + palette_offset, 1.0);
vec4 color = texture(u_lut, t);

// orbit trap blend
float trap = 1.0 - smoothstep(0.0, trap_radius, orbit_min_r);
color = mix(color, trap_color, trap_strength * trap);

fragColor = color;
```

**Pass 3 — Post-Process & Tonemap**

Second fullscreen pass over `accumFBO`:
- Brightness / contrast / saturation via ACES-approximate tonemap
- Optional Sobel edge-detection overlay (highlights set boundary)
- Gamma correction (sRGB output)

**Pass 4 — Export (on demand)**

Re-renders at target resolution into `exportFBO`. Pixel readback uses `PIXEL_PACK_BUFFER` with `gl.fenceSync` + `requestAnimationFrame` polling — the main thread stays responsive between strips. For exports > `gl.MAX_RENDERBUFFER_SIZE`, renders in strips stitched on an `OffscreenCanvas`. See §12.

### 6.3 Progressive Refinement

All tiles are always 256 × 256 px. "Coarse" means low `max_iter`, not smaller dimensions.

| Time | What the user sees |
|---|---|
| t = 0 ms | Zoom transform applied to `accumFBO` — GPU scale, instant but blurry |
| t = 0–50 ms | `iter_level=0` tiles (256 px, 64 iter) stream in — full resolution, iteration banding visible |
| t = 50–300 ms | `iter_level=1` tiles (256 px, 512 iter) replace them |
| t = 300 ms+ | `iter_level=2` tiles (256 px, max iter) fill center-out |

The GPU scale step reuses the previous `accumFBO` so the user sees something coherent within one frame rather than a blank screen.

---

## 7. TypeScript UI Shell

### 7.1 Module Map

| Module | Key exports | Responsibility |
|---|---|---|
| `session.ts` | `FractalSession` | Pure TypeScript orchestrator — no WASM instance; manages ref orbit lifecycle; routes commands between Orbit Worker, Scheduler, and GL pipeline |
| `viewport.ts` | `ViewState`, `ZoomPanFSM` | Coordinate transforms; FSM for pointer / wheel input |
| `workers.ts` | `WorkerPool` | Spawns N workers; manages `MessageChannel` ports; collects tile completions |
| `gl-pipeline.ts` | `GlPipeline` | WebGL2 context; FBO setup; all render passes; reads tile data from shared memory via `MemoryLayout` |
| `color-editor.ts` | `PaletteEditor` | Gradient stop UI; LUT generation; palette preset management |
| `export.ts` | `ExportDialog` | Resolution picker; format selector; export FBO orchestration; PBO readback |
| `ui-overlay.ts` | `OverlayController` | Coordinates panel, info bar, keyboard shortcuts, Julia preview |

### 7.2 Input Handling — ZoomPanFSM

| State | Triggers in | Triggers out | Action |
|---|---|---|---|
| `IDLE` | App start | pointerdown, wheel | No-op; renders accumFBO |
| `PANNING` | pointerdown + move | pointerup, escape | Translate view matrix; re-dispatch tiles when velocity drops |
| `ZOOMING` | wheel event | 300 ms debounce timeout | Exponential zoom; recompute ref orbit on settle |
| `ANIMATING` | double-click, bookmark jump | animation end | Lerp to target; pause tile dispatch until done |

View coordinate is always `(cx: BigDecimal string, cy: BigDecimal string, zoom_exp: f64)` where `zoom_exp` is the log₁₀ of the magnification factor. This avoids JS float precision loss in URL serialization and correctly handles the full v1 zoom range.

### 7.3 Color Editor

- Gradient strip with draggable colour stops (up to 32)
- Per-stop HSL / hex editor popover
- Iteration-space mapping curve (linear / log / power)
- Orbit-trap strength and radius sliders
- Palette speed (gradient repetitions per decade of iteration count)
- Export / import palette as JSON; bundled preset library

### 7.4 Keyboard Shortcuts & URL State

| Key / URL param | Effect |
|---|---|
| `+` / `−` / scroll | Zoom in / out at cursor |
| Space + drag | Pan |
| `R` | Reset to default view |
| `J` | Toggle Julia preview pane |
| `E` | Open export dialog |
| `C` | Toggle colour editor |
| `#cx=…&cy=…&z=…&f=…&p=…` | Full bookmarkable / shareable view state (`z` = `zoom_exp` as decimal string) |

---

## 8. Extended Precision & Deep Zoom

Standard `f64` gives ~15–16 significant decimal digits. The strategy at each depth tier:

| Zoom depth | Precision mode | Rust type | Approx. cost |
|---|---|---|---|
| zoom_exp < 10 | Native f64 | `Complex<f64>` | Baseline |
| zoom_exp 10–20 | Double-double | `Complex<F64x2>` | ~3× f64 |
| zoom_exp 20–300 | Perturbation + BigFloat<N> | `BigFloat<N>` ref, `DeltaC` probe | Ref once; probes at f64 speed |
| zoom_exp > 300 | v2: rescaled `DeltaC` | extended `DeltaC` internals | `DeltaC` opaque type is the upgrade path |

Precision tier is selected automatically at runtime. `zoom_exp` is the log₁₀ of the magnification (e.g. `zoom_exp = 100` for 10⁻¹⁰⁰ depth):

```rust
fn required_limbs(zoom_exp: f64) -> usize {
    let digits_needed = zoom_exp + 8.0; // 8-digit safety margin
    ((digits_needed / 15.9).ceil() as usize).max(2)
}
```

**v1 zoom cap:** `zoom_exp` is clamped to 300 in the UI and asserted in `required_limbs`. At `zoom_exp = 300`, per-pixel `DeltaC` values are ~10⁻³⁰³ — safely within normal f64 range (~2.2×10⁻³⁰⁸). Beyond that, f64 `DeltaC` underflows silently. The `DeltaC` type is opaque at all perturbation call sites so the internals can be upgraded to a rescaled representation in v2 without changing callers.

### 8.1 Reference Orbit Lifecycle

1. User stops scrolling → 300 ms debounce fires in `ZoomPanFSM`
2. `session.ts` posts `{ type: 'compute_orbit', cx, cy, zoom_exp, max_iter }` to the Orbit Worker
3. Orbit Worker's WASM computes the `BigFloat<N>` reference orbit into the primary orbit slot of shared memory, then immediately builds the BLA table into the BLA region
4. Orbit Worker posts `{ type: 'orbit_ready', ref_orbit_id, orbit_len, bla_len }` — this `postMessage` acts as the memory fence guaranteeing all shared memory writes are visible
5. `session.ts` forwards the signal to the Scheduler; Scheduler increments `ref_orbit_id`, invalidates the tile queue, re-queues with the new ID
6. Glitch detection runs per-tile; glitched pixel clusters trigger secondary reference orbit computation (see §9.4)

---

## 9. Perturbation Theory

### 9.1 Core Equation

Every pixel `C = C_ref + ΔC`. The perturbation `δₙ = Zₙ − Z_ref,n` evolves as:

```
δₙ₊₁ = 2·Z_ref,n·δₙ + δₙ² + ΔC
```

`Z_ref,n` is a cheap `Complex<f64>` lookup per iteration. `δₙ` and `ΔC` remain in `f64` range up to `zoom_exp = 300`. The BigFloat cost is paid once; all pixels run f64 perturbation.

### 9.2 Implementation

// EscapeResult is an enum — three mutually exclusive outcomes:
//
//   Escaped  { iter, smooth_t, orbit_min_r, orbit_min_z, angle_final }
//   Interior { orbit_min_r, orbit_min_z }
//   Glitched
//
// orbit_min_r / orbit_min_z are tracked for both Escaped and Interior so
// the coloring crate can apply orbit-trap overlays to all pixels.
// From<EscapeResult> for TilePixel panics on Glitched — glitched pixels
// are re-rendered via a secondary orbit before GPU upload.

```rust
pub fn perturb_pixel(
    ref_orbit: &[Complex<f64>],
    delta_c: DeltaC,       // opaque type; f64 pair in v1
    max_iter: u32,
) -> EscapeResult {
    let mut delta = Complex::new(0.0, 0.0);
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = Complex::<f64>::zero();

    for (n, &z_ref) in ref_orbit.iter().enumerate() {
        delta = 2.0 * z_ref * delta + delta * delta + delta_c.as_complex_f64();

        let z_approx = z_ref + delta;

        // Track orbit minimum for orbit-trap coloring.
        let r = z_approx.norm_sqr().sqrt();
        if r < orbit_min_r { orbit_min_r = r; orbit_min_z = z_approx; }

        // Glitch detection (Pauldelbrot criterion: |δ| > 1e-3 · |Z_ref|,
        // equivalently |δ|² > 1e-6 · |Z_ref|²).
        if delta.norm_sqr() > 1e-6 * z_ref.norm_sqr() {
            return EscapeResult::Glitched;
        }

        // Escape test on approximated full orbit.
        if z_approx.norm_sqr() > 4.0 {
            let smooth_t = (n as f64 + 1.0)
                - (z_approx.norm_sqr().sqrt().ln().ln() / core::f64::consts::LN_2);
            return EscapeResult::Escaped {
                iter: n as u32, smooth_t, orbit_min_r, orbit_min_z,
                angle_final: z_approx.im.atan2(z_approx.re),
            };
        }

        if n == ref_orbit.len() - 1 {
            return EscapeResult::Interior { orbit_min_r, orbit_min_z };
        }
    }
    EscapeResult::Interior { orbit_min_r, orbit_min_z }
}
```

### 9.3 BLA (Bilinear Approximation)

In smooth regions, `δₙ₊ₖ ≈ A·δₙ + B·ΔC`. The BLA table precomputes valid `(A, B, skip, valid_radius)` entries for each starting iteration using Zhuoran's level-doubling algorithm.

```rust
pub struct BlaEntry {
    pub a: Complex<f64>,
    pub b: Complex<f64>,
    pub skip: u32,
    pub valid_radius: f64,
}
```

**Merge formula (level-doubling):** When merging two consecutive entries `(A₀, B₀, skip₀, r₀)` and `(A₁, B₁, skip₁, r₁)`:

```
A_merged = A₁ · A₀
B_merged = A₁ · B₀ + B₁
skip_merged = skip₀ + skip₁
valid_radius_merged = min(r₀,  r₁ / |A₀|)   ← divide by |A₀|; omitting this is a common bug
```

The `/ |A₀|` factor accounts for perturbation amplification through the first half-skip. Using `min(r₀, r₁)` instead produces subtly wrong escape counts in smooth regions, visible as faint false banding. Validate BLA output against a full-perturbation reference render before shipping Phase 2.

BLA operates entirely on f64 — it is built from the already-downcast `Vec<Complex<f64>>` reference orbit and never touches `BigFloat`. The BLA table lives in the shared `WebAssembly.Memory` BLA region; Tile Workers read it with zero copy via the byte offset from `layout()`.

Pixel loop with BLA:

```rust
let mut n = 0;
while n < ref_orbit.len() {
    let entry = &bla_table[n];
    if delta.norm() < entry.valid_radius {
        // Skip k iterations in one step
        delta = entry.a * delta + entry.b * delta_c.as_complex_f64();
        n += entry.skip as usize;
    } else {
        // Full perturbation step
        delta = 2.0 * ref_orbit[n] * delta + delta * delta + delta_c.as_complex_f64();
        n += 1;
    }
}
```

### 9.4 Glitch Correction

Glitched pixels (where `|δ| > threshold · |Z_ref|`) are flagged in the tile result. The Scheduler collects glitch clusters and issues secondary `TileJob`s using a secondary orbit slot.

Secondary reference orbits are computed at **f64 perturbation speed** — not BigFloat. The secondary reference point is close to the primary in screen space, so its orbit can be computed by running the perturbation equations forward from the primary reference. This is fast enough that 1–3 secondary orbits per frame add negligible latency.

Three secondary orbit slots are reserved in shared memory (slots 1–3 of the orbit region). Each holds an 8-byte header plus `MAX_ITER × 16` bytes (single `Complex<f64>` entry per step — secondary orbits always use the simple Mandelbrot perturbation form). The `ref_orbit_id` field in `TileJob` selects which slot to use transparently.

### 9.5 Reference Point Selection

- **v1:** Screen center for the primary reference, with secondary orbit retry on glitch clusters
- **v2 candidate:** Multiple references per frame — one main, secondary references at glitch cluster centroids (Kalles Fraktaler approach)

The `ref_orbit_id` field in `TileJob` already supports transparent reference swapping without scheduler changes.

---

## 10. Adding New Fractal Types

### 10.1 The Three Axes of Variation

Adding a new fractal type touches three independent axes. Conflating them is the main design trap:

| Axis | Question | Examples |
|---|---|---|
| **Iteration map** | What is the recurrence? | `z² + c`, `zⁿ + c`, `sin(z) + c`, Newton map |
| **Perturbation eligibility** | Does it need deep zoom? | Mandelbrot: yes. Newton: no. Burning Ship: yes (modified eq) |
| **Coloring domain** | What orbit data is meaningful? | Escape time, convergence speed, root index, Lyapunov exponent |

### 10.2 Trait Decomposition

Rather than one monolithic `FractalKernel`, three focused traits. All `Complex<f64>` references below are `arith::Complex<f64>` — the single workspace-wide complex type.

```rust
/// Axis 1: The iteration map itself.
pub trait IterationMap: Send + Sync {
    fn step(&self, z: Complex<f64>, c: Complex<f64>) -> StepResult;
    fn escape_radius_sq(&self) -> f64;
    fn converged(&self, z: Complex<f64>, z_prev: Complex<f64>) -> bool {
        false // default: escape-time maps don't converge
    }
}

/// Axis 2: Optional perturbation support.
/// Only implemented by maps that need deep zoom.
pub trait PerturbationSupport: IterationMap {
    /// Per-step reference state (e.g. sign bits for Burning Ship).
    type RefState: Copy + Default;

    /// The type of one entry in the reference orbit buffer.
    /// Complex<f64> for most maps; (Complex<f64>, Complex<f64>) for Phoenix.
    type RefOrbitEntry: Copy + Default;

    fn ref_state(&self, z_ref: Complex<f64>) -> Self::RefState;

    fn perturb_step(
        &self,
        z_ref: Complex<f64>,
        state: Self::RefState,
        delta: Complex<f64>,
        delta_c: DeltaC,
    ) -> Complex<f64>;

    fn glitch_threshold(&self) -> f64 { 1e-3 }
}

/// Axis 3: What raw data the orbit emits for coloring.
pub trait OrbitData: Default {
    fn accumulate(&mut self, step: &StepResult, n: u32);
}
```

The `RefOrbitEntry` associated type on `PerturbationSupport` determines the stride of orbit entries in shared memory. The 8-byte header at the start of each orbit slot records the actual entry size, allowing Tile Workers to locate entries correctly regardless of fractal type. Default `RefOrbitEntry = Complex<f64>` (16 bytes) for Mandelbrot/Multibrot; `(Complex<f64>, Complex<f64>)` (32 bytes) for Phoenix.

### 10.3 The Fractal Catalog

| Fractal | Perturbation? | Notes |
|---|---|---|
| Mandelbrot z²+c | Yes | Reference implementation |
| Multibrot zⁿ+c | Yes | Same perturbation eq; different exponent |
| Burning Ship | Yes (modified) | Abs-value folds change the perturbation equation; `RefState` must expose sign bits |
| Tricorn / Mandelbar | Partial | Conjugate complicates BLA |
| Nova / Nova Mandelbrot | No | Newton map + additive `c`; root-index coloring |
| Newton | No | Polynomial roots; f64 sufficient |
| Lyapunov | No | Sequence of logistic maps; Lyapunov exponent coloring |
| Magnet I / II | No | Rational map |
| Phoenix | Yes (2-step) | `zₙ₊₁ = zₙ² + Re(c) + p·zₙ₋₁`; `RefOrbitEntry = (Complex<f64>, Complex<f64>)` |

#### Burning Ship perturbation equation

The absolute values create fold discontinuities. The modified equation tracks sign state at the reference:

```
δₙ₊₁ = 2·(sign(Re Z_ref,n)·Re δₙ + i·sign(Im Z_ref,n)·Im δₙ)·Z_ref,n + δₙ² + ΔC
```

`RefState` for Burning Ship = `{ sign_re: f64, sign_im: f64 }`.

#### Phoenix two-step recurrence

```
δₙ₊₁ = 2·Z_ref,n·δₙ + p·δₙ₋₁ + δₙ² + ΔC
```

`RefOrbitEntry = (Complex<f64>, Complex<f64>)` — the orbit header records entry size 32; Tile Workers read 32 bytes per step.

### 10.4 Newton / Convergent Coloring Data

```rust
#[derive(Default)]
pub struct NewtonOrbitData {
    pub converged_to:     Option<usize>,  // root index
    pub convergence_iter: u32,
    pub final_z:          Complex<f64>,
}

impl OrbitData for NewtonOrbitData {
    fn accumulate(&mut self, step: &StepResult, n: u32) {
        self.convergence_iter = n;
        self.final_z = step.z;
    }
}
```

### 10.5 Registration

New types are registered via a compile-time macro that generates the `match` dispatch in `FractalSession::new()` and the JS enum variants. The TypeScript union type is generated automatically by `wasm-bindgen` from the `#[wasm_bindgen]` enum — no custom codegen step:

```rust
fractal_registry! {
    Mandelbrot  => MandelbrotMap  + MandelbrotPerturbation + SmoothIterData,
    BurningShip => BurningShipMap + BurningShipPerturbation + SmoothIterData,
    Newton      => NewtonMap      + NoPerturbation          + NewtonOrbitData,
    Phoenix     => PhoenixMap     + PhoenixPerturbation     + SmoothIterData,
}
```

Adding a new fractal type is: one registry line + trait implementations for the three axes. The scheduler, shared memory model, and GL pipeline are unaffected.

---

## 11. Web Worker Topology

| Worker | Count | Owns | Communicates via |
|---|---|---|---|
| Orbit Worker | 1 | Reference orbit + BLA computation (BigFloat) | postMessage to `session.ts` |
| Tile Workers | `hardwareConcurrency − 2` (min 2) | Perturbation loop, BLA, pixel emission | Dedicated `MessageChannel` port per worker (pull model); atomic slot states |
| Scheduler Worker | 1 | TileJob priority queue; dispatches to Tile Workers | `MessageChannel` from `session.ts` for viewport changes; per-worker ports for job dispatch |

**Shared memory:** All workers receive the same `WebAssembly.Memory` object via structured clone at startup. Each worker instantiates its own WASM module instance with this shared memory — the compiled `WebAssembly.Module` is transferred (zero-copy) at spawn; instantiation costs < 2 ms.

**Pull dispatch:** When a Tile Worker completes a tile, it posts `{ tileId, slotIndex, ref_orbit_id }` to the Scheduler on its dedicated `MessageChannel` port and waits. The Scheduler responds with the next highest-priority `TileJob`. This keeps priority ordering exact and the Scheduler latency negligible.

**Orbit + BLA atomicity:** The Orbit Worker writes the orbit and BLA table, then sends a single `postMessage` to `session.ts`. The `postMessage` acts as the memory fence — all shared memory writes before it are visible to Tile Workers after they receive the forwarded signal from the Scheduler.

---

## 12. Image Export Pipeline

### 12.1 Formats

| Format | Library | Use case |
|---|---|---|
| PNG (8 or 16 bit) | `canvas.toBlob("image/png")` | Default; lossless |
| JPEG | `canvas.toBlob("image/jpeg", quality)` | Small file; lossy |
| EXR (32-bit float) | `openexr-wasm` (Rust → WASM) | Full HDR; preserves raw `smooth_t` for compositing |

### 12.2 Large Export Strategy

Export pauses live viewport rendering and dedicates the full worker pool to export tiles. The Scheduler enters `ExportMode`, draining in-flight viewport tiles and blocking new viewport dispatch until export completes.

For exports > `gl.MAX_RENDERBUFFER_SIZE` (~16 384 px), the image is rendered in horizontal strips:

1. Subdivide view frustum into bands
2. Render each band into `exportFBO`
3. Initiate async pixel readback via `PIXEL_PACK_BUFFER`:
   ```js
   gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
   gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, 0);
   const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
   // poll in requestAnimationFrame until gl.getSyncParameter returns SIGNALED
   ```
4. On sync signal: read PBO data, write strip to `OffscreenCanvas` accumulator
5. Repeat for next strip; encode stitched canvas to target format on completion

PBO readback keeps the main thread responsive between strips — the progress bar updates and the cancel button remains functional throughout.

EXR export bypasses the PNG/JPEG quantization step — the 32-bit `accumFBO` data is encoded directly via the Rust/WASM encoder.

---

## 13. Build & Dev Tooling

| Tool | Version | Role |
|---|---|---|
| Rust toolchain | stable (1.78+) | Core compiler; WASM target via `rustup` |
| wasm-pack | 0.12+ | Builds and bundles WASM crate; manual `wasm-opt` pass required (see §5.3) |
| wasm-opt | (Binaryen) | Dead code elimination, SIMD lowering, atomics preservation |
| Vite | 5+ | Dev server with COOP/COEP headers; HMR for TS/GLSL; rollup bundle |
| vitest | 1+ | Unit tests for TS modules; WASM integration tests via browser mode |
| Playwright | 1.44+ | E2E: zoom to known coordinates and pixel-match golden renders |
| cargo-criterion | latest | Micro-benchmarks for arith / kernel hot paths; CI regression gating |

### GLSL Hot Reload

Vite's `import.meta.glob` imports shader source as strings. A Vite plugin watches `shaders/` and triggers a `gl-pipeline` re-initialise (WebGL context preserved) on change — no full page reload when tweaking the coloring pass.

### Dual WASM Bundle

Vite produces two bundles: SIMD and non-SIMD. A synchronous feature probe runs before any network request:

```js
function detectSimd(): boolean {
  // Minimal WASM module with one v128.const instruction
  return WebAssembly.validate(new Uint8Array([
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,
    2,1,0,10,10,1,8,0,65,0,253,15,253,98,11
  ]));
}

const wasmUrl = detectSimd() ? '/pkg/wasm_bridge_simd.wasm' : '/pkg/wasm_bridge.wasm';
```

This runs before `fetch`, so non-SIMD browsers never request the SIMD bundle. Ships only `N=2,4,8` `BigFloat` variants to limit monomorphisation bloat; dynamic dispatch for `N > 8`.

---

## 14. Testing Strategy

| Layer | Test type | Tool | Key assertions |
|---|---|---|---|
| `arith` | Unit + property | `cargo test` + `proptest` | `F64x2` / `BigFloat` arithmetic matches mpfr reference to full precision |
| `kernel` | Unit + golden | `cargo-criterion` | Escape counts for known coordinates match published values; BLA skips ≥ 80%; BLA output matches full-perturbation render pixel-for-pixel |
| `wasm-bridge` | Integration | `wasm-pack test --chrome` | JS API round-trips; shared memory writes correct data; `layout()` offsets consistent with actual writes |
| GL pipeline | Screenshot regression | Playwright + pixelmatch | `|Δpixel| < 2 LSB` on 8-bit for canonical zoom coordinates |
| UI shell | Component + E2E | vitest + Playwright | FSM state transitions; export produces correct file size / format |

---

## 15. Performance Targets

| Metric | Target | Measurement |
|---|---|---|
| Pan frame time (GPU only) | < 4 ms | `gl.EXT_disjoint_timer_query` |
| First tile visible after zoom settle | < 100 ms | `performance.mark / measure` |
| Full-res repaint (1440p, 1000 iter) | < 2 s wall clock | Worker pool completion event |
| Reference orbit (10 000 iter, F64x2) | < 50 ms | cargo-criterion |
| Reference orbit (10 000 iter, BigFloat<4>) | < 500 ms | cargo-criterion |
| 4K PNG export (3840 × 2160) | < 10 s | export.ts timing |

---

## 16. Phased Roadmap

| Phase | Deliverable | Crates / modules in scope |
|---|---|---|
| 0 — Scaffold | Workspace, CI, shared-memory WASM hello-world renders to canvas | `wasm-bridge` (stub), shared memory init |
| 1 — Mandelbrot f64 | Basic escape-time Mandelbrot, smooth coloring, pan/zoom | `kernel`, `coloring`, `gl-pipeline`, `viewport.ts` |
| 2 — Deep Zoom | F64x2 / BigFloat; perturbation theory; BLA (with validation); glitch correction | `arith`, `kernel` (perturb), `scheduler` |
| 3 — Newton | Newton fractal with polynomial config UI and root-index coloring | `kernel` (newton), `coloring` (root) |
| 4 — Color Editor | Full palette editor, orbit trap, LUT, preset library | `coloring`, `color-editor.ts` |
| 5 — Export | PNG / JPEG / EXR export pipeline; PBO readback; large-image stitching | `export.ts`, `openexr-wasm` |
| 6 — Polish | Playwright golden tests, perf regression CI, URL bookmarks, Julia preview | All |

---

## 17. Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| COOP/COEP headers block third-party embeds | Medium | Service worker fetch proxy for cross-origin assets |
| Glitch detection misses subtle artifacts at extreme zoom | Medium | Per-tile checksum of escape counts; automatic secondary orbit retry |
| Safari: SharedArrayBuffer requires HTTPS | Low (dev only) | Self-signed cert for local dev; production HTTPS already required |
| `BigFloat<N>` WASM module size bloat | Medium | Feature-gate N variants; ship N=2,4,8 only; dynamic dispatch for N>8 |
| wasm-simd128 on older browsers | Low | Dual-bundle; synchronous `WebAssembly.validate()` probe selects correct bundle before fetch |
| BLA valid_radius merge bug produces false banding | Medium | Mandatory pixel-exact BLA vs full-perturbation validation test in Phase 2 CI |
| f64 `DeltaC` underflow beyond zoom_exp = 300 | Low (v1) | Hard cap at 300 in UI and `required_limbs`; `DeltaC` opaque type is the v2 upgrade path |
| Shared `WebAssembly.Memory` maximum too small | Low | Calculate MAX_PAGES from `MAX_ITER` and `MAX_WORKERS` at startup; assert before creation |

---

## 18. References

- Pauldelbrot — "Superfractalthing Arbitrary Size Mandelbrot Renderer" (perturbation theory, BLA)
- Zhuoran — BLA table construction algorithm (fractalforums.org, 2021)
- Hida, Li, Bailey — "Quad-Double Arithmetic: Algorithms, Implementation, and Application" (LBNL 2000)
- Shewchuk — "Adaptive Precision Floating-Point Arithmetic" (1997)
- wasm-bindgen Reference — <https://rustwasm.github.io/docs/wasm-bindgen/>
- WebGL 2 Specification — <https://registry.khronos.org/webgl/specs/latest/2.0/>
- wasm-pack Book — <https://rustwasm.github.io/docs/wasm-pack/>
- ACES Filmic Tone Mapping — Narkowicz & Evangelista (2015)
- WebAssembly Threads proposal — <https://github.com/WebAssembly/threads>
