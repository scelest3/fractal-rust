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

- Correct deep-zoom rendering to zoom_exp ≈ 20 via F64x2 double-double arithmetic (v1 cap; see §8)
- Smooth ≥ 60 fps pan / zoom via progressive tile streaming
- PNG image export at user-defined resolution (up to 16K × 16K), 16-bit RGB, with DPI metadata; EXR is v2
- Interactive color palette editor (gradient stops, iteration-space mapping, orbit-trap overlay)
- Julia set as a new fractal type with configurable parameter `c`; Mandelbrot hover preview (pointer position drives Julia `c`) follows as a second step
- Newton fractal with configurable polynomial and root coloring
- Web Worker offload — main thread never blocked

### Non-Goals (v1)

- Zoom beyond zoom_exp ≈ 20 (v2 — perturbation theory + BigFloat required; see §8)
- Perturbation theory, BLA, and glitch correction (v2 — architecture is prepared but deferred)
- Zoom beyond 10⁻³⁰⁰ (planned v2 — `DeltaC` opaque type is the upgrade path)
- 3D fractal rendering (Mandelbox, Menger sponge)
- Server-side rendering / compute cluster offload
- Mobile touch (planned v2)
- WebGPU backend (planned v2 — architecture is abstraction-ready)
- EXR export (v2 — `exportFBO` is RGBA32F so no pipeline change is required when added; see §12)
- Complex polynomial coefficients for Newton (v2 — interesting in the context of Julia/Mandelbrot variants)
- Transcendental Newton fractals (v2 — `sin(z)`, `exp(z)`, etc. require a different solver)
- Per-root hue pickers and root re-ordering UI for Newton (v2 — auto-assigned hues ship in v1)

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
    coloring/       # color_pixel — three coloring paths (escaped, interior, Newton)
    encoding/       # encode_png — RGB16 PNG with pHYs, sRGB, tEXt metadata
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
pub fn render_tile_escape<M: IterationMap>(
    map: &M, coords: &[Complex<f64>], max_iter: u32, out: &mut [TilePixel]);

// Phase 2: compute reference orbit at precision T (generic; wasm-bridge monomorphises for N=2,4,8).
// Returns the number of non-escaped entries written to `out`.
// Terminates early if the reference point itself escapes before max_iter.
pub fn compute_ref_orbit<T: Precision, M: PerturbationSupport>(
    map: &M, c: Complex<T>, max_iter: u32, out: &mut [Complex<f64>]) -> usize;

// Phase 2: per-pixel perturbation hot path (no BLA; each step is a direct f64 recurrence).
// Returns Glitched when |δz_n|² > threshold² · |Z_n|².
pub fn perturb_pixel<M: PerturbationSupport>(
    map: &M, ref_orbit: &[Complex<f64>], delta_c: DeltaC, max_iter: u32) -> EscapeResult;

// Phase 2: BLA-accelerated perturbation. ref_orbit.len() must equal bla_table.len().
// Falls back to direct perturbation when |δz_n| ≥ bla_table[n].valid_radius.
pub fn perturb_pixel_bla<M: PerturbationSupport>(
    map: &M, ref_orbit: &[Complex<f64>], bla_table: &[BlaEntry],
    delta_c: DeltaC, max_iter: u32) -> EscapeResult;

// Phase 2: build BLA table from f64 orbit (called once by Orbit Worker after compute_ref_orbit).
// Returns one BlaEntry per orbit position; valid_radius = 0 means no BLA applicable there.
pub fn build_bla_table(orbit: &[Complex<f64>]) -> Vec<BlaEntry>;
```

#### 4.2.1 Mandelbrot / Julia Kernel

Standard `z ← z² + c` iteration. The perturbation equation for probe point `z = Z + δ`:

```
δₙ₊₁ = 2·Zₙ·δₙ + δₙ² + Δc
```

The reference orbit is computed once per zoom event at full `BigFloat<N>` precision via `compute_ref_orbit`, stored in the shared `WebAssembly.Memory` orbit region as `Complex<f64>` entries for the perturbation loop. Glitch detection follows the Pauldelbrot criterion: `|δ| > ε·|Z|`. Glitched pixels are re-seeded with a secondary reference orbit (see §9.4).

#### 4.2.2 Newton Fractal Kernel

For polynomial `p(z)`, Newton map is `N(z) = z − p(z) / p'(z)`.

**Polynomial representation.** Real coefficients only; `coeffs[k]` is the coefficient of `z^k`. Represented as a fixed `[f64; 11]` stack array with a `degree: usize` live-count field. Maximum degree 10. No heap allocation — consistent with the `arith` no-alloc constraint. Complex coefficients and transcendental Newton are v2.

**Derivative.** `p′(z)` computed by exact degree-reduction once at polynomial setup (not per-iteration).

**Convergence criterion.** `|p(z)| < ε` where `ε = 1e-6`. Evaluated from the same Horner pass that computes the Newton step — no extra cost per iteration.

**Divergence bailout.** `|z| > 1e8`. Prevents NaN/Inf propagation near roots of `p′` and unbounded orbits.

**Root-finding.** Durand-Kerner run once per polynomial change via `compute_roots` in `wasm-bridge`. After convergence, roots are sorted by `arg(root)` ascending for a canonical, hue-stable ordering. All tile workers read the same sorted root list from the Newton params shared memory slot.

**Return type.** Newton uses `NewtonResult`, a separate enum from `EscapeResult` — the two iteration models are semantically distinct and `EscapeResult`'s `Glitched` variant has no meaning for Newton:

```rust
pub enum NewtonResult {
    /// Orbit converged to a root within max_iter steps.
    Converged { root_index: u32, convergence_iter: u32 },
    /// Diverged (|z| > bailout) or hit max_iter without converging.
    /// Both failure modes map to a single `u_unresolved_color` uniform in the shader.
    Unresolved,
}
```

**Tile channel packing.** `r = convergence_t` (convergence_iter normalised to [0,1]), `g = 0.0` (unused), `b = root_index` (f32), `a = 1.0` if Converged else `0.0`. The GPU shader branches on `a` to choose between Newton HSL coloring and the unresolved uniform.

Newton fractals do not require perturbation theory — they are not deep-zoom targets. Standard f64 is sufficient.

#### 4.2.3 Bilinear Approximation (BLA)

Precomputes a table mapping `(iteration step, perturbation magnitude) → maximum skippable steps`. Built from the reference orbit in O(N log N) by the Orbit Worker immediately after orbit computation. Skips 80–99% of iterations in smooth interior regions; falls back to full perturbation near the boundary where detail is densest.

The BLA table lives in the shared `WebAssembly.Memory` BLA region alongside the reference orbit. Tile Workers read it directly — zero copy. See §9.3 for the correct merge formula.

### 4.3 `crate: coloring`

Coloring is separated from escape computation. `EscapeResult` is an enum with three mutually exclusive outcomes:

```rust
pub enum EscapeResult {
    /// Orbit escaped within max_iter steps.
    Escaped {
        iter:        u32,      // orbit index at which z_n first exceeded escape radius
        smooth_t:    f64,      // fractional escape for band-free coloring
        orbit_min_r: f64,      // orbit-trap: min |z| over {z_1, z_2, …}
        orbit_min_z: Complex<f64>,
        angle_final: f64,      // arg(z) at escape
    },
    /// Orbit did not escape (interior or convergent point).
    Interior {
        orbit_min_r: f64,
        orbit_min_z: Complex<f64>,
    },
    /// Pixel needs re-rendering with a better reference orbit.
    /// `From<EscapeResult> for TilePixel` panics on this variant —
    /// glitched pixels must be resolved before GPU upload.
    Glitched,
}
```

`orbit_min_r` and `orbit_min_z` are populated for both `Escaped` and `Interior` so the coloring crate can apply orbit-trap overlays to all pixels. `Interior` deliberately omits `iter` — the coloring crate does not need it, and removing it makes the type harder to misuse.

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

/// Called once per polynomial change (blur/enter in the UI).
/// Runs Durand-Kerner on the supplied real coefficients (ascending by degree),
/// then sorts the roots by arg(root) ascending for a canonical hue-stable ordering.
/// session.ts writes the returned roots + coefficients into the newton_params shared memory slot
/// before dispatching tiles.
#[wasm_bindgen]
pub fn compute_roots(coeffs: &[f64]) -> RootResult;
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
| **Newton params** | 264 B | `repr(C)`: `{ degree: u32, _pad: u32, epsilon: f64, coeffs: [f64; 11], roots: [Complex<f64>; 10] }`. Written once by `session.ts` after `compute_roots` returns; read by Tile Workers. Exposed via `layout().newton_params_offset`. |
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
| `exportFBO` | RGBA32F | Export accumulation — raw iteration data per strip; mirrors `accumFBO` format; RGBA32F preserves EXR upgrade path and is read back directly for Rust PNG encoding |

### 6.2 Render Passes

**Pass 1 — Tile Upload**

When the Scheduler forwards a `READY` tile, the main thread calls `gl.texSubImage3D` to write the raw iteration-count tile from the shared `WebAssembly.Memory` ring slot into the texture array layer. Raw data only — no color mapping here. Channels: `r = smooth_t`, `g = orbit_min_r`, `b = angle`, `a = escaped flag`.

**Pass 2 — Smooth-Color Accumulation**

Fullscreen quad reads from `tileTexArray` and `lut1D`. Branches on `u_fractal_kind`:

```glsl
if (u_fractal_kind == FRACTAL_NEWTON) {
    // a channel: 1.0 = Converged, 0.0 = Unresolved
    if (tile.a < 0.5) {
        fragColor = u_unresolved_color;
    } else {
        uint root_idx = uint(tile.b + 0.5);
        float hue = float(root_idx) / float(u_newton_degree) * 360.0;
        float lightness = mix(0.2, 0.8, tile.r);  // r = convergence_t
        fragColor = hsl_to_rgb(hue, 0.8, lightness);
    }
} else {
    // Mandelbrot / escape-time path
    float t = mod(tile.r * palette_speed + palette_offset, 1.0);
    vec4 color = texture(u_lut, t);

    // orbit trap blend
    float trap = 1.0 - smoothstep(0.0, trap_radius, tile.g);
    color = mix(color, trap_color, trap_strength * trap);

    fragColor = color;
}
```

Newton coloring uses auto-assigned evenly-spaced hues (`hue = root_index / degree * 360°`) with convergence speed encoded as lightness. No LUT is sampled for Newton pixels. Per-root hue overrides are a v2 addition to the palette editor.

**Pass 3 — Post-Process & Tonemap**

Second fullscreen pass over `accumFBO`:
- Brightness / contrast / saturation via ACES-approximate tonemap
- Optional Sobel edge-detection overlay (highlights set boundary)
- Gamma correction (sRGB output)

**Pass 4 — Export (on demand)**

Re-renders at target resolution into `exportFBO` (RGBA32F, strip-sized), reads back raw float pixels via `PIXEL_PACK_BUFFER` + `fenceSync` + RAF polling, and encodes to PNG via the Rust `encoding` crate running in a Tile Worker. See §12.

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
| `export.ts` | `ExportDialog`, `ExportSession`, `WorkerLease` | Resolution picker (PNG-only, DPI input, physical size hint); `ExportSession` drives strip render loop via `WorkerLease`; raw RGBA32F PBO readback; dispatches WASM `encode_png` to a Tile Worker |
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

| Zoom depth | Precision mode | Rust type | Status |
|---|---|---|---|
| zoom_exp < 15 | Native f64 escape-time | `Complex<f64>` | ✅ v1 — implemented |
| zoom_exp 15–20 | Double-double escape-time | `Complex<F64x2>` | 🔨 v1 — in progress |
| zoom_exp 20–300 | Perturbation + BigFloat<N> | `BigFloat<N>` ref, `DeltaC` probe | 🔜 v2 — deferred |
| zoom_exp > 300 | Rescaled `DeltaC` | extended `DeltaC` internals | 🔜 v2 — deferred |

**v1 zoom cap: zoom_exp ≈ 20.** F64x2 (double-double, ~30 decimal digits) covers this range with pure escape-time — no perturbation theory, no reference orbits, no glitch correction. Each pixel's coordinate is computed as `Complex<F64x2>` by adding the viewport center (parsed from the BigDecimal string) to the pixel offset (an exact f64 integer multiple of `pixel_step`). The arithmetic cost is ~3–4× f64 per iteration, acceptable for this zoom range.

Precision tier is selected automatically at runtime. `zoom_exp` is the log₁₀ of the magnification (e.g. `zoom_exp = 15` for 10⁻¹⁵ depth).

**v2 perturbation path:** The BigFloat<N> reference orbit, BLA acceleration, and glitch correction infrastructure is architecturally prepared (see §9) but deferred to v2. The `DeltaC` opaque type, `required_limbs`, and all kernel trait signatures are designed for forward compatibility — adding the perturbation dispatch in `wasm-bridge` is the only code change required to unlock zoom_exp > 20.

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

The perturbation loop uses a **pre-step escape check**: escape is tested on `z_n = Z_n + δz_n`
(the current full orbit value) _before_ computing `δz_{n+1}`. This aligns with `escape_time`'s
post-step convention when the smooth-t formula uses `n` instead of `n + 1` — the same escaped
z is used in both cases, so `TilePixel` output is bit-for-bit identical when `c_ref = 0`
(verified by the 16×16 grid test and three golden-coordinate regression tests).

`orbit_min_r` is tracked over `{z_1, z_2, …}`, excluding z_0 = 0, to match `escape_time`'s
convention and avoid a trivial orbit-trap hit at the fixed starting point.

The Pauldelbrot glitch criterion `|δz_n|² > 1e-6 · |Z_n|²` skips positions where `|Z_n| ≈ 0`
(guarded by `|Z_n|² > 1e-30`). Without this guard, the criterion fires incorrectly at orbit
positions where the reference passes near the origin — most notably at `n = 0` where `Z_0 = 0`
always. This is a common implementation pitfall.

```rust
pub fn perturb_pixel<M: PerturbationSupport>(
    map: &M,
    ref_orbit: &[Complex<f64>],
    delta_c: DeltaC,       // opaque type; f64 pair in v1
    max_iter: u32,
) -> EscapeResult {
    let dc = delta_c.as_complex_f64();
    let mut dz = Complex::<f64>::zero();
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = Complex::<f64>::zero();

    for (n, &z_ref) in ref_orbit[..limit].iter().enumerate() {
        let z_n = z_ref + dz;

        // Track orbit_min over {z_1, z_2, …} — z_0 = 0 excluded.
        if n > 0 {
            let r = z_n.norm_sqr().sqrt();
            if r < orbit_min_r { orbit_min_r = r; orbit_min_z = z_n; }
        }

        // Pre-step escape check on z_n = Z_n + δz_n.
        // smooth_t = n − ln(ln|z_n|)/ln 2 equals escape_time's (iter+1) − …
        // for the same escaped z (escape_time's iter = n − 1 for the same z).
        let r_sq = z_n.norm_sqr();
        if r_sq > map.escape_radius_sq() {
            let smooth_t = n as f64 - r_sq.sqrt().ln().ln() / LN_2;
            return EscapeResult::Escaped {
                iter: n as u32, smooth_t, orbit_min_r, orbit_min_z,
                angle_final: z_n.im.atan2(z_n.re),
            };
        }

        // Pauldelbrot glitch criterion: |δz_n|² > threshold² · |Z_n|².
        // Guard: skip when Z_n ≈ 0 (criterion undefined near origin).
        let ref_sq = z_ref.norm_sqr();
        if ref_sq > 1e-30 && dz.norm_sqr() > glitch_thresh_sq * ref_sq {
            return EscapeResult::Glitched;
        }

        dz = map.perturb_step(dz, dc, z_ref);
    }
    EscapeResult::Interior { orbit_min_r, orbit_min_z }
}
```

### 9.3 BLA (Bilinear Approximation)

In smooth regions, `δₙ₊ₖ ≈ A·δₙ + B·ΔC`. The BLA table precomputes valid `(A, B, skip, valid_radius)` entries for each starting iteration using Zhuoran's level-doubling algorithm.

```rust
// repr(C); total size 48 bytes (verified by size_of test).
pub struct BlaEntry {
    pub a:            Complex<f64>,   // 16 B — δz coefficient
    pub b:            Complex<f64>,   // 16 B — δc coefficient
    pub skip:         u32,            //  4 B — steps this entry covers
    _pad:             u32,            //  4 B — alignment padding before f64
    pub valid_radius: f64,            //  8 B — max |δz_n| for valid approximation
}
```

**Level-0 initialisation:** `a = 2·Z_n` (the Jacobian of one perturbation step w.r.t. δz), `b = 1`, `skip = 1`, `valid_radius = |Z_n|`. Positions where `Z_n = 0` (always true at `n = 0`) get `valid_radius = 0` — the BLA entry is never applicable there, and the rendering loop does a direct step.

**Merge formula (level-doubling):** When merging two consecutive entries `(A₀, B₀, skip₀, r₀)` and `(A₁, B₁, skip₁, r₁)`:

```
A_merged = A₁ · A₀
B_merged = A₁ · B₀ + B₁
skip_merged = skip₀ + skip₁
valid_radius_merged = min(r₀,  r₁ / |A₀|)   ← divide by |A₀|; omitting this is a common bug
```

The `/ |A₀|` factor accounts for the amplification of δz_n through the first half-skip before
it must satisfy the second half-skip's radius constraint. The reasoning: the second entry is
valid when `|δz_{n+skip₀}| < r₁`; since `δz_{n+skip₀} ≈ A₀·δz_n`, we need `|A₀|·|δz_n| < r₁`,
i.e. `|δz_n| < r₁/|A₀|`. Using `min(r₀, r₁)` instead produces subtly wrong escape counts in
smooth regions, visible as faint false banding. This is validated by a mandatory pixel-exact
comparison test (`bla_pixel_exact_256x256_tile_at_interior_ref`) that fails if the merge formula
is wrong.

**Build algorithm:** `build_bla_table` constructs `⌈log₂(N)⌉` levels of entries and stores the
highest-level valid entry per orbit position. Only positions where _both_ halves have
`valid_radius > 0` are merged; otherwise a sentinel entry with `valid_radius = 0` is stored for
that level. The final table has the same length as the input orbit; entries near the end have
smaller skip values because fewer remaining orbit steps are available.

BLA operates entirely on f64 — it is built from the already-downcast `Vec<Complex<f64>>` reference orbit and never touches `BigFloat`. The BLA table lives in the shared `WebAssembly.Memory` BLA region; Tile Workers read it with zero copy via the byte offset from `layout()`.

Pixel loop with BLA:

```rust
let mut n = 0;
while n < orbit_len {
    // … escape and glitch checks on z_n = ref_orbit[n] + dz …

    let entry = bla_table[n];
    let next_n = n + entry.skip as usize;
    if entry.valid_radius > 0.0 && dz.norm_sqr().sqrt() < entry.valid_radius
        && next_n <= orbit_len
    {
        // Skip entry.skip iterations in one multiply-add.
        dz = entry.a * dz + entry.b * dc;
        n = next_n;
    } else {
        // Direct perturbation step.
        dz = map.perturb_step(dz, dc, ref_orbit[n]);
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

    /// One step of the reference orbit in high-precision T.
    /// Generic over T so the same trait impl drives compute_ref_orbit for
    /// T = f64, F64x2, or BigFloat<N> without branching on precision.
    fn ref_step<T: Precision>(&self, z: Complex<T>, c: Complex<T>) -> Complex<T>;

    /// One step of the perturbation recurrence in f64 (the hot path).
    fn perturb_step(
        &self,
        dz: Complex<f64>,
        dc: Complex<f64>,
        ref_z: Complex<f64>,
    ) -> Complex<f64>;

    fn glitch_threshold(&self) -> f64 { 1e-3 }

    fn ref_state(&self, z_ref: Complex<f64>) -> Self::RefState;
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
| Julia z²+c | No (v1) | Fixed `c` parameter; pixel coord is `z₀`. Same escape-time math as Mandelbrot; perturbation not needed for the live-preview use case (f64 sufficient). Phase 7. |
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

Newton does not use `EscapeResult` or the `OrbitData` trait — it has its own return type. The kernel function returns `NewtonResult` directly:

```rust
pub enum NewtonResult {
    Converged { root_index: u32, convergence_iter: u32 },
    /// Diverged (|z| > 1e8) or reached max_iter without converging.
    Unresolved,
}
```

`NewtonResult` is packed into the standard 4-channel `TilePixel` by a Newton-specific packing function (not `From<EscapeResult>`):

| Channel | Value |
|---|---|
| `r` | `convergence_iter as f32 / max_iter as f32` (convergence_t) |
| `g` | `0.0` (unused) |
| `b` | `root_index as f32` |
| `a` | `1.0` if `Converged`, `0.0` if `Unresolved` |

The GPU shader reads these channels and computes HSL color directly (see §6.2). No LUT is sampled for Newton pixels. The `u_unresolved_color` uniform (dark, near-black by default) handles `Unresolved` pixels; a single uniform covers both divergent and max-iter-exceeded cases.

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
| PNG | `crates/encoding` (Rust, `png` 0.17) | Default; lossless; 16-bit RGB; pHYs (DPI→px/m), sRGB, tEXt metadata; encoded in a Tile Worker via WASM |
| EXR (32-bit float) | v2 — deferred | Full HDR; preserves raw `smooth_t` for compositing. `exportFBO` is RGBA32F so no GL pipeline change is required when added. |

### 12.2 Coordinate System

Export step = `fractalHeight(zoom_exp) / export_height` — identical formula to the viewport's `pixelStep`, using the export height. This preserves the same vertical fractal extent as the viewport regardless of export resolution; higher-resolution exports show the same view at greater pixel density. If the export aspect ratio differs from the viewport, extra fractal content is revealed on the wider axis — the export is never cropped or letterboxed.

Exports are always centered on the current `(cx, cy)`. The resolution picker offers presets (1080p, 4K, 8K, 16K) plus a free custom width × height input; aspect ratio is not locked.

### 12.3 Worker Handoff — WorkerLease

Export does not use a mode flag on `FractalSession`. Instead, `session.pause()` returns a `WorkerLease`:

```ts
interface WorkerLease {
  workers: Worker[];
  tileSab: SharedArrayBuffer;
  release(): void;   // restores default tile handler, calls scheduleDispatch()
}
```

`ExportSession` takes the lease, drives the workers directly for the duration of the export, and calls `lease.release()` in a `finally` block — the viewport always resumes regardless of completion, cancellation, or error. The viewport canvas is not cleared during the pause; the user continues to see the last rendered frame.

### 12.4 Strip Rendering

`exportFBO` is sized `export_width × strip_height` where `strip_height = Math.min(gl.MAX_RENDERBUFFER_SIZE, export_height)`. All exports — including those that fit in a single strip — use the same strip code path.

`ExportSession.run()` sequences strips strictly one at a time:

1. `gl.initExportFBOs(export_width, strip_height)` — allocate/resize `exportFBO` (RGBA32F) and PBO
2. Dispatch all 256×256 tiles covering the strip; route completions to `gl.uploadExportTile`
3. `gl.readbackRawStrip()` — async PBO readback of raw RGBA32F pixels directly from `exportFBO`:
   ```js
   gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
   gl.readPixels(0, 0, w, stripH, gl.RGBA, gl.FLOAT, 0);
   const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
   // poll via requestAnimationFrame until SIGNALED; return Float32Array
   ```
4. Accumulate strip into `allRaw: Float32Array` (full image, bottom-up row order)
5. Check `cancelled` flag; if set return `null` immediately
6. Repeat for next strip; after final strip, transfer `allRaw` to a Tile Worker for encoding

`gl.readPixels` is never called synchronously on the main thread. Progress is reported as `onProgress(tilesComplete, totalTiles)` after each tile, spanning all strips.

### 12.5 Encoding and Download

After all strips are collected, `ExportSession` posts an `encode_png` message to `lease.workers[0]`, transferring `allRaw` (ownership, zero-copy). The Tile Worker calls `wasm.encode_png(...)` — the wasm-bridge function that constructs `ColorParams` from the flat coloring arguments, calls `encoding::encode_png`, and returns `Uint8Array` PNG bytes. The main thread receives `encode_done`, wraps the bytes in a `Blob`, and triggers the download:

```ts
const a = document.createElement('a');
a.href = URL.createObjectURL(blob);
a.download = `fractal-${cx}-zoom${zoom_exp}.png`;
a.click();
setTimeout(() => URL.revokeObjectURL(a.href), 0);
```

**Row orientation:** GL `readPixels` returns rows bottom-up. `encoding::encode_png` iterates rows in reverse (`for row in (0..height).rev()`) so the PNG output is top-down with no intermediate allocation.

**`ColoringState`:** `FractalSession` accumulates live coloring state (LUT, all shader uniform values, fractal kind) in a `coloringState` field updated on every palette callback. `ExportParams` carries this alongside `dpi`, `view`, and fractal params — the encoding side has everything it needs with no GL state reads at export time.

`gl.MAX_RENDERBUFFER_SIZE` is queried once when the dialog opens; custom resolution inputs are clamped to this value with a visible warning.

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
| `kernel` | Unit + golden | `cargo-criterion` | Escape counts for known coordinates match published values; BLA skips ≥ 80%; BLA output matches full-perturbation render pixel-for-pixel. Newton: Durand-Kerner finds roots of `z³−1` to `1e-10`; convergence to correct root for known basin points; `Unresolved` on boundary points; Horner evaluation correct at degree 5 and 10 |
| `wasm-bridge` | Integration | `wasm-pack test --chrome` | JS API round-trips; shared memory writes correct data; `layout()` offsets consistent with actual writes |
| GL pipeline | Screenshot regression | Playwright + pixelmatch | `|Δpixel| < 2 LSB` on 8-bit for canonical zoom coordinates |
| UI shell | Component + E2E | vitest + Playwright | FSM state transitions |
| Export math | Unit | vitest | `computeStripHeight`, `computeStripCount`, `computeExportStep` return correct values for known inputs |
| Export pipeline | E2E golden | Playwright | PNG at 1080p for Mandelbrot default view: `\|Δpixel\| < 2 LSB`; IHDR asserts bit depth=16, color type=2 (RGB) |

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

> **Note:** `cargo-criterion` CI regression gating (blocking builds on > 10% perf regression) is deferred to v2. Benchmarks remain runnable locally; they do not gate CI in v1.

---

## 16. Phased Roadmap

| Phase | Deliverable | Crates / modules in scope |
|---|---|---|
| 0 — Scaffold | Workspace, CI, shared-memory WASM hello-world renders to canvas | `wasm-bridge` (stub), shared memory init |
| 1 — Mandelbrot f64 | Basic escape-time Mandelbrot, smooth coloring, pan/zoom | `kernel`, `coloring`, `gl-pipeline`, `viewport.ts` |
| 2 — Deep Zoom (v1) | F64x2 escape-time for zoom_exp 15–20; automatic precision dispatch | `arith` (F64x2), `wasm-bridge` (F64x2 dispatch), `session.ts` |
| 2b — Deep Zoom (v2) | Perturbation theory; BigFloat<N>; BLA; glitch correction / rebasing | `arith` (BigFloat), `kernel` (perturb, BLA), `scheduler` |
| 3 — Newton | Newton fractal: `NewtonResult` type, Durand-Kerner root-finding, Newton params shared memory slot, HSL shader branch, preset picker (`z³−1` default) with advanced coefficient overrides, `compute_roots` wasm-bridge entry point | `kernel` (newton), `wasm-bridge` (compute_roots, newton_params_offset), `gl-pipeline.ts` (shader branch), `session.ts` (polynomial change → compute_roots → tile dispatch) |
| 4 — Color Editor | Full palette editor, orbit trap, LUT, preset library | `coloring`, `color-editor.ts` |
| 5 — Export | PNG export pipeline (16-bit RGB, DPI, metadata); `ColoringState` accumulation; `WorkerLease` pause/resume; `ExportSession` strip loop; raw RGBA32F PBO readback; Rust `encode_png` via Tile Worker. EXR deferred to v2. | `crates/coloring`, `crates/encoding`, `wasm-bridge`, `export.ts`, `gl-pipeline.ts`, `session.ts` |
| 6 — Polish | UI polish pass (layout, spacing, accessibility, keyboard shortcut discoverability); URL bookmark support for Mandelbrot view state (Newton already done — tracked as GitHub issue); Playwright golden tests at specific Mandelbrot coordinates with known escape counts | `web/src/`, `web/e2e/` |
| 7 — Julia | Julia set as a new fractal type: `IterationMap` impl, parameter `c` picker in UI, `FractalKind::Julia` registration. Second step: Mandelbrot hover preview — pointer position over the Mandelbrot canvas drives the Julia `c` parameter in real time | `kernel` (julia map), `coloring`, `wasm-bridge`, `web/src/ui-overlay.ts`, `web/src/session.ts` |

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

### Perturbation Theory & BLA

- **K.I. Martin ("Superfractalthing")** — Original description of perturbation theory applied to the Mandelbrot set; introduced the core formula `δₙ₊₁ = 2·Z_n·δₙ + δₙ² + ΔC` and the glitch-detection criterion. Fractalforums.org, 2013. <https://fractalforums.org/index.php?topic=4022>
- **Pauldelbrot** — Extended Superfractalthing with BLA (Bilinear Approximation), secondary orbit correction, and practical glitch-threshold guidance. Fractalforums.org, 2013–2021. <https://fractalforums.org/index.php?topic=4353>
- **Zhuoran** — Level-doubling algorithm for BLA table construction; the `valid_radius` merge formula `min(r₀, r₁ / |A₀|)` presented here. Fractalforums.org, 2021.
- **Claude Heiland-Allen** — "Perturbation Theory for the Mandelbrot Set" — clear mathematical writeup of the BLA derivation and glitch correction. <https://mathr.co.uk/mandelbrot/perturbation-theory.html>
- **Kalles Fraktaler 2** — Open-source deep-zoom renderer; reference implementation for BLA and glitch correction in C++. <https://fractalwiki.org/wiki/Kalles_Fraktaler>

### Extended-Precision Arithmetic

- **Dekker, T.J.** — "A Floating-Point Technique for Extending the Available Precision." *Numerische Mathematik* 18(3), 1971. Introduces Veltkamp splitting (the `2²⁷ + 1` constant) and the TwoProd exact-product algorithm used in `F64x2::mul`.
- **Knuth, D.E.** — *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*, §4.2.2. Addison-Wesley, 1969 (3rd ed. 1997). Source of the TwoSum algorithm (`s = a + b; e = (a − (s − v)) + (b − v)`).
- **Shewchuk, J.R.** — "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates." *Discrete & Computational Geometry* 18(3), 1997. Comprehensive treatment of error-free transformations including TwoSum and TwoProd. <https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf>
- **Hida, Li, Bailey** — "Quad-Double Arithmetic: Algorithms, Implementation, and Application." LBNL Technical Report, 2000. Extends double-double to quad-double; useful background on the F64x2 invariant `|lo| ≤ ½ ulp(hi)`.
- **Fousse, Hanrot, Lefèvre, Pélissier, Zimmermann** — "MPFR: A Multiple-Precision Binary Floating-Point Library with Correct Rounding." *ACM TOMS* 33(2), 2007. The MPFR library (via the `rug` Rust crate) is used as the oracle in `arith` property-based tests for `F64x2` and `BigFloat<N>`.

### WebAssembly & Web Platform

- **wasm-bindgen Reference** — <https://rustwasm.github.io/docs/wasm-bindgen/>
- **wasm-pack Book** — <https://rustwasm.github.io/docs/wasm-pack/>
- **WebAssembly Threads proposal** — Atomics, `SharedArrayBuffer` as WASM memory, `wait`/`notify`. <https://github.com/WebAssembly/threads>
- **WebAssembly SIMD proposal** — `v128` type and intrinsics used by the `simd` feature gate on `F64x2` hot loops. <https://github.com/WebAssembly/simd>
- **WebGL 2 Specification** — <https://registry.khronos.org/webgl/specs/latest/2.0/>

### Coloring & Rendering

- **Linas Vepstas** — "Renormalising the Mandelbrot Escape" — derivation of the smooth iteration count formula `iter + 1 − log₂(log|z|)`. <https://linas.org/art-gallery/escape/escape.html>
- **Narkowicz & Evangelista** — "ACES Filmic Tone Mapping Curve." 2015. The approximation used in the post-process pass. <https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/>
- **Inigo Quilez** — Orbit trap techniques and coloring methods for escape-time fractals. <https://iquilezles.org/articles/> (various articles on Mandelbrot coloring)
