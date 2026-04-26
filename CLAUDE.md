# CLAUDE.md

This file tells Claude Code how to work in this repository.

---

## Project Overview

High-resolution fractal explorer built on Rust/WASM/WebGL 2. The Rust kernel handles all
heavy math (perturbation theory, extended precision arithmetic, BLA). The TypeScript shell
handles UI, input, and WebGL orchestration. They communicate through a thin wasm-bindgen
surface backed by a shared `WebAssembly.Memory` tile ring — there is no WASM instance on
the main thread; `session.ts` is pure TypeScript.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for full design documentation.

---

## Repository Layout

```
fractal-workspace/
  Cargo.toml          # workspace root
  .cargo/config.toml  # wasm32 atomics + bulk-memory flags — do not remove
  ARCHITECTURE.md     # full design doc — read this first
  CLAUDE.md           # this file
  crates/
    arith/            # no_std extended-precision arithmetic
    kernel/           # iteration maps, perturbation, BLA
    coloring/         # EscapeResult → RGBA coloring algorithms
    scheduler/        # tile priority queue
    wasm-bridge/      # wasm-bindgen surface (only crate with wasm-bindgen dep)
  web/
    src/              # TypeScript UI shell
    shaders/          # GLSL vertex + fragment source
    pkg/              # wasm-pack output — DO NOT EDIT, gitignored
    index.html
    vite.config.ts
```

---

## Build Commands

### Rust / WASM

```sh
# Build WASM module — two steps: wasm-pack then wasm-opt (must preserve atomics)
wasm-pack build crates/wasm-bridge \
  --target web \
  --no-pack \
  --out-dir ../../web/pkg \
  -- --features simd

# wasm-opt must run with --enable-threads or atomics are silently stripped
wasm-opt web/pkg/wasm_bridge_bg.wasm \
  -O3 --enable-threads --enable-simd \
  -o web/pkg/wasm_bridge_bg.wasm

# Non-SIMD bundle (for older browsers)
wasm-pack build crates/wasm-bridge \
  --target web \
  --no-pack \
  --out-dir ../../web/pkg-nosimd

wasm-opt web/pkg-nosimd/wasm_bridge_bg.wasm \
  -O3 --enable-threads \
  -o web/pkg-nosimd/wasm_bridge_bg.wasm

# Run all Rust tests (native, no browser)
cargo test --workspace

# Run benchmarks
cargo criterion -p kernel
cargo criterion -p arith

# Lint
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all
```

### TypeScript / Web

```sh
cd web

# Install dependencies
npm install

# Dev server (sets COOP/COEP headers automatically)
npm run dev

# Production build
npm run build

# Unit tests
npm run test

# E2E tests (requires dev server running)
npm run test:e2e
```

### Run WASM integration tests in browser

```sh
wasm-pack test crates/wasm-bridge --chrome --headless
```

---

## Dev Server Requirements

The dev server **must** set these headers or `SharedArrayBuffer` will be unavailable:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

These are already configured in `vite.config.ts`. For production, mirror them in Nginx.

---

## Key Architectural Constraints

**Never add `wasm-bindgen` to any crate except `wasm-bridge`.** The math crates (`arith`,
`kernel`, `coloring`, `scheduler`) must stay `no_std`-compatible and testable on native
targets.

**Never allocate heap memory in `arith`.** `BigFloat<N>` is a fixed-size stack array.
If you're reaching for `Vec` in `arith`, stop and reconsider.

**Never block the main thread.** All compute goes through the Worker pool. The main thread
only runs WebGL draw calls and DOM event handlers. `session.ts` contains no WASM instance
and no heavy computation — it is a pure TypeScript orchestrator.

**Never instantiate WASM using the wasm-bindgen default init.** Workers instantiate
manually via `WebAssembly.instantiateStreaming`, injecting the shared `WebAssembly.Memory`
created on the main thread. The generated `initSync` / `init` exports are not used.

**Coordinate precision.** View state is always serialized as `BigDecimal` strings
(`cx`, `cy`) and a `zoom_exp: f64` log₁₀ exponent. Never store the raw zoom scale factor
as f64 — it overflows near 10⁻³⁰⁰. Losing precision in the URL hash or a saved bookmark
is a silent correctness bug.

**`DeltaC` is an opaque type.** Never construct `Complex<f64>` directly for per-pixel
perturbation coordinates — always go through `DeltaC`. In v1 the internals are f64, but
the type must stay opaque so v2 can upgrade to rescaled extended precision without changing
call sites. The v1 hard cap is `zoom_exp = 300`; assert this in `required_limbs`.

**BLA `valid_radius` merge formula.** When merging two BLA entries `(A₀, r₀)` and
`(A₁, r₁)`, the correct merged radius is `min(r₀, r₁ / |A₀|)`. Using `min(r₀, r₁)`
is a silent correctness bug that produces faint false banding in smooth regions.

---

## Adding a New Fractal Type

1. Implement `IterationMap` for the new map in `crates/kernel/src/maps/`
2. If it needs deep zoom, implement `PerturbationSupport` with the correct associated types:
   - `RefState` — per-step reference state (e.g. sign bits for Burning Ship; `()` for Mandelbrot)
   - `RefOrbitEntry` — one entry in the reference orbit buffer (`Complex<f64>` for most maps;
     `(Complex<f64>, Complex<f64>)` for Phoenix). The shared memory orbit header records the
     entry size; Tile Workers read it from there — don't hardcode the stride.
3. Implement `OrbitData` for the coloring data the orbit produces
4. Add a `ColoringAlgorithm` impl in `crates/coloring/src/` for the new `OrbitData` type
5. Register in `fractal_registry!` macro in `crates/wasm-bridge/src/registry.rs` — the
   TypeScript `FractalKind` union type is generated automatically by `wasm-pack`; no manual
   sync needed
6. Add parameter panel in `web/src/ui-overlay.ts` if the fractal has user-configurable params
7. Add golden render test in `web/e2e/`

See [ARCHITECTURE.md §10](./ARCHITECTURE.md#10-adding-new-fractal-types) for the full
trait design and the fractal catalog.

---

## Precision Tier Selection

The correct `BigFloat<N>` limb count is chosen automatically via `required_limbs(zoom_exp)`
in `crates/arith/src/lib.rs`. `zoom_exp` is the log₁₀ of the magnification depth (e.g.
`zoom_exp = 100` for 10⁻¹⁰⁰). Don't pass a raw scale factor — it will overflow.
Don't hardcode limb counts in kernel or bridge code. The shipped WASM module includes
monomorphizations for `N = 2, 4, 8` only; `N > 8` uses dynamic dispatch.

---

## Testing Guidelines

Use the `/tdd` skill when implementing new behavior in `arith`, `kernel`, `coloring`, or
the TypeScript FSM / coordinate modules. These layers have clear public interfaces and
pure-function behavior — the red-green-refactor loop fits naturally. Do not use `/tdd` for
the GL pipeline or WASM bridge integration tests; those require a real browser and are
covered by Playwright and `wasm-pack test` respectively.

**Mocking boundary:** The WASM bridge (`web/pkg/`) is the system boundary for TypeScript
tests. Mock that interface in vitest; never mock internal TypeScript modules. For Rust,
don't mock — use `proptest` properties and golden-coordinate comparisons instead.

- **`arith` and `kernel`:** All correctness tests run natively (`cargo test`). Use
  `proptest` for arithmetic properties; compare against known-good coordinates from
  published deep-zoom benchmarks.
- **BLA validation:** Before Phase 2 ships, BLA output must be validated pixel-for-pixel
  against a full-perturbation reference render at a known deep-zoom coordinate. A mismatch
  almost always means the `valid_radius` merge formula is wrong.
- **WASM bridge:** Integration tests via `wasm-pack test --chrome --headless`. These
  cover the shared memory round-trip, `layout()` offset correctness, and JS API contracts.
- **GL pipeline:** Screenshot regression via Playwright + pixelmatch. Golden images live
  in `web/e2e/goldens/`. Regenerate with `npm run test:e2e -- --update-snapshots`.
- **Pixel tolerance:** `|Δpixel| < 2 LSB` on 8-bit output. Larger deltas indicate a
  coloring or precision regression, not a rendering variance.
- **Performance:** `cargo criterion` benches gate on CI. A regression > 10% in `arith`
  or `kernel` hot paths fails the build.

---

## Coloring / LUT

The LUT is always 4 096 entries × RGBA f32. It is uploaded to a WebGL `lut1D` texture
and sampled by the fragment shader. Rebuild the LUT whenever palette parameters change;
avoid rebuilding on every frame. The `PaletteEditor` in `color-editor.ts` owns the
rebuild lifecycle.

Orbit trap parameters (`trap_radius`, `trap_strength`, `trap_color`) are passed as
WebGL uniforms, not baked into the LUT.

---

## Export Notes

- PNG and JPEG go through `canvas.toBlob` — no extra dependencies.
- EXR export requires `openexr-wasm` (separate WASM module, loaded lazily on first export).
- Exports > 8 192 px use strip rendering stitched on `OffscreenCanvas`. The strip count
  is determined by `gl.MAX_RENDERBUFFER_SIZE` at runtime, not hardcoded.
- Strip readback uses `PIXEL_PACK_BUFFER` + `gl.fenceSync` + RAF polling — never
  `gl.readPixels` synchronously on the main thread (blocks the tab for large buffers).
- Export pauses live viewport rendering: the Scheduler enters `ExportMode`, drains
  in-flight tiles, and dedicates the full worker pool to export tiles until complete.
- The export pipeline re-renders at target resolution; it does not upscale from the
  viewport `accumFBO`.

---

## Common Pitfalls

| Symptom | Likely cause |
|---|---|
| Tiles render but colors look wrong at deep zoom | Ref orbit not recomputed after zoom settle; check `ref_orbit_id` |
| Scattered wrong-colored pixels in a render | Glitch detection threshold too loose; tighten `glitch_threshold()` |
| Subtle false banding in smooth regions | BLA `valid_radius` merge missing `/ |A₀|` factor; see §9.3 of ARCHITECTURE.md |
| All pixels identical past zoom_exp ≈ 305 | `DeltaC` f64 underflowing; v1 cap is 300 — check UI clamp and `required_limbs` assertion |
| `SharedArrayBuffer is not defined` in console | COOP/COEP headers missing; check Vite config |
| WASM module fails to load in Safari | HTTPS required for SharedArrayBuffer even on localhost |
| WASM instantiation fails with `LinkError` on shared memory | `wasm-opt` stripped atomics; re-run with `--enable-threads` |
| `BigFloat<N>` compile errors after adding new N | Add monomorphization to the feature gate list in `wasm-bridge/Cargo.toml` |
| Tile workers see stale orbit data after zoom settle | Memory fence missing — orbit_ready postMessage must come after all shared memory writes |
| Shared memory allocation fails at startup | `MAX_PAGES` too small; recalculate from `MAX_ITER` and `MAX_WORKERS` and assert before `new WebAssembly.Memory` |
| wasm-bindgen module won't accept injected `env.memory` | With stable Rust, wasm-bindgen produces a module that exports (not imports) its own memory. Injecting a shared `WebAssembly.Memory` requires `-Z build-std` (nightly) to compile std with `+atomics`. Phase 1 works around this with per-worker WASM instances + a single `tileSab`; each worker owns one slot and copies into it. Upgrade when the toolchain constraint is lifted. |
