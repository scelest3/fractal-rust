# Fractal Explorer

High-resolution fractal explorer built on Rust, WebAssembly, and WebGL 2. The Rust kernel handles all heavy math (perturbation theory, extended precision arithmetic, BLA acceleration). The TypeScript shell handles UI, input, and WebGL orchestration.

Supports Mandelbrot, Julia sets, Multibrot, Burning Ship, Newton, and Phoenix fractals with smooth coloring, palette editing, and high-resolution PNG/JPEG/EXR export.

---

## Prerequisites

- [Rust](https://rustup.rs/) (stable)
- `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [Binaryen](https://github.com/WebAssembly/binaryen) (provides `wasm-opt`): `brew install binaryen`
- Node.js 20+

---

## Local Development

```sh
# 1. Build WASM bundles (SIMD + non-SIMD fallback)
./build-wasm.sh

# 2. Install web dependencies
cd web && npm install

# 3. Start dev server
npm run dev
```

The dev server runs at `http://localhost:5173` and automatically sets the required `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers for `SharedArrayBuffer` support.

---

## Production Build

```sh
./build-wasm.sh
cd web && npm run build
```

Output lands in `web/dist/`. The WASM bundles are copied automatically from `web/public/` as part of the Vite build.

### Required server headers

Every response (HTML, JS, WASM) must include:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without these, `SharedArrayBuffer` is unavailable and the app will not start. A `_headers` file is included in the build output for hosts that support that format. For Nginx or other servers, configure these headers in your server block.

---

## Running Tests

```sh
# Rust unit tests
cargo test --workspace

# TypeScript unit tests
cd web && npm test

# E2E tests (requires dev server running on :5173)
npm run test:e2e

# WASM integration tests (requires Chrome)
wasm-pack test crates/wasm-bridge --chrome --headless
```

---

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for full design documentation, including the tile rendering pipeline, perturbation theory implementation, BLA acceleration, and the shared WebAssembly memory layout.
