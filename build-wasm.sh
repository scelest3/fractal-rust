#!/usr/bin/env bash
# Build both WASM bundles (SIMD and non-SIMD) with atomics preserved.
#
# Prerequisites: wasm-pack, wasm-opt (brew install binaryen), wasm32 target
#   rustup target add wasm32-unknown-unknown
#
# Usage: ./build-wasm.sh [--dev]
set -euo pipefail

DEV=""
if [[ "${1:-}" == "--dev" ]]; then
  DEV="--dev"
fi

echo "==> Building SIMD bundle..."
wasm-pack build crates/wasm-bridge \
  --target web \
  --no-pack \
  $DEV \
  --out-dir ../../web/pkg \
  -- --features simd

echo "==> wasm-opt pass (SIMD + threads)..."
wasm-opt web/pkg/wasm_bridge_bg.wasm \
  -O3 --enable-threads --enable-simd \
  -o web/pkg/wasm_bridge_bg.wasm

echo "==> Building non-SIMD bundle..."
wasm-pack build crates/wasm-bridge \
  --target web \
  --no-pack \
  $DEV \
  --out-dir ../../web/pkg-nosimd

echo "==> wasm-opt pass (threads only)..."
wasm-opt web/pkg-nosimd/wasm_bridge_bg.wasm \
  -O3 --enable-threads \
  -o web/pkg-nosimd/wasm_bridge_bg.wasm

echo "==> Done. Bundles in web/pkg/ and web/pkg-nosimd/"
