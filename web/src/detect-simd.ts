/**
 * Synchronous WASM SIMD feature probe.
 *
 * Validates a minimal WASM module containing one `v128.const` instruction.
 * Runs before any network request — no fetch, no async, no DOM required.
 * Returns true if the browser supports wasm-simd128.
 */
export function detectSimd(): boolean {
  // Minimal WASM module with a single v128.const instruction.
  // If the browser can validate it, SIMD is available.
  return WebAssembly.validate(
    new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10,
      10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
    ]),
  );
}

/**
 * Selects the correct WASM bundle URL based on SIMD support.
 * Called once at startup before instantiation.
 */
export function wasmBundleUrl(): string {
  return detectSimd()
    ? "/pkg/wasm_bridge_bg.wasm"
    : "/pkg-nosimd/wasm_bridge_bg.wasm";
}
