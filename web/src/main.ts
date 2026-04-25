import { detectSimd, wasmBundleUrl } from "./detect-simd.ts";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Verify SharedArrayBuffer is available (requires COOP/COEP headers).
if (typeof SharedArrayBuffer === "undefined") {
  console.error(
    "SharedArrayBuffer is not available. " +
      "Check that COOP/COEP headers are set correctly.",
  );
} else {
  console.log("SharedArrayBuffer: available ✓");
}

const simd = detectSimd();
console.log(`SIMD support: ${simd ? "available ✓" : "not available — using fallback"}`);
console.log(`WASM bundle: ${wasmBundleUrl()}`);

// Placeholder: Phase 0 issue #6 will instantiate WASM with shared memory
// and render the first tile to this canvas.
const ctx = canvas.getContext("2d");
if (ctx) {
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#555";
  ctx.font = "16px monospace";
  ctx.fillText("Phase 0 scaffold — shared-memory hello-world coming in issue #6", 20, 40);
}
