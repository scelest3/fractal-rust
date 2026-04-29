import { FractalSession } from "./session.ts";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
// Start at HiDPI if the display supports it; the session toggle can drop to 1×.
const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
canvas.width  = Math.round(window.innerWidth  * dpr);
canvas.height = Math.round(window.innerHeight * dpr);
if (dpr !== 1) {
  canvas.style.width  = window.innerWidth  + "px";
  canvas.style.height = window.innerHeight + "px";
}

if (typeof SharedArrayBuffer === "undefined") {
  document.body.innerHTML =
    "<p style='color:red;padding:1rem'>SharedArrayBuffer unavailable — " +
    "run with <code>npm run dev</code> (COOP/COEP headers required).</p>";
} else {
  new FractalSession(canvas);
}
