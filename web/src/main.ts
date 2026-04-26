import { FractalSession } from "./session.ts";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

if (typeof SharedArrayBuffer === "undefined") {
  document.body.innerHTML =
    "<p style='color:red;padding:1rem'>SharedArrayBuffer unavailable — " +
    "run with <code>npm run dev</code> (COOP/COEP headers required).</p>";
} else {
  new FractalSession(canvas);
}
