/**
 * Palette data model, HSL LUT builder, and preset constants.
 *
 * buildLut() produces a 4096×RGBA f32 array suitable for GlPipeline.uploadLut().
 * Interpolation is done in HSL space with per-stop hue-direction control.
 * The gradient is cyclic: the last stop wraps back to the first.
 */

export interface ColorStop {
  t: number;                 // position in [0, 1), stops must be sorted by t
  r: number;                 // sRGB 0–1
  g: number;
  b: number;
  hueDir: "short" | "long"; // hue arc direction to the next stop
}

export interface Palette {
  stops: ColorStop[];  // ≥ 2 stops; cyclic — last wraps back to first
  cycleLen: number;    // uCycleLen shader uniform (band width in smooth_t units)
}

// ── HSL helpers ───────────────────────────────────────────────────────────────

/** RGB [0,1] → HSL with h ∈ [0,360), s ∈ [0,1], l ∈ [0,1]. */
function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l]; // achromatic — hue undefined, treat as 0

  const d = max - min;
  const s = l < 0.5 ? d / (max + min) : d / (2 - max - min);
  let h: number;
  if      (max === r) h = ((g - b) / d % 6 + 6) % 6 * 60;
  else if (max === g) h = ((b - r) / d + 2) * 60;
  else                h = ((r - g) / d + 4) * 60;
  return [h, s, l];
}

/** HSL (h ∈ [0,360), s ∈ [0,1], l ∈ [0,1]) → RGB [0,1]. */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if      (h < 60)  { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) {         g = c; b = x; }
  else if (h < 240) {         g = x; b = c; }
  else if (h < 300) { r = x;         b = c; }
  else              { r = c;         b = x; }
  return [r + m, g + m, b + m];
}

/**
 * Interpolate hue from h1 to h2 by fraction t using the specified arc.
 * "short" takes the arc ≤ 180°; "long" takes the arc > 180°.
 */
function lerpHue(h1: number, h2: number, t: number, dir: "short" | "long"): number {
  const cw = ((h2 - h1) % 360 + 360) % 360; // clockwise delta, always [0, 360)
  const delta = dir === "short"
    ? (cw > 180 ? cw - 360 : cw)  // short: magnitude ≤ 180
    : (cw < 180 ? cw - 360 : cw); // long:  magnitude > 180
  return ((h1 + delta * t) % 360 + 360) % 360;
}

// ── LUT builder ───────────────────────────────────────────────────────────────

/**
 * Build a 4096-entry RGBA f32 LUT from a Palette.
 *
 * Each entry is produced by HSL interpolation within its segment. The last
 * segment wraps cyclically from stops[n-1] back to stops[0]. Alpha is always
 * 1.0 — the shader controls interior vs. exterior, not the LUT.
 */
export function buildLut(palette: Palette): Float32Array {
  const SIZE = 4096;
  const { stops } = palette;
  const n = stops.length;
  const data = new Float32Array(SIZE * 4);

  for (let i = 0; i < SIZE; i++) {
    const t = i / SIZE;

    // Find segment: the largest seg index where stops[seg].t <= t.
    let seg = n - 1;
    for (let j = 0; j < n - 1; j++) {
      if (t < stops[j + 1].t) { seg = j; break; }
    }

    const a = stops[seg];
    const b = stops[(seg + 1) % n];
    const segEnd = seg < n - 1 ? b.t : 1.0; // last segment ends at 1.0
    const segLen = segEnd - a.t;
    const f = segLen > 0 ? (t - a.t) / segLen : 0;

    const [h1, s1, l1] = rgbToHsl(a.r, a.g, a.b);
    const [h2, s2, l2] = rgbToHsl(b.r, b.g, b.b);
    const h = lerpHue(h1, h2, f, a.hueDir);
    const s = s1 + (s2 - s1) * f;
    const l = l1 + (l2 - l1) * f;
    const [r, g, bVal] = hslToRgb(h, s, l);

    data[i * 4]     = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = bVal;
    data[i * 4 + 3] = 1.0;
  }
  return data;
}

// ── Presets ───────────────────────────────────────────────────────────────────

const S = (t: number, r: number, g: number, b: number): ColorStop =>
  ({ t, r, g, b, hueDir: "short" });

export const CLASSIC: Palette = {
  stops: [S(0.00, 0.00, 0.00, 0.50),  // deep blue
          S(0.25, 0.00, 0.80, 1.00),  // cyan
          S(0.50, 1.00, 1.00, 1.00),  // white
          S(0.75, 1.00, 0.60, 0.00)], // gold → wraps to deep blue
  cycleLen: 64,
};

export const FIRE: Palette = {
  stops: [S(0.00, 0.00, 0.00, 0.00),  // black
          S(0.25, 1.00, 0.00, 0.00),  // red
          S(0.50, 1.00, 0.50, 0.00),  // orange
          S(0.75, 1.00, 1.00, 0.00)], // yellow → wraps to black
  cycleLen: 64,
};

export const ICE: Palette = {
  stops: [S(0.00, 0.00, 0.00, 0.00),  // black
          S(0.25, 0.00, 0.00, 0.50),  // navy
          S(0.50, 0.00, 0.90, 1.00),  // cyan
          S(0.75, 1.00, 1.00, 1.00)], // white → wraps to black
  cycleLen: 64,
};

export const ELECTRIC: Palette = {
  stops: [S(0.00, 0.00, 0.00, 0.00),  // black
          S(0.25, 0.50, 0.00, 1.00),  // purple
          S(0.50, 1.00, 0.00, 1.00),  // magenta
          S(0.75, 0.00, 1.00, 1.00)], // cyan → wraps to black
  cycleLen: 64,
};

export const GRAYSCALE: Palette = {
  stops: [S(0.00, 0.00, 0.00, 0.00),  // black
          S(0.50, 1.00, 1.00, 1.00)], // white → wraps to black
  cycleLen: 64,
};

export const SHANNON: Palette = {
  stops: [S(0.00, 0.00, 0.00, 0.00),  // black
          S(0.25, 0.40, 0.00, 0.80),  // purple
          S(0.50, 0.70, 0.60, 1.00),  // lavender
          S(0.75, 0.00, 0.60, 0.00)], // green → wraps to black
  cycleLen: 64,
};

export const ALL_PRESETS: { name: string; palette: Palette }[] = [
  { name: "Classic",   palette: CLASSIC },
  { name: "Fire",      palette: FIRE },
  { name: "Ice",       palette: ICE },
  { name: "Electric",  palette: ELECTRIC },
  { name: "Grayscale", palette: GRAYSCALE },
  { name: "Shannon",   palette: SHANNON },
];
