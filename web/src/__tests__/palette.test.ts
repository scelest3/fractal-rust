import { describe, it, expect } from "vitest";
import {
  buildLut,
  CLASSIC, GRAYSCALE,
  type Palette,
} from "../palette.ts";

const TOL = 1 / 255; // one 8-bit step

function makeRedBluePalette(dir: "short" | "long"): Palette {
  return {
    stops: [
      { t: 0.0, r: 1, g: 0, b: 0, hueDir: dir }, // red  h=0
      { t: 0.5, r: 0, g: 0, b: 1, hueDir: dir }, // blue h=240
    ],
    cycleLen: 64,
  };
}

// ── Size and format ───────────────────────────────────────────────────────────

describe("buildLut output format", () => {
  it("has exactly 4096 × 4 floats", () => {
    expect(buildLut(CLASSIC).length).toBe(4096 * 4);
  });

  it("alpha channel is always 1.0", () => {
    const lut = buildLut(CLASSIC);
    for (let i = 0; i < 4096; i++) {
      expect(lut[i * 4 + 3]).toBe(1.0);
    }
  });
});

// ── Stop boundary accuracy ────────────────────────────────────────────────────

describe("stop boundary colors", () => {
  it("index 0 matches CLASSIC stop 0 (deep blue)", () => {
    const lut = buildLut(CLASSIC);
    // stop 0: r=0, g=0, b=0.5
    expect(lut[0]).toBeCloseTo(0.0, 5);
    expect(lut[1]).toBeCloseTo(0.0, 5);
    expect(lut[2]).toBeCloseTo(0.5, 4);
  });

  it("index 1024 matches CLASSIC stop 1 (cyan) — t=0.25", () => {
    const lut = buildLut(CLASSIC);
    // stop 1: r=0, g=0.8, b=1.0
    expect(lut[1024 * 4 + 0]).toBeCloseTo(0.0, 4);
    expect(lut[1024 * 4 + 1]).toBeCloseTo(0.8, 3);
    expect(lut[1024 * 4 + 2]).toBeCloseTo(1.0, 4);
  });

  it("index 2048 matches CLASSIC stop 2 (white) — t=0.5", () => {
    const lut = buildLut(CLASSIC);
    expect(lut[2048 * 4 + 0]).toBeCloseTo(1.0, 4);
    expect(lut[2048 * 4 + 1]).toBeCloseTo(1.0, 4);
    expect(lut[2048 * 4 + 2]).toBeCloseTo(1.0, 4);
  });
});

// ── Cyclic wrap ───────────────────────────────────────────────────────────────

describe("cyclic wrap", () => {
  it("index 4095 is close to CLASSIC stop 0 (deep blue)", () => {
    const lut = buildLut(CLASSIC);
    // t=4095/4096 is one step before the wrap — should be very close to deep blue
    expect(lut[4095 * 4 + 0]).toBeCloseTo(0.0, 2);
    expect(lut[4095 * 4 + 2]).toBeGreaterThan(0.4); // blue channel dominant
  });

  it("GRAYSCALE index 2048 is white (midpoint between black and white)", () => {
    const lut = buildLut(GRAYSCALE);
    expect(lut[2048 * 4 + 0]).toBeCloseTo(1.0, 4);
    expect(lut[2048 * 4 + 1]).toBeCloseTo(1.0, 4);
    expect(lut[2048 * 4 + 2]).toBeCloseTo(1.0, 4);
  });
});

// ── HSL hue direction ─────────────────────────────────────────────────────────

describe("hueDir", () => {
  it('"short" interpolates red→blue via magenta at segment midpoint (h≈330)', () => {
    // 2-stop palette: red (t=0) and blue (t=0.5). Segment 0 covers t=[0,0.5).
    // Midpoint of segment 0 is t=0.25 → index 1024. Short arc: h goes 0→-120 → h≈330.
    const lut = buildLut(makeRedBluePalette("short"));
    const r = lut[1024 * 4 + 0];
    const g = lut[1024 * 4 + 1];
    const b = lut[1024 * 4 + 2];
    // Magenta/pink zone (h≈330): red and blue both significant, green low
    expect(r).toBeGreaterThan(0.3);
    expect(b).toBeGreaterThan(0.3);
    expect(g).toBeLessThan(0.15);
  });

  it('"long" interpolates red→blue via green at segment midpoint (h=120)', () => {
    // Midpoint of segment 0 at t=0.25 → index 1024. Long arc: h goes 0→+240 → h=120.
    const lut = buildLut(makeRedBluePalette("long"));
    const r = lut[1024 * 4 + 0];
    const g = lut[1024 * 4 + 1];
    const b = lut[1024 * 4 + 2];
    // Green zone (h=120): green dominant, red and blue near zero
    expect(g).toBeGreaterThan(0.3);
    expect(r).toBeLessThan(0.15);
    expect(b).toBeLessThan(0.15);
  });
});
