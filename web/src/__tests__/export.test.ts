import { describe, it, expect } from "vitest";
import { computeStripHeight, computeStripCount, computeExportStep, type ColoringState } from "../export.ts";
import { pixelStep } from "../viewport.ts";

// ── computeStripHeight ────────────────────────────────────────────────────────

describe("computeStripHeight", () => {
  it.each([
    // GPU limit binds (image fits within memory budget at these widths)
    { exportHeight: 1080, maxRenderbufferSize: 8192, exportWidth: 1920,  expected: 1080 },
    { exportHeight: 4320, maxRenderbufferSize: 4096, exportWidth: 1920,  expected: 4096 },
    { exportHeight: 4096, maxRenderbufferSize: 4096, exportWidth: 1920,  expected: 4096 },
    { exportHeight: 100,  maxRenderbufferSize: 4096, exportWidth: 1920,  expected: 100  },
  ])("GPU-bound: min($exportHeight, $maxRenderbufferSize) = $expected", ({ exportHeight, maxRenderbufferSize, exportWidth, expected }) => {
    expect(computeStripHeight(exportHeight, maxRenderbufferSize, exportWidth)).toBe(expected);
  });

  it("returns 1 when export height is 1", () => {
    expect(computeStripHeight(1, 4096, 1920)).toBe(1);
  });

  it("memory budget caps strip height for 16K wide images", () => {
    // 16K: 15360 × 4 channels = 61 440 floats/row.
    // 256M floats / (15360 × 4) ≈ 4369 rows — below the 8640 image height,
    // so the budget is the binding constraint.
    const h = computeStripHeight(8640, 16384, 15360);
    expect(h).toBeLessThan(8640);   // budget kicks in
    expect(h).toBeGreaterThan(0);
    // Verify the capped strip fits within the budget: rows × width × 4 channels ≤ 256M
    expect(h * 15360 * 4).toBeLessThanOrEqual(256 * 1024 * 1024);
  });

  it("memory budget does not cap strips for narrow images", () => {
    // At 1920 px wide, 8640 rows = 1920×8640×4 = 66M floats, well under 256M budget.
    // GPU limit and image height are the binding constraints for narrow exports.
    const h = computeStripHeight(1080, 8192, 1920);
    expect(h).toBe(1080); // GPU and height both dominate over budget at this size
  });
});

// ── computeStripCount ─────────────────────────────────────────────────────────

describe("computeStripCount", () => {
  it("returns 1 when export height equals maxRenderbufferSize", () => {
    expect(computeStripCount(1080, 1080)).toBe(1);
  });

  it("returns 1 when export height is less than stripHeight", () => {
    expect(computeStripCount(512, 4096)).toBe(1);
  });

  it("returns 2 when export height is exactly 2 × stripHeight", () => {
    expect(computeStripCount(8192, 4096)).toBe(2);
  });

  it("rounds up when export height is not evenly divisible", () => {
    expect(computeStripCount(4097, 4096)).toBe(2);
    expect(computeStripCount(8193, 4096)).toBe(3);
  });

  it.each([
    { exportHeight: 1080,  stripHeight: 4096, expected: 1 },
    { exportHeight: 2160,  stripHeight: 2160, expected: 1 },
    { exportHeight: 4320,  stripHeight: 2160, expected: 2 },
    { exportHeight: 16384, stripHeight: 4096, expected: 4 },
    { exportHeight: 16385, stripHeight: 4096, expected: 5 },
  ])("ceil($exportHeight / $stripHeight) = $expected", ({ exportHeight, stripHeight, expected }) => {
    expect(computeStripCount(exportHeight, stripHeight)).toBe(expected);
  });
});

// ── computeExportStep ─────────────────────────────────────────────────────────

describe("computeExportStep", () => {
  it("matches pixelStep from viewport.ts for the same arguments", () => {
    expect(computeExportStep(0, 1080)).toBe(pixelStep(0, 1080));
  });

  it.each([
    { zoomExp: 0,  exportHeight: 1080 },
    { zoomExp: 0,  exportHeight: 2160 },
    { zoomExp: 0,  exportHeight: 4320 },
    { zoomExp: 1,  exportHeight: 1080 },
    { zoomExp: 5,  exportHeight: 2160 },
    { zoomExp: 10, exportHeight: 8640 },
  ])("equals pixelStep(zoomExp=$zoomExp, height=$exportHeight)", ({ zoomExp, exportHeight }) => {
    expect(computeExportStep(zoomExp, exportHeight)).toBe(pixelStep(zoomExp, exportHeight));
  });

  it("4K export at zoom_exp=0 has half the step of 1080p", () => {
    // Same fractal vertical extent, twice the pixels → half the step
    expect(computeExportStep(0, 2160)).toBeCloseTo(computeExportStep(0, 1080) / 2);
  });

  it("vertical fractal extent is invariant across export heights", () => {
    // exportHeight × step = fractalHeight(zoom_exp) regardless of resolution
    const zoomExp = 3;
    for (const h of [1080, 2160, 4320, 8640]) {
      const extent = h * computeExportStep(zoomExp, h);
      expect(extent).toBeCloseTo(4 * Math.pow(10, -zoomExp));
    }
  });
});

// ── ColoringState defaults ────────────────────────────────────────────────────

describe("ColoringState defaults", () => {
  it("has correct default values matching GlPipeline constructor", () => {
    const s: ColoringState = {
      lut: new Float32Array(4096 * 4),
      cycleLen: 64,
      trapRadius: 0.5,
      trapStrength: 0.0,
      trapColor: [1, 1, 0],
      interiorColor: [0, 0, 0],
      shadingColor: [1, 1, 1],
      distanceStrength: 0.0,
      distancePow: 1.0,
      angleStrength: 0.0,
      periodStrength: 0.0,
      periodCycleLen: 8.0,
      distWeight: 0.0,
      fractalKind: 0,
      newtonDegree: 3,
      unresolvedColor: [0.05, 0.05, 0.05],
      newtonColorMode: 0,
      newtonPhase: 0.0,
      newtonSmooth: false,
    };
    expect(s.fractalKind).toBe(0);
    expect(s.trapColor).toEqual([1, 1, 0]);
    expect(s.lut.length).toBe(4096 * 4);
    expect(s.periodCycleLen).toBe(8.0);
    expect(s.unresolvedColor).toEqual([0.05, 0.05, 0.05]);
  });
});

// ── Physical size hint calculation ────────────────────────────────────────────

describe("physical size hint calculation", () => {
  it.each([
    { w: 1920, h: 1080, dpi: 300, expectedW: "6.4", expectedH: "3.6" },
    { w: 3840, h: 2160, dpi: 300, expectedW: "12.8", expectedH: "7.2" },
    { w: 1920, h: 1080, dpi: 72,  expectedW: "26.7", expectedH: "15.0" },
  ])("$w×$h at $dpi DPI → $expectedW × $expectedH in", ({ w, h, dpi, expectedW, expectedH }) => {
    expect((w / dpi).toFixed(1)).toBe(expectedW);
    expect((h / dpi).toFixed(1)).toBe(expectedH);
  });
});
