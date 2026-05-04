import { describe, it, expect } from "vitest";
import { computeStripHeight, computeStripCount, computeExportStep } from "../export.ts";
import { pixelStep } from "../viewport.ts";

// ── computeStripHeight ────────────────────────────────────────────────────────

describe("computeStripHeight", () => {
  it.each([
    { exportHeight: 1080, maxRenderbufferSize: 8192, expected: 1080 },
    { exportHeight: 4320, maxRenderbufferSize: 4096, expected: 4096 },
    { exportHeight: 4096, maxRenderbufferSize: 4096, expected: 4096 },
    { exportHeight: 100,  maxRenderbufferSize: 4096, expected: 100  },
  ])("min($exportHeight, $maxRenderbufferSize) = $expected", ({ exportHeight, maxRenderbufferSize, expected }) => {
    expect(computeStripHeight(exportHeight, maxRenderbufferSize)).toBe(expected);
  });

  it("returns 1 when export height is 1", () => {
    expect(computeStripHeight(1, 4096)).toBe(1);
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
