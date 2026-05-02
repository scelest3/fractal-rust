import { describe, it, expect } from "vitest";
import {
  newtonParamsZ3,
  NEWTON_PRESETS,
  serializeNewtonState,
  deserializeNewtonState,
  newtonParamsFromPreset,
  type NewtonParams,
} from "../newton.ts";

describe("newtonParamsZ3", () => {
  it("returns degree 3", () => {
    expect(newtonParamsZ3().degree).toBe(3);
  });

  it("coeffs encode z³ − 1", () => {
    const { coeffs } = newtonParamsZ3();
    expect(coeffs.length).toBe(11);
    expect(coeffs[0]).toBe(-1);
    expect(coeffs[3]).toBe(1);
    expect(coeffs[1]).toBe(0);
    expect(coeffs[2]).toBe(0);
  });

  it("rootsRe and rootsIm have degree elements", () => {
    const p = newtonParamsZ3();
    expect(p.rootsRe.length).toBe(3);
    expect(p.rootsIm.length).toBe(3);
  });

  it("roots are sorted by arg ascending", () => {
    const { rootsRe, rootsIm, degree } = newtonParamsZ3();
    const args = Array.from({ length: degree }, (_, i) =>
      Math.atan2(rootsIm[i], rootsRe[i]),
    );
    for (let i = 0; i + 1 < args.length; i++) {
      expect(args[i]).toBeLessThanOrEqual(args[i + 1]);
    }
  });

  it("roots are within 1e-10 of exact z³−1 values", () => {
    const sqrt3_2 = Math.sqrt(3) / 2;
    const expected = [
      { re: -0.5,  im: -sqrt3_2 },
      { re:  1.0,  im:  0.0     },
      { re: -0.5,  im:  sqrt3_2 },
    ];
    const { rootsRe, rootsIm } = newtonParamsZ3();
    for (let i = 0; i < 3; i++) {
      const dist = Math.hypot(rootsRe[i] - expected[i].re, rootsIm[i] - expected[i].im);
      expect(dist).toBeLessThan(1e-10);
    }
  });
});

// ── NEWTON_PRESETS ────────────────────────────────────────────────────────────

describe("NEWTON_PRESETS", () => {
  it("contains exactly 5 presets", () => {
    expect(NEWTON_PRESETS.length).toBe(5);
  });

  it("first preset is z3-1", () => {
    expect(NEWTON_PRESETS[0].name).toBe("z3-1");
  });

  it("all presets have degree 1–10", () => {
    for (const p of NEWTON_PRESETS) {
      expect(p.degree).toBeGreaterThanOrEqual(1);
      expect(p.degree).toBeLessThanOrEqual(10);
    }
  });

  it("z4-z has non-zero z¹ coefficient", () => {
    const p = NEWTON_PRESETS.find(p => p.name === "z4-z")!;
    expect(p.coeffs[1]).not.toBe(0);
  });
});

// ── serializeNewtonState ──────────────────────────────────────────────────────

describe("serializeNewtonState", () => {
  it("emits preset=z3-1 for z³−1 default params", () => {
    const s = serializeNewtonState(newtonParamsZ3());
    expect(s).toContain("preset=z3-1");
    expect(s).not.toContain("coeffs=");
  });

  it("emits preset= for each named preset", () => {
    for (const p of NEWTON_PRESETS) {
      const params = newtonParamsFromPreset(p.name)!;
      const s = serializeNewtonState(params);
      expect(s).toContain(`preset=${p.name}`);
    }
  });

  it("emits coeffs= for a custom polynomial", () => {
    // z² + 1: coeffs [1, 0, 1] ascending
    const coeffs = new Float64Array(11);
    coeffs[0] = 1; coeffs[2] = 1;
    const params: NewtonParams = {
      degree: 2, coeffs,
      rootsRe: new Float64Array([0, 0]),
      rootsIm: new Float64Array([-1, 1]),
    };
    const s = serializeNewtonState(params);
    expect(s).toContain("coeffs=");
    expect(s).not.toContain("preset=");
  });

  it("omits trailing zeros in coeffs", () => {
    // z³ − 1 entered manually (not matched as preset — same coefficients, so it IS a preset)
    // Instead use z² + z − 2: coeffs [-2, 1, 1]
    const coeffs = new Float64Array(11);
    coeffs[0] = -2; coeffs[1] = 1; coeffs[2] = 1;
    const params: NewtonParams = {
      degree: 2, coeffs,
      rootsRe: new Float64Array([0, 0]),
      rootsIm: new Float64Array([0, 0]),
    };
    const s = serializeNewtonState(params);
    // Should be coeffs=-2,1,1 — no trailing zeros
    expect(s).toBe("f=newton&coeffs=-2,1,1");
  });

  it("canonicalizes custom coeffs matching a preset to preset form", () => {
    // Manually build z³−1 coefficients — should match preset
    const coeffs = new Float64Array(11);
    coeffs[0] = -1; coeffs[3] = 1;
    const params: NewtonParams = {
      degree: 3, coeffs,
      rootsRe: new Float64Array(3),
      rootsIm: new Float64Array(3),
    };
    const s = serializeNewtonState(params);
    expect(s).toContain("preset=z3-1");
  });
});

// ── deserializeNewtonState ────────────────────────────────────────────────────

describe("deserializeNewtonState", () => {
  it("returns null for non-Newton fragment", () => {
    expect(deserializeNewtonState("f=mandelbrot")).toBeNull();
    expect(deserializeNewtonState("")).toBeNull();
  });

  it("resolves preset=z3-1 to z³−1 params", () => {
    const p = deserializeNewtonState("f=newton&preset=z3-1")!;
    expect(p).not.toBeNull();
    expect(p.degree).toBe(3);
    expect(p.coeffs[0]).toBe(-1);
    expect(p.coeffs[3]).toBe(1);
  });

  it("resolves preset= for each named preset", () => {
    for (const preset of NEWTON_PRESETS) {
      const p = deserializeNewtonState(`f=newton&preset=${preset.name}`)!;
      expect(p).not.toBeNull();
      expect(p.degree).toBe(preset.degree);
    }
  });

  it("parses coeffs= correctly", () => {
    const p = deserializeNewtonState("f=newton&coeffs=-1,0,0,1")!;
    expect(p).not.toBeNull();
    expect(p.degree).toBe(3);
    expect(p.coeffs[0]).toBe(-1);
    expect(p.coeffs[3]).toBe(1);
  });

  it("parses coeffs= with non-zero first coefficient", () => {
    // z² + 1: coeffs=1,0,1
    const p = deserializeNewtonState("f=newton&coeffs=1,0,1")!;
    expect(p).not.toBeNull();
    expect(p.degree).toBe(2);
    expect(p.coeffs[0]).toBe(1);
    expect(p.coeffs[2]).toBe(1);
  });

  it("returns null for unknown preset name", () => {
    expect(deserializeNewtonState("f=newton&preset=bogus")).toBeNull();
  });
});

// ── round-trips ───────────────────────────────────────────────────────────────

describe("serialize/deserialize round-trips", () => {
  it("round-trips all 5 named presets", () => {
    for (const preset of NEWTON_PRESETS) {
      const original = newtonParamsFromPreset(preset.name)!;
      const fragment = serializeNewtonState(original);
      const restored = deserializeNewtonState(fragment)!;
      expect(restored).not.toBeNull();
      expect(restored.degree).toBe(original.degree);
      for (let i = 0; i <= original.degree; i++) {
        expect(restored.coeffs[i]).toBeCloseTo(original.coeffs[i], 10);
      }
    }
  });

  it("round-trips a custom polynomial", () => {
    const coeffs = new Float64Array(11);
    coeffs[0] = -2; coeffs[1] = 1; coeffs[2] = 1;
    const original: NewtonParams = {
      degree: 2, coeffs,
      rootsRe: new Float64Array(2),
      rootsIm: new Float64Array(2),
    };
    const fragment = serializeNewtonState(original);
    const restored = deserializeNewtonState(fragment)!;
    expect(restored.degree).toBe(2);
    expect(restored.coeffs[0]).toBe(-2);
    expect(restored.coeffs[1]).toBe(1);
    expect(restored.coeffs[2]).toBe(1);
  });
});
