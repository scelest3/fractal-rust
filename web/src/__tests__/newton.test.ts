import { describe, it, expect } from "vitest";
import { newtonParamsZ3 } from "../newton.ts";

describe("newtonParamsZ3", () => {
  it("returns degree 3", () => {
    expect(newtonParamsZ3().degree).toBe(3);
  });

  it("coeffs encode z³ − 1", () => {
    const { coeffs } = newtonParamsZ3();
    expect(coeffs.length).toBe(11);
    expect(coeffs[0]).toBe(-1);   // constant term
    expect(coeffs[3]).toBe(1);    // z³ term
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
