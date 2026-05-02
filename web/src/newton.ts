/**
 * Newton fractal utilities — pure functions, no DOM, no WASM.
 *
 * Newton params for the hardwired z³−1 preset are expressed as closed-form
 * constants. Issue #31 replaces these with compute_roots called from a tile
 * worker once the polynomial UI is wired up.
 */

export interface NewtonParams {
  degree: number;
  /** Polynomial coefficients ascending by degree (11 elements, padded with 0). */
  coeffs: Float64Array;
  /** Root real parts, sorted by arg(root) ascending (degree elements). */
  rootsRe: Float64Array;
  /** Root imaginary parts, sorted by arg(root) ascending (degree elements). */
  rootsIm: Float64Array;
}

const SQRT3_2 = Math.sqrt(3) / 2;

/**
 * Newton params for z³ − 1.
 *
 * Roots sorted by arg ascending:
 *   index 0: −½ − i√3/2  (arg ≈ −2π/3)
 *   index 1:  1 + 0i      (arg = 0)
 *   index 2: −½ + i√3/2  (arg ≈ +2π/3)
 */
export function newtonParamsZ3(): NewtonParams {
  const coeffs = new Float64Array(11);
  coeffs[0] = -1;
  coeffs[3] = 1;

  const rootsRe = new Float64Array([-0.5, 1.0, -0.5]);
  const rootsIm = new Float64Array([-SQRT3_2, 0.0, SQRT3_2]);

  return { degree: 3, coeffs, rootsRe, rootsIm };
}
