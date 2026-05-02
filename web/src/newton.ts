/**
 * Newton fractal utilities — pure functions, no DOM, no WASM.
 *
 * Covers: preset definitions, Newton params construction, URL serialization
 * and deserialization. All functions are pure and vitest-testable.
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

// ── Presets ───────────────────────────────────────────────────────────────────

interface Preset {
  name: string;
  label: string;
  degree: number;
  coeffs: readonly number[]; // length 11, ascending by degree
}

export const NEWTON_PRESETS: readonly Preset[] = [
  { name: "z3-1", label: "z³−1",  degree: 3, coeffs: [-1, 0,  0, 1, 0, 0, 0, 0, 0, 0, 0] },
  { name: "z4-1", label: "z⁴−1",  degree: 4, coeffs: [-1, 0,  0, 0, 1, 0, 0, 0, 0, 0, 0] },
  { name: "z5-1", label: "z⁵−1",  degree: 5, coeffs: [-1, 0,  0, 0, 0, 1, 0, 0, 0, 0, 0] },
  { name: "z6-1", label: "z⁶−1",  degree: 6, coeffs: [-1, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0] },
  { name: "z4-z", label: "z⁴−z",  degree: 4, coeffs: [ 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0] },
] as const;

const DEFAULT_PRESET_NAME = "z3-1";

/**
 * Build a NewtonParams from a named preset. Returns null if the name is unknown.
 * Roots are left zeroed — caller must populate via compute_roots.
 */
export function newtonParamsFromPreset(name: string): NewtonParams | null {
  const preset = NEWTON_PRESETS.find(p => p.name === name);
  if (!preset) return null;
  return {
    degree: preset.degree,
    coeffs: new Float64Array(preset.coeffs),
    rootsRe: new Float64Array(preset.degree),
    rootsIm: new Float64Array(preset.degree),
  };
}

/** Returns the name of a matching preset, or null if no preset matches. */
function matchPreset(coeffs: Float64Array, degree: number): string | null {
  for (const p of NEWTON_PRESETS) {
    if (p.degree !== degree) continue;
    if (p.coeffs.every((c, i) => c === coeffs[i])) return p.name;
  }
  return null;
}

// ── URL serialization ─────────────────────────────────────────────────────────

/**
 * Serialize Newton state to a URL fragment key=value string (no leading `#`).
 *
 * Emits `f=newton&preset=<name>` for named presets; canonicalizes custom
 * coefficients that match a preset to the preset form. Custom polynomials
 * emit `f=newton&coeffs=<c0>,<c1>,…,<cn>` (ascending by degree, trailing
 * zeros omitted).
 */
export function serializeNewtonState(params: NewtonParams): string {
  const presetName = matchPreset(params.coeffs, params.degree);
  if (presetName) return `f=newton&preset=${presetName}`;

  // Trim trailing zeros from coefficient array.
  let last = params.degree;
  while (last > 0 && params.coeffs[last] === 0) last--;
  const trimmed = Array.from(params.coeffs.subarray(0, last + 1));
  return `f=newton&coeffs=${trimmed.join(",")}`;
}

/**
 * Parse a URL fragment string (no leading `#`) for Newton state.
 *
 * Returns null if the fragment is not Newton (`f=newton` absent or missing).
 * Roots are zeroed — caller must populate via compute_roots.
 */
export function deserializeNewtonState(fragment: string): NewtonParams | null {
  const params = new URLSearchParams(fragment);
  if (params.get("f") !== "newton") return null;

  const presetName = params.get("preset");
  if (presetName) return newtonParamsFromPreset(presetName);

  const coeffsStr = params.get("coeffs");
  if (!coeffsStr) return null;

  const parsed = coeffsStr.split(",").map(Number);
  if (parsed.some(isNaN) || parsed.length < 2) return null;

  const degree = parsed.length - 1;
  if (degree < 1 || degree > 10) return null;

  const coeffs = new Float64Array(11);
  parsed.forEach((c, i) => { coeffs[i] = c; });

  return {
    degree,
    coeffs,
    rootsRe: new Float64Array(degree),
    rootsIm: new Float64Array(degree),
  };
}

// ── Default params ────────────────────────────────────────────────────────────

const SQRT3_2 = Math.sqrt(3) / 2;

/**
 * Newton params for z³ − 1 with closed-form roots (sorted by arg ascending).
 *   index 0: −½ − i√3/2  (arg ≈ −2π/3)
 *   index 1:  1 + 0i      (arg = 0)
 *   index 2: −½ + i√3/2  (arg ≈ +2π/3)
 *
 * Used as the default before compute_roots has been called.
 */
export function newtonParamsZ3(): NewtonParams {
  const coeffs = new Float64Array(11);
  coeffs[0] = -1;
  coeffs[3] = 1;
  return {
    degree: 3,
    coeffs,
    rootsRe: new Float64Array([-0.5, 1.0, -0.5]),
    rootsIm: new Float64Array([-SQRT3_2, 0.0, SQRT3_2]),
  };
}

/** Default NewtonParams from the default preset name, with closed-form z³−1 roots. */
export function defaultNewtonParams(): NewtonParams {
  return DEFAULT_PRESET_NAME === "z3-1"
    ? newtonParamsZ3()
    : (newtonParamsFromPreset(DEFAULT_PRESET_NAME) ?? newtonParamsZ3());
}
