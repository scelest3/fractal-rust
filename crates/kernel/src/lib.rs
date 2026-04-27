//! Fractal iteration engine — escape-time computation, perturbation theory, BLA.
//!
//! `kernel` imports `arith::{Complex, DeltaC, Precision}` and owns all fractal
//! algorithms. `Complex<f64>` here is always `arith::Complex<f64>` — the
//! workspace-wide type. `wasm-bridge` is the only crate that names concrete
//! `BigFloat<N>` types; `kernel` is generic over `T: Precision` for the cold
//! reference orbit path (Phase 2).

use arith::{Complex, Precision};

// ── Output types ──────────────────────────────────────────────────────────────

/// Result of one `IterationMap::step`.
#[derive(Copy, Clone, Debug)]
pub struct StepResult {
    /// The new z value after one iteration.
    pub z: Complex<f64>,
    /// True if `|z|² > escape_radius_sq` after this step.
    pub escaped: bool,
}

/// Escape-time result for one pixel. Consumed by the `coloring` crate.
#[derive(Copy, Clone, Debug)]
pub enum EscapeResult {
    /// Orbit escaped within `max_iter` steps.
    Escaped {
        /// Iteration index at which escape was detected (0-indexed).
        iter: u32,
        /// Smooth iteration count: `(iter + 1) − ln(ln(|z|)) / ln(2)`.
        smooth_t: f64,
        /// Minimum `|z|` over the orbit `{z₁, z₂, …}`. For orbit traps.
        orbit_min_r: f64,
        /// z at which `orbit_min_r` was attained.
        orbit_min_z: Complex<f64>,
        /// `arg(z)` at escape (radians). For angle-domain coloring.
        angle_final: f64,
    },
    /// Orbit did not escape within `max_iter` steps (interior or convergent).
    Interior {
        /// Minimum `|z|` over the orbit.
        orbit_min_r: f64,
        /// z at which `orbit_min_r` was attained.
        orbit_min_z: Complex<f64>,
    },
    /// Pixel requires glitch correction; must not be converted to `TilePixel`
    /// directly. Only produced by the perturbation path (Phase 2).
    Glitched,
}

/// Raw per-pixel data written into the tile ring slot.
/// Four f32 channels map to the RGBA32F WebGL tile texture:
///   r = smooth_t, g = orbit_min_r, b = angle, a = escaped (1.0 / 0.0).
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct TilePixel {
    pub smooth_t: f32,
    pub orbit_min_r: f32,
    pub angle: f32,
    pub escaped: f32,
}

impl From<EscapeResult> for TilePixel {
    fn from(r: EscapeResult) -> Self {
        match r {
            EscapeResult::Escaped { smooth_t, orbit_min_r, angle_final, .. } => Self {
                smooth_t: smooth_t as f32,
                orbit_min_r: orbit_min_r as f32,
                angle: angle_final as f32,
                escaped: 1.0,
            },
            EscapeResult::Interior { orbit_min_r, orbit_min_z } => Self {
                smooth_t: 0.0,
                orbit_min_r: orbit_min_r as f32,
                angle: (orbit_min_z.im.atan2(orbit_min_z.re)) as f32,
                escaped: 0.0,
            },
            EscapeResult::Glitched => {
                panic!("Glitched pixel must not be converted to TilePixel — apply glitch correction first")
            }
        }
    }
}

/// Which fractal map to iterate. Extended in Phase 3.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FractalKind {
    Mandelbrot,
    Newton,
}

// ── IterationMap trait ────────────────────────────────────────────────────────

/// One step of a fractal recurrence z ← f(z, c).
///
/// Implemented by concrete map structs (`MandelbrotMap`, etc.). `kernel`
/// functions are generic over `M: IterationMap`, monomorphized at each call site.
pub trait IterationMap {
    /// Advance one step: returns the new z and whether it escaped.
    fn step(&self, z: Complex<f64>, c: Complex<f64>) -> StepResult;

    /// Escape-radius squared. `|z|² > escape_radius_sq()` triggers escape.
    fn escape_radius_sq(&self) -> f64;

    /// Convergence test for Newton / Nova maps. Default: always false.
    fn converged(&self, _z: Complex<f64>, _z_prev: Complex<f64>) -> bool {
        false
    }
}

// ── MandelbrotMap ─────────────────────────────────────────────────────────────

/// Standard Mandelbrot / filled-Julia iteration: z ← z² + c.
pub struct MandelbrotMap;

impl IterationMap for MandelbrotMap {
    #[inline(always)]
    fn step(&self, z: Complex<f64>, c: Complex<f64>) -> StepResult {
        let z_next = z * z + c;
        StepResult {
            escaped: z_next.norm_sqr() > self.escape_radius_sq(),
            z: z_next,
        }
    }

    #[inline(always)]
    fn escape_radius_sq(&self) -> f64 {
        4.0
    }
}

// ── PerturbationSupport trait ─────────────────────────────────────────────────

/// Extension of `IterationMap` for fractal types that support deep-zoom via
/// perturbation theory.
///
/// The reference orbit is computed once in high precision T; per-pixel work
/// uses cheap f64 arithmetic on small perturbation deltas δz.
pub trait PerturbationSupport: IterationMap {
    /// One entry in the reference orbit buffer (downcast to f64).
    /// `Complex<f64>` for most maps; `(Complex<f64>, Complex<f64>)` for Phoenix.
    type RefOrbitEntry: Copy;

    /// Per-step auxiliary state for the perturbation recurrence.
    /// `()` for Mandelbrot; sign bits for Burning Ship.
    type RefState: Copy;

    /// One step of the reference orbit in high-precision T.
    fn ref_step<T: Precision>(&self, z: Complex<T>, c: Complex<T>) -> Complex<T>;

    /// One step of the perturbation recurrence (always f64).
    ///
    /// `dz` is the current perturbation delta, `dc` the per-pixel ΔC,
    /// `ref_z` is Z_n from the reference orbit at step n.
    fn perturb_step(
        &self,
        dz: Complex<f64>,
        dc: Complex<f64>,
        ref_z: Complex<f64>,
    ) -> Complex<f64>;

    /// Glitch threshold: a pixel is glitched when `|dz_n| > threshold × |Z_n|`.
    fn glitch_threshold(&self) -> f64;

    /// Extract per-step reference state from a reference orbit value.
    fn ref_state(&self, z: Complex<f64>) -> Self::RefState;
}

impl PerturbationSupport for MandelbrotMap {
    type RefOrbitEntry = Complex<f64>;
    type RefState = ();

    #[inline(always)]
    fn ref_step<T: Precision>(&self, z: Complex<T>, c: Complex<T>) -> Complex<T> {
        z * z + c
    }

    #[inline(always)]
    fn perturb_step(
        &self,
        dz: Complex<f64>,
        dc: Complex<f64>,
        ref_z: Complex<f64>,
    ) -> Complex<f64> {
        // δz_{n+1} = 2·Z_n·δz_n + δz_n² + δc
        2.0 * ref_z * dz + dz * dz + dc
    }

    #[inline(always)]
    fn glitch_threshold(&self) -> f64 {
        1e-3
    }

    #[inline(always)]
    fn ref_state(&self, _z: Complex<f64>) -> Self::RefState {}
}

// ── compute_ref_orbit ─────────────────────────────────────────────────────────

/// Compute the reference orbit in precision T, storing downcast f64 entries.
///
/// Returns the number of entries written to `out` (≤ `max_iter`, ≤ `out.len()`).
/// `out[n]` = Z_n (the value before the n-th step). The loop stops early if the
/// reference point escapes; the escaped value is not stored.
///
/// # Arguments
/// * `map`      — controls the high-precision step and escape radius
/// * `c`        — reference point in precision T
/// * `max_iter` — upper bound on orbit length
/// * `out`      — caller-supplied buffer; length sets an independent upper bound
pub fn compute_ref_orbit<T: Precision, M: PerturbationSupport>(
    map: &M,
    c: Complex<T>,
    max_iter: u32,
    out: &mut [Complex<f64>],
) -> usize {
    let mut z = Complex::<T>::zero();
    let limit = (max_iter as usize).min(out.len());
    for (i, slot) in out[..limit].iter_mut().enumerate() {
        *slot = Complex::new(z.re.to_f64_lossy(), z.im.to_f64_lossy());
        let z_next = map.ref_step(z, c);
        if z_next.norm_sqr().to_f64_lossy() > map.escape_radius_sq() {
            return i + 1;
        }
        z = z_next;
    }
    limit
}

// ── escape_time ───────────────────────────────────────────────────────────────

/// Compute the escape-time result for one pixel using plain f64 iteration.
///
/// Orbit minimum is tracked over `{z₁, z₂, …}` (z₀ = 0 is excluded so the
/// orbit trap is not trivially satisfied by the fixed starting point).
///
/// Smooth-coloring formula (ARCHITECTURE.md §9.2):
/// ```text
/// smooth_t = (iter + 1) − ln(ln(|z|)) / ln(2)
/// ```
pub fn escape_time<M: IterationMap>(map: &M, c: Complex<f64>, max_iter: u32) -> EscapeResult {
    let mut z = Complex::<f64>::zero();
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = Complex::<f64>::zero();

    for iter in 0..max_iter {
        let step = map.step(z, c);
        let z_new = step.z;

        // Track orbit min over {z₁, z₂, …} — z₀ = 0 excluded.
        let r = z_new.norm_sqr().sqrt();
        if r < orbit_min_r {
            orbit_min_r = r;
            orbit_min_z = z_new;
        }

        let z_prev = z;
        z = z_new;

        if step.escaped {
            let norm = z.norm_sqr().sqrt();
            let smooth_t =
                (iter as f64 + 1.0) - (norm.ln().ln() / core::f64::consts::LN_2);
            return EscapeResult::Escaped {
                iter,
                smooth_t,
                orbit_min_r,
                orbit_min_z,
                angle_final: z.im.atan2(z.re),
            };
        }

        if map.converged(z, z_prev) {
            return EscapeResult::Interior { orbit_min_r, orbit_min_z };
        }
    }

    EscapeResult::Interior { orbit_min_r, orbit_min_z }
}

// ── render_tile_escape ────────────────────────────────────────────────────────

/// Fill `out` with escape-time results for each coordinate in `coords`.
///
/// `coords` and `out` must have the same length (asserted in debug builds).
/// Each entry in `coords` is the complex constant `c` for one pixel.
pub fn render_tile_escape<M: IterationMap>(
    map: &M,
    coords: &[Complex<f64>],
    max_iter: u32,
    out: &mut [TilePixel],
) {
    debug_assert_eq!(coords.len(), out.len());
    for (c, slot) in coords.iter().zip(out.iter_mut()) {
        *slot = TilePixel::from(escape_time(map, *c, max_iter));
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arith::{Complex, F64x2};

    const MAP: MandelbrotMap = MandelbrotMap;

    // ── MandelbrotMap::step ───────────────────────────────────────────────────

    #[test]
    fn step_zero_plus_c_gives_c() {
        let z = Complex::new(0.0_f64, 0.0);
        let c = Complex::new(0.5_f64, 0.0);
        let s = MAP.step(z, c);
        assert_eq!(s.z, c);
        assert!(!s.escaped);
    }

    #[test]
    fn step_escapes_when_norm_sq_exceeds_four() {
        // z = 3 + 0i, c = 0: z² = 9, |z²|² = 81 > 4
        let z = Complex::new(3.0_f64, 0.0);
        let c = Complex::new(0.0_f64, 0.0);
        let s = MAP.step(z, c);
        assert!(s.escaped);
        assert_eq!(s.z.re, 9.0);
    }

    #[test]
    fn step_does_not_escape_at_exactly_four() {
        // z = 0, c = 2: z_next = 2, norm_sq = 4, NOT > 4 (strictly >)
        let z = Complex::new(0.0_f64, 0.0);
        let c = Complex::new(2.0_f64, 0.0);
        let s = MAP.step(z, c);
        assert!(!s.escaped, "norm_sq == 4.0 should not escape (strictly >)");
    }

    // ── EscapeResult enum ─────────────────────────────────────────────────────

    #[test]
    fn tile_pixel_from_escaped_has_escaped_one() {
        let r = EscapeResult::Escaped {
            iter: 5,
            smooth_t: 1.5,
            orbit_min_r: 0.3,
            orbit_min_z: Complex::new(0.1_f64, 0.2),
            angle_final: 0.7,
        };
        let p = TilePixel::from(r);
        assert_eq!(p.escaped, 1.0);
        assert!((p.smooth_t - 1.5_f32).abs() < 1e-6);
        assert!((p.orbit_min_r - 0.3_f32).abs() < 1e-6);
        assert!((p.angle - 0.7_f32).abs() < 1e-6);
    }

    #[test]
    fn tile_pixel_from_interior_has_escaped_zero() {
        let r = EscapeResult::Interior {
            orbit_min_r: 0.4,
            orbit_min_z: Complex::new(0.0_f64, 0.0),
        };
        let p = TilePixel::from(r);
        assert_eq!(p.escaped, 0.0);
        assert_eq!(p.smooth_t, 0.0);
        assert!((p.orbit_min_r - 0.4_f32).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Glitched pixel")]
    fn tile_pixel_from_glitched_panics() {
        let _ = TilePixel::from(EscapeResult::Glitched);
    }

    // ── escape_time: interior points ──────────────────────────────────────────

    #[test]
    fn origin_is_interior() {
        let r = escape_time(&MAP, Complex::new(0.0_f64, 0.0), 100);
        assert!(matches!(r, EscapeResult::Interior { .. }));
    }

    #[test]
    fn minus_half_is_interior() {
        let r = escape_time(&MAP, Complex::new(-0.5_f64, 0.0), 1000);
        assert!(matches!(r, EscapeResult::Interior { .. }));
    }

    #[test]
    fn interior_orbit_min_r_is_non_negative() {
        let r = escape_time(&MAP, Complex::new(-0.5_f64, 0.0), 100);
        if let EscapeResult::Interior { orbit_min_r, .. } = r {
            assert!(orbit_min_r >= 0.0);
        } else {
            panic!("expected Interior");
        }
    }

    // ── escape_time: exterior points ──────────────────────────────────────────

    #[test]
    fn c_equals_2_escapes_at_iter_1() {
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        if let EscapeResult::Escaped { iter, .. } = r {
            assert_eq!(iter, 1, "c=2 must escape at iteration 1");
        } else {
            panic!("c=2 must escape");
        }
    }

    #[test]
    fn c_equals_2_exit_angle_is_zero() {
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        if let EscapeResult::Escaped { angle_final, .. } = r {
            assert!(angle_final.abs() < 1e-10, "exit angle should be ~0 for z=6+0i");
        } else {
            panic!("c=2 must escape");
        }
    }

    #[test]
    fn smooth_t_matches_formula_for_c_equals_2() {
        // iter=1, z_escape = 6+0i, |z| = 6.0
        // smooth_t = 2 - ln(ln(6)) / ln(2)
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        if let EscapeResult::Escaped { smooth_t, .. } = r {
            let expected = 2.0 - f64::ln(f64::ln(6.0)) / core::f64::consts::LN_2;
            assert!(
                (smooth_t - expected).abs() < 1e-10,
                "smooth_t = {smooth_t}, expected ≈ {expected}"
            );
        } else {
            panic!("c=2 must escape");
        }
    }

    #[test]
    fn far_exterior_escapes_at_iter_0() {
        let r = escape_time(&MAP, Complex::new(10.0_f64, 0.0), 1000);
        if let EscapeResult::Escaped { iter, .. } = r {
            assert_eq!(iter, 0);
        } else {
            panic!("c=10 must escape");
        }
    }

    #[test]
    fn smooth_t_is_finite_for_escaped_pixel() {
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        if let EscapeResult::Escaped { smooth_t, .. } = r {
            assert!(smooth_t.is_finite(), "smooth_t must be finite");
        } else {
            panic!("c=2 must escape");
        }
    }

    #[test]
    fn orbit_min_r_tracks_minimum() {
        let r = escape_time(&MAP, Complex::new(0.5_f64, 0.0), 1000);
        if let EscapeResult::Escaped { orbit_min_r, .. } = r {
            assert!(orbit_min_r >= 0.0);
            assert!(orbit_min_r < f64::MAX);
        } else {
            panic!("c=0.5 must escape");
        }
    }

    // ── TilePixel ─────────────────────────────────────────────────────────────

    #[test]
    fn tile_pixel_is_16_bytes_repr_c() {
        assert_eq!(core::mem::size_of::<TilePixel>(), 16);
    }

    // ── render_tile_escape ────────────────────────────────────────────────────

    #[test]
    fn render_tile_mixed_interior_and_escaped() {
        let coords: Vec<Complex<f64>> = (-2..=1)
            .flat_map(|re| (-2..=1).map(move |im| Complex::new(re as f64, im as f64)))
            .collect();
        let mut out = vec![TilePixel::default(); coords.len()];
        render_tile_escape(&MAP, &coords, 100, &mut out);

        let escaped_count = out.iter().filter(|p| p.escaped == 1.0).count();
        let interior_count = out.iter().filter(|p| p.escaped == 0.0).count();
        assert!(escaped_count > 0, "expected at least one escaped pixel");
        assert!(interior_count > 0, "expected at least one interior pixel");
    }

    #[test]
    fn render_tile_all_escaped_far_exterior() {
        let coords: Vec<Complex<f64>> =
            (0..16).map(|i| Complex::new(10.0 + i as f64, 0.0)).collect();
        let mut out = vec![TilePixel::default(); coords.len()];
        render_tile_escape(&MAP, &coords, 100, &mut out);
        assert!(out.iter().all(|p| p.escaped == 1.0));
    }

    #[test]
    fn render_tile_output_length_matches_coords() {
        let coords = vec![Complex::new(0.0_f64, 0.0); 64];
        let mut out = vec![TilePixel::default(); 64];
        render_tile_escape(&MAP, &coords, 50, &mut out);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn render_tile_interior_smooth_t_is_zero() {
        let coords = vec![Complex::new(0.0_f64, 0.0)];
        let mut out = vec![TilePixel::default(); 1];
        render_tile_escape(&MAP, &coords, 100, &mut out);
        assert_eq!(out[0].escaped, 0.0);
        assert_eq!(out[0].smooth_t, 0.0);
    }

    #[test]
    fn render_tile_escaped_smooth_t_is_finite() {
        let coords = vec![Complex::new(2.0_f64, 0.0)];
        let mut out = vec![TilePixel::default(); 1];
        render_tile_escape(&MAP, &coords, 100, &mut out);
        assert_eq!(out[0].escaped, 1.0);
        assert!(out[0].smooth_t.is_finite());
    }

    // ── PerturbationSupport: MandelbrotMap ────────────────────────────────────

    #[test]
    fn mandelbrot_glitch_threshold_is_1e_3() {
        assert_eq!(MAP.glitch_threshold(), 1e-3);
    }

    #[test]
    fn mandelbrot_perturb_step_at_zero_ref() {
        // ref_z = 0, dz = 0, dc = (0.1, 0): dz_next = 0 + 0 + dc = (0.1, 0)
        let dz = Complex::new(0.0_f64, 0.0);
        let dc = Complex::new(0.1_f64, 0.0);
        let ref_z = Complex::new(0.0_f64, 0.0);
        let result = MAP.perturb_step(dz, dc, ref_z);
        assert!((result.re - 0.1).abs() < 1e-15);
        assert!(result.im.abs() < 1e-15);
    }

    #[test]
    fn mandelbrot_perturb_step_matches_formula() {
        // δz_{n+1} = 2·Z_n·δz_n + δz_n² + δc
        // Z = (1, 0.5), dz = (0.01, 0.02), dc = (0.001, 0.002)
        let ref_z = Complex::new(1.0_f64, 0.5);
        let dz = Complex::new(0.01_f64, 0.02);
        let dc = Complex::new(0.001_f64, 0.002);

        let two_z_dz = 2.0 * ref_z * dz;
        let dz_sq = dz * dz;
        let expected = two_z_dz + dz_sq + dc;
        let result = MAP.perturb_step(dz, dc, ref_z);

        assert!((result.re - expected.re).abs() < 1e-14);
        assert!((result.im - expected.im).abs() < 1e-14);
    }

    // ── compute_ref_orbit ─────────────────────────────────────────────────────

    #[test]
    fn ref_orbit_at_origin_never_escapes() {
        // c = 0: z stays at 0, never escapes
        let c = Complex::new(0.0_f64, 0.0);
        let mut out = vec![Complex::<f64>::zero(); 10];
        let len = compute_ref_orbit(&MAP, c, 10, &mut out);
        assert_eq!(len, 10);
        assert!(out.iter().all(|z| z.re == 0.0 && z.im == 0.0));
    }

    #[test]
    fn ref_orbit_c_equals_2_escapes_after_two_entries() {
        // z_0=0, z_1=2 (not escaped, norm_sq=4 not > 4), z_2=6 (escaped)
        // out = [0, 2], returns 2
        let c = Complex::new(2.0_f64, 0.0);
        let mut out = vec![Complex::<f64>::zero(); 20];
        let len = compute_ref_orbit(&MAP, c, 20, &mut out);
        assert_eq!(len, 2);
        assert_eq!(out[0].re, 0.0);
        assert_eq!(out[0].im, 0.0);
        assert_eq!(out[1].re, 2.0);
        assert_eq!(out[1].im, 0.0);
    }

    #[test]
    fn ref_orbit_respects_max_iter() {
        let c = Complex::new(0.0_f64, 0.0); // never escapes
        let mut out = vec![Complex::<f64>::zero(); 100];
        let len = compute_ref_orbit(&MAP, c, 5, &mut out);
        assert_eq!(len, 5);
    }

    #[test]
    fn ref_orbit_respects_output_buffer_len() {
        let c = Complex::new(0.0_f64, 0.0);
        let mut out = vec![Complex::<f64>::zero(); 5];
        let len = compute_ref_orbit(&MAP, c, 100, &mut out);
        assert_eq!(len, 5);
    }

    #[test]
    fn ref_orbit_f64x2_matches_f64_for_simple_point() {
        // Verify generic dispatch: F64x2 and f64 give identical orbit entries
        // for a simple escaped point (c = 2 + 0i).
        let c_f64 = Complex::new(2.0_f64, 0.0);
        let c_f64x2 = Complex::new(F64x2::from_f64(2.0), F64x2::from_f64(0.0));

        let mut out_f64 = vec![Complex::<f64>::zero(); 20];
        let mut out_f64x2 = vec![Complex::<f64>::zero(); 20];

        let len_f64 = compute_ref_orbit(&MAP, c_f64, 20, &mut out_f64);
        let len_f64x2 = compute_ref_orbit(&MAP, c_f64x2, 20, &mut out_f64x2);

        assert_eq!(len_f64, len_f64x2, "orbit lengths must match");
        for (a, b) in out_f64[..len_f64].iter().zip(out_f64x2[..len_f64x2].iter()) {
            assert!(
                (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                "orbit entry mismatch: f64={a:?} f64x2={b:?}"
            );
        }
    }

    #[test]
    fn ref_orbit_interior_point_orbit_is_bounded() {
        // c = -0.5 + 0i is inside the Mandelbrot set; orbit should stay bounded
        let c = Complex::new(-0.5_f64, 0.0);
        let mut out = vec![Complex::<f64>::zero(); 200];
        let len = compute_ref_orbit(&MAP, c, 200, &mut out);
        assert_eq!(len, 200, "interior orbit should not escape");
        for z in &out[..len] {
            assert!(z.re.abs() <= 2.0 + 1e-10 && z.im.abs() <= 2.0 + 1e-10);
        }
    }
}
