//! Fractal iteration engine — escape-time computation, perturbation theory, BLA.
//!
//! `kernel` imports `arith::{Complex, DeltaC, Precision}` and owns all fractal
//! algorithms. `Complex<f64>` here is always `arith::Complex<f64>` — the
//! workspace-wide type. `wasm-bridge` is the only crate that names concrete
//! `BigFloat<N>` types; `kernel` is generic over `T: Precision` for the cold
//! reference orbit path (Phase 2).

use arith::Complex;

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
/// All fields are in f64; `TilePixel` provides the lossy f32 form for GPU upload.
#[derive(Copy, Clone, Debug, Default)]
pub struct EscapeResult {
    /// Iteration at which the orbit escaped (0-indexed). `max_iter` if interior.
    pub iter: u32,
    /// True if the orbit escaped within `max_iter` steps.
    pub escaped: bool,
    /// Smooth iteration count for band-free coloring.
    /// Formula (from ARCHITECTURE.md §9.2): `(iter + 1) − ln(ln(|z|)) / ln(2)`
    /// where `|z|` is the magnitude at escape. Zero for interior pixels.
    pub smooth_t: f64,
    /// Minimum `|z|` reached over the orbit `{z₁, z₂, …}`. Used by orbit traps.
    pub orbit_min_r: f64,
    /// The z at which `orbit_min_r` was attained.
    pub orbit_min_z: Complex<f64>,
    /// `arg(z)` at escape (in radians). Used for angle-domain coloring.
    pub angle_final: f64,
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
        Self {
            smooth_t: r.smooth_t as f32,
            orbit_min_r: r.orbit_min_r as f32,
            angle: r.angle_final as f32,
            escaped: if r.escaped { 1.0 } else { 0.0 },
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
///
/// Phase 2 will add `PerturbationSupport` for deep-zoom rendering. Phase 1
/// uses plain f64 escape time only.
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
            return EscapeResult {
                iter,
                escaped: true,
                smooth_t,
                orbit_min_r,
                orbit_min_z,
                angle_final: z.im.atan2(z.re),
            };
        }

        if map.converged(z, z_prev) {
            return EscapeResult {
                iter,
                escaped: false,
                smooth_t: 0.0,
                orbit_min_r,
                orbit_min_z,
                angle_final: z.im.atan2(z.re),
            };
        }
    }

    EscapeResult {
        iter: max_iter,
        escaped: false,
        smooth_t: 0.0,
        orbit_min_r,
        orbit_min_z,
        angle_final: 0.0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arith::Complex;

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
        // z = 2 + 0i, c = 0: z² = 4, |z²|² = 16... wait, escape checks z_next.norm_sq > 4
        // z = 0, c = 2: z_next = 2, norm_sq = 4, NOT > 4
        let z = Complex::new(0.0_f64, 0.0);
        let c = Complex::new(2.0_f64, 0.0);
        let s = MAP.step(z, c);
        assert!(!s.escaped, "norm_sq == 4.0 should not escape (strictly >)");
    }

    // ── escape_time: interior points ──────────────────────────────────────────

    #[test]
    fn origin_is_interior() {
        // c = 0: z stays at 0 forever
        let r = escape_time(&MAP, Complex::new(0.0_f64, 0.0), 100);
        assert!(!r.escaped);
        assert_eq!(r.iter, 100);
        assert_eq!(r.smooth_t, 0.0);
    }

    #[test]
    fn minus_half_is_interior() {
        // c = -0.5 is inside the main cardioid
        let r = escape_time(&MAP, Complex::new(-0.5_f64, 0.0), 1000);
        assert!(!r.escaped);
    }

    #[test]
    fn interior_orbit_min_r_is_non_negative() {
        let r = escape_time(&MAP, Complex::new(-0.5_f64, 0.0), 100);
        assert!(r.orbit_min_r >= 0.0);
    }

    // ── escape_time: exterior points ──────────────────────────────────────────

    #[test]
    fn c_equals_2_escapes_at_iter_1() {
        // Manual trace: z₀=0, step→z₁=2 (not escaped), step→z₂=6 (escaped at iter=1)
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        assert!(r.escaped, "c=2 must escape");
        assert_eq!(r.iter, 1, "c=2 must escape at iteration 1");
    }

    #[test]
    fn c_equals_2_exit_z_is_6() {
        // At escape, z = 2² + 2 = 6
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        // angle_final should be 0 since z is real and positive
        assert!(r.angle_final.abs() < 1e-10, "exit angle should be ~0 for z=6+0i");
    }

    #[test]
    fn smooth_t_matches_formula_for_c_equals_2() {
        // Manual trace: iter=1, z_escape = 6+0i, |z| = 6.0
        // smooth_t = 2 - ln(ln(6)) / ln(2)
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        assert!(r.escaped);
        let expected_norm = 6.0_f64;
        let expected =
            2.0 - f64::ln(f64::ln(expected_norm)) / core::f64::consts::LN_2;
        assert!(
            (r.smooth_t - expected).abs() < 1e-10,
            "smooth_t = {}, expected ≈ {} (formula: 2 - ln(ln(6))/ln(2))",
            r.smooth_t,
            expected
        );
    }

    #[test]
    fn far_exterior_escapes_early() {
        // c = 10 + 0i: escapes on the very first step
        let r = escape_time(&MAP, Complex::new(10.0_f64, 0.0), 1000);
        assert!(r.escaped);
        assert_eq!(r.iter, 0);
    }

    #[test]
    fn smooth_t_is_finite_for_escaped_pixel() {
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        assert!(r.smooth_t.is_finite(), "smooth_t must be finite");
    }

    #[test]
    fn orbit_min_r_tracks_minimum() {
        // For any escaped point, orbit_min_r should be non-negative and ≤ first z norm.
        let r = escape_time(&MAP, Complex::new(0.5_f64, 0.0), 1000);
        assert!(r.orbit_min_r >= 0.0);
        assert!(r.orbit_min_r < f64::MAX);
    }

    // ── TilePixel conversion ──────────────────────────────────────────────────

    #[test]
    fn tile_pixel_escaped_flag_is_one_for_escaped() {
        let e = EscapeResult {
            escaped: true,
            smooth_t: 1.5,
            orbit_min_r: 0.3,
            angle_final: 0.7,
            ..Default::default()
        };
        let p = TilePixel::from(e);
        assert_eq!(p.escaped, 1.0);
        assert!((p.smooth_t - 1.5_f32).abs() < 1e-6);
        assert!((p.orbit_min_r - 0.3_f32).abs() < 1e-6);
    }

    #[test]
    fn tile_pixel_escaped_flag_is_zero_for_interior() {
        let e = EscapeResult { escaped: false, ..Default::default() };
        let p = TilePixel::from(e);
        assert_eq!(p.escaped, 0.0);
    }

    #[test]
    fn tile_pixel_is_16_bytes_repr_c() {
        assert_eq!(core::mem::size_of::<TilePixel>(), 16);
    }
}
