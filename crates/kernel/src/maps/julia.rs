use arith::Complex;
use crate::{IterationMap, StepResult};

/// Julia set: z ← z^n + c via the principal branch exp(n · ln(z)) + c.
/// The pixel coordinate is used as `z₀`; `c` is a fixed parameter on the struct.
/// `exponent = 2.0` gives the classic Julia set; other values give Multibrot-Julia.
/// Does not support deep zoom or distance-estimate colouring.
pub struct JuliaMap {
    pub exponent: f64,
    pub c: Complex<f64>,
    escape_radius_sq: f64,
}

impl JuliaMap {
    pub fn new(exponent: f64, c: Complex<f64>) -> Self {
        let r = 2.0_f64.max(2.0_f64.powf(1.0 / (exponent - 1.0).max(1e-6)));
        Self { exponent, c, escape_radius_sq: r * r }
    }
}

impl IterationMap for JuliaMap {
    #[inline(always)]
    fn step(&self, z: Complex<f64>, _c_ignored: Complex<f64>) -> StepResult {
        // z^n via principal branch: exp(n · ln(z)); treat z = 0 as 0^n = 0.
        let z_next = if z.re == 0.0 && z.im == 0.0 {
            self.c
        } else {
            let r = z.norm_sqr().sqrt();
            let theta = z.im.atan2(z.re);
            let ln_r = r.ln();
            let re = (self.exponent * ln_r).exp() * (self.exponent * theta).cos();
            let im = (self.exponent * ln_r).exp() * (self.exponent * theta).sin();
            Complex::new(re + self.c.re, im + self.c.im)
        };
        StepResult {
            escaped: z_next.norm_sqr() > self.escape_radius_sq,
            z: z_next,
        }
    }

    #[inline(always)]
    fn escape_radius_sq(&self) -> f64 {
        self.escape_radius_sq
    }

    #[inline(always)]
    fn log_exponent(&self) -> f64 {
        self.exponent.ln()
    }

    #[inline(always)]
    fn supports_distance_estimate(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{escape_time_julia, EscapeResult};

    // c = -0.7+0.27i is just outside the Mandelbrot set; the orbit from z₀=0
    // escapes within ~500 iterations. Use for exterior/boundary tests only.
    fn classic_map() -> JuliaMap {
        JuliaMap::new(2.0, Complex::new(-0.7, 0.27))
    }

    // c = -0.5+0.5i is inside the main Mandelbrot cardioid (|1-√(1-4c)| ≈ 0.985 < 1),
    // so z₀=0 is a bounded interior point for this Julia set.
    fn interior_map() -> JuliaMap {
        JuliaMap::new(2.0, Complex::new(-0.5, 0.5))
    }

    #[test]
    fn interior() {
        let r = escape_time_julia(&interior_map(), Complex::new(0.0, 0.0), 500);
        assert!(matches!(r, EscapeResult::Interior { .. }), "z₀=0 should be interior");
    }

    #[test]
    fn exterior_escapes() {
        let r = escape_time_julia(&classic_map(), Complex::new(1.0, 0.0), 500);
        assert!(matches!(r, EscapeResult::Escaped { .. }), "z₀=1 should escape");
    }

    #[test]
    fn golden_boundary() {
        let map = classic_map();
        let z0 = Complex::new(0.3, 0.5);
        let r = escape_time_julia(&map, z0, 500);
        match r {
            EscapeResult::Escaped { iter, smooth_t, .. } => {
                // Recompute escape z by stepping manually from z0.
                let mut z = z0;
                for _ in 0..=iter {
                    z = map.step(z, Complex::new(0.0, 0.0)).z;
                }
                let norm = z.norm_sqr().sqrt();
                let expected = (iter as f64 + 1.0) - norm.ln().ln() / core::f64::consts::LN_2;
                assert!(
                    smooth_t.is_finite(),
                    "smooth_t must be finite, got {smooth_t}"
                );
                assert!(
                    (smooth_t - expected).abs() < 1e-10,
                    "smooth_t={smooth_t}, expected={expected}"
                );
            }
            _ => panic!("z₀=0.3+0.5i should escape for c=-0.7+0.27i"),
        }
    }

    #[test]
    fn log_exponent_returns_ln_exponent() {
        assert!((classic_map().log_exponent() - core::f64::consts::LN_2).abs() < 1e-15);
    }

    #[test]
    fn no_distance_estimate() {
        assert!(!classic_map().supports_distance_estimate());
    }
}
