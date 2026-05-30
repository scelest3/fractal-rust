use arith::Complex;
use crate::{IterationMap, StepResult};

/// Multibrot set: z ← z^n + c via the principal branch exp(n · ln(z)) + c.
/// Supports non-integer exponents. Branch-cut seam on the negative real axis
/// is expected. Does not support deep zoom or distance-estimate colouring.
pub struct MultibroMap {
    pub exponent: f64,
    escape_radius_sq: f64,
}

impl MultibroMap {
    pub fn new(exponent: f64) -> Self {
        // r > 2^(1/(n-1)) guarantees orbit diverges once escaped.
        let r = 2.0_f64.max(2.0_f64.powf(1.0 / (exponent - 1.0).max(1e-6)));
        Self { exponent, escape_radius_sq: r * r }
    }
}

impl IterationMap for MultibroMap {
    #[inline(always)]
    fn step(&self, z: Complex<f64>, c: Complex<f64>) -> StepResult {
        // z^n via principal branch: exp(n · ln(z)); treat z = 0 as 0^n = 0.
        let z_next = if z.re == 0.0 && z.im == 0.0 {
            c
        } else {
            let r = z.norm_sqr().sqrt();
            let theta = z.im.atan2(z.re);
            let ln_r = r.ln();
            let re = (self.exponent * ln_r).exp() * (self.exponent * theta).cos();
            let im = (self.exponent * ln_r).exp() * (self.exponent * theta).sin();
            Complex::new(re + c.re, im + c.im)
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
    use crate::{escape_time, EscapeResult};

    fn map3() -> MultibroMap { MultibroMap::new(3.0) }

    #[test]
    fn origin_interior() {
        let r = escape_time(&map3(), Complex::new(0.0, 0.0), 500);
        assert!(matches!(r, EscapeResult::Interior { .. }), "c=0 should be interior");
    }

    #[test]
    fn far_exterior_escapes_immediately() {
        let r = escape_time(&map3(), Complex::new(5.0, 0.0), 500);
        match r {
            EscapeResult::Escaped { iter, .. } => assert_eq!(iter, 0),
            _ => panic!("c=5 should escape at iter=0"),
        }
    }

    #[test]
    fn smooth_t_uses_ln_exponent() {
        // c = 1+0i, exp=3: orbits escape quickly; verify smooth_t formula uses ln(3).
        let r = escape_time(&map3(), Complex::new(1.0, 0.0), 500);
        match r {
            EscapeResult::Escaped { iter, smooth_t, .. } => {
                // Recompute the escape z by stepping manually.
                let map = map3();
                let mut z = Complex::new(0.0_f64, 0.0);
                let c = Complex::new(1.0, 0.0);
                for _ in 0..=iter {
                    let sr = map.step(z, c);
                    z = sr.z;
                }
                let norm = z.norm_sqr().sqrt();
                let expected = (iter as f64 + 1.0) - norm.ln().ln() / 3.0_f64.ln();
                assert!(
                    (smooth_t - expected).abs() < 1e-10,
                    "smooth_t={smooth_t}, expected={expected}"
                );
            }
            _ => panic!("c=1 should escape"),
        }
    }

    #[test]
    fn log_exponent_returns_ln_exponent() {
        assert!((map3().log_exponent() - 3.0_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn no_distance_estimate() {
        assert!(!map3().supports_distance_estimate());
    }

    #[test]
    fn escape_radius_boundary_exponent_1_5() {
        let map = MultibroMap::new(1.5);
        // r = 2^(1/0.5) = 4.0, so escape_radius_sq = 16.0
        let expected = 2.0_f64.powf(1.0 / 0.5).powi(2);
        assert!((map.escape_radius_sq - expected).abs() < 1e-10,
            "escape_radius_sq={}, expected={expected}", map.escape_radius_sq);
    }

    #[test]
    fn golden_boundary_regression() {
        // c = -0.8+0.8i escapes at iter≈2 for exp=3; verifies the full path is wired correctly.
        // (The issue spec originally cited -0.5+0.5i, but that point is interior for z^3+c.)
        let r = escape_time(&map3(), Complex::new(-0.8, 0.8), 500);
        match r {
            EscapeResult::Escaped { smooth_t, .. } => {
                assert!(smooth_t.is_finite() && smooth_t > 0.0,
                    "smooth_t must be finite and positive, got {smooth_t}");
            }
            _ => panic!("c=-0.8+0.8i should escape for exp=3"),
        }
    }
}
