use arith::{Complex, Precision};

use crate::{IterationMap, PerturbationSupport, StepResult};

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

    #[inline(always)]
    fn converged(&self, z: Complex<f64>, z_prev: Complex<f64>) -> Option<u32> {
        if (z - z_prev).norm_sqr() < 1e-20 { Some(1) } else { None }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use arith::Complex;

    const MAP: MandelbrotMap = MandelbrotMap;

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

    #[test]
    fn glitch_threshold_is_1e_3() {
        assert_eq!(MAP.glitch_threshold(), 1e-3);
    }

    #[test]
    fn perturb_step_at_zero_ref() {
        // ref_z = 0, dz = 0, dc = (0.1, 0): dz_next = 0 + 0 + dc = (0.1, 0)
        let dz = Complex::new(0.0_f64, 0.0);
        let dc = Complex::new(0.1_f64, 0.0);
        let ref_z = Complex::new(0.0_f64, 0.0);
        let result = MAP.perturb_step(dz, dc, ref_z);
        assert!((result.re - 0.1).abs() < 1e-15);
        assert!(result.im.abs() < 1e-15);
    }

    #[test]
    fn perturb_step_matches_formula() {
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

    #[test]
    fn log_exponent_returns_ln_2() {
        assert_eq!(MAP.log_exponent(), core::f64::consts::LN_2);
    }

    #[test]
    fn supports_distance_estimate_returns_true() {
        assert!(MAP.supports_distance_estimate());
    }
}
