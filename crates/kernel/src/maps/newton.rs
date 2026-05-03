use arith::Complex;

use crate::TilePixel;

// ── NewtonMap ─────────────────────────────────────────────────────────────────

pub struct NewtonMap {
    /// Polynomial coefficients ascending by degree: `coeffs[k]` is the coefficient of `z^k`.
    pub coeffs: [f64; 11],
    /// Live degree of the polynomial (1–10).
    pub degree: usize,
    /// First derivative coefficients: `dcoeffs[k] = (k+1) * coeffs[k+1]`.
    dcoeffs: [f64; 10],
    /// Second derivative coefficients: `dcoeffs2[k] = (k+2)*(k+1) * coeffs[k+2]`.
    dcoeffs2: [f64; 9],
}

impl NewtonMap {
    pub fn new(coeffs: [f64; 11], degree: usize) -> Self {
        let mut dcoeffs = [0.0f64; 10];
        for k in 0..degree {
            dcoeffs[k] = (k + 1) as f64 * coeffs[k + 1];
        }
        let mut dcoeffs2 = [0.0f64; 9];
        if degree >= 2 {
            for k in 0..degree - 1 {
                dcoeffs2[k] = (k + 2) as f64 * (k + 1) as f64 * coeffs[k + 2];
            }
        }
        Self { coeffs, degree, dcoeffs, dcoeffs2 }
    }

    /// Evaluate `p(z)`, `p′(z)`, and `p″(z)` simultaneously via Horner's method.
    pub fn eval(&self, z: Complex<f64>) -> (Complex<f64>, Complex<f64>, Complex<f64>) {
        let mut p = Complex::new(self.coeffs[self.degree], 0.0);
        for k in (0..self.degree).rev() {
            p = p * z + Complex::new(self.coeffs[k], 0.0);
        }

        let dp_degree = self.degree - 1;
        let mut dp = Complex::new(self.dcoeffs[dp_degree], 0.0);
        for k in (0..dp_degree).rev() {
            dp = dp * z + Complex::new(self.dcoeffs[k], 0.0);
        }

        let ddp = if self.degree >= 2 {
            let ddp_degree = self.degree - 2;
            let mut ddp = Complex::new(self.dcoeffs2[ddp_degree], 0.0);
            for k in (0..ddp_degree).rev() {
                ddp = ddp * z + Complex::new(self.dcoeffs2[k], 0.0);
            }
            ddp
        } else {
            Complex::zero()
        };

        (p, dp, ddp)
    }
}

// ── NewtonResult ──────────────────────────────────────────────────────────────

pub enum NewtonResult {
    /// Orbit converged to a root.
    ///
    /// `log_deriv` is the accumulated log|N′(z)| over the orbit — a negative
    /// number whose magnitude encodes basin depth. Deep inside a basin the
    /// Newton derivative |N′| → 0, making log_deriv very negative. Near the
    /// basin boundary |N′| stays closer to 1, so log_deriv is near 0.
    Converged { root_index: u32, log_deriv: f32 },
    /// Diverged (`|z| > 1e8`) or reached `max_iter` without converging.
    Unresolved,
}

// ── iterate_newton ────────────────────────────────────────────────────────────

const CONVERGENCE_EPS: f64 = 1e-6;
const DIVERGENCE_RADIUS_SQ: f64 = 1e16; // |z| > 1e8

/// Iterate the Newton map from `z0`, tracking the derivative for distance estimation.
///
/// Accumulates `log|N′(z_k)|` at each step where `N′(z) = p(z)·p″(z) / p′(z)²`.
/// This quantity is negative near roots (|N′| → 0) and near 0 near basin boundaries
/// (|N′| ≈ 1). The result encodes geometric proximity to the basin boundary without
/// dependence on `max_iter` or global position in the plane.
pub fn iterate_newton(
    map: &NewtonMap,
    z0: Complex<f64>,
    roots: &[Complex<f64>; 10],
    max_iter: u32,
) -> NewtonResult {
    let mut z = z0;
    let mut log_w = 0.0f64;

    for _ in 0..max_iter {
        let (p, dp, ddp) = map.eval(z);

        let p_norm      = p.norm_sqr().sqrt();
        let dp_norm_sq  = dp.norm_sqr();
        let ddp_norm    = ddp.norm_sqr().sqrt();

        // Accumulate log|N′(z)| = log(|p|·|p″| / |p′|²).
        // Guard against zero denominator (critical points) and zero ddp (low degree).
        if dp_norm_sq > 1e-30 && ddp_norm > 0.0 {
            let n_prime = p_norm * ddp_norm / dp_norm_sq;
            if n_prime > 0.0 {
                log_w += n_prime.ln();
            }
        }

        if p_norm < CONVERGENCE_EPS {
            let root_index = nearest_root_index(z, roots, map.degree);
            return NewtonResult::Converged { root_index, log_deriv: log_w as f32 };
        }

        if z.norm_sqr() > DIVERGENCE_RADIUS_SQ {
            return NewtonResult::Unresolved;
        }

        if dp_norm_sq < 1e-30 {
            return NewtonResult::Unresolved;
        }

        z = z - p / dp;
    }
    NewtonResult::Unresolved
}

// ── find_roots ────────────────────────────────────────────────────────────────

const DK_MAX_ITER: usize = 200;
const DK_EPS: f64 = 1e-12;

/// Find all roots of the polynomial encoded in `map` using Durand-Kerner.
///
/// Returns up to `map.degree` roots sorted by `arg(root)` ascending.
/// Unused slots (indices `degree..10`) are zeroed.
pub fn find_roots(map: &NewtonMap) -> [Complex<f64>; 10] {
    let n = map.degree;
    let lead = map.coeffs[n];

    let radius = (map.coeffs[0] / lead).abs().powf(1.0 / n as f64).max(0.5);
    let seed = Complex::new(0.4, 0.9);
    let mut roots = [Complex::<f64>::zero(); 10];
    let mut seed_power = Complex::new(radius, 0.0);
    for k in 0..n {
        roots[k] = seed_power;
        seed_power = seed_power * seed;
    }

    for _ in 0..DK_MAX_ITER {
        let mut max_step = 0.0f64;
        for i in 0..n {
            let (p, _, _) = map.eval(roots[i]);
            let mut denom = Complex::new(1.0, 0.0);
            for j in 0..n {
                if j != i {
                    denom = denom * (roots[i] - roots[j]);
                }
            }
            let denom_sq = denom.norm_sqr();
            if denom_sq < 1e-60 { continue; }
            let step = (1.0 / denom_sq) * (p * denom.conj());
            max_step = max_step.max(step.norm_sqr().sqrt());
            roots[i] = roots[i] - step;
        }
        if max_step < DK_EPS { break; }
    }

    roots[..n].sort_by(|a, b| {
        let arg_a = a.im.atan2(a.re);
        let arg_b = b.im.atan2(b.re);
        arg_a.partial_cmp(&arg_b).unwrap_or(core::cmp::Ordering::Equal)
    });

    roots
}

// ── newton_tile_pixel ─────────────────────────────────────────────────────────

/// Pack a `NewtonResult` into the standard 4-channel `TilePixel`.
///
/// Channel layout (matches the fragment shader's Newton branch in §6.2):
///   r = log_deriv (accumulated log|N′|; negative = deep in basin)
///   g = 0.0
///   b = root_index as f32
///   a = 1.0 (Converged) | 0.0 (Unresolved)
pub fn newton_tile_pixel(result: NewtonResult) -> TilePixel {
    match result {
        NewtonResult::Converged { root_index, log_deriv } => TilePixel {
            smooth_t:    log_deriv,
            orbit_min_r: 0.0,
            angle:       root_index as f32,
            escaped:     1.0,
        },
        NewtonResult::Unresolved => TilePixel {
            smooth_t:    0.0,
            orbit_min_r: 0.0,
            angle:       0.0,
            escaped:     0.0,
        },
    }
}

fn nearest_root_index(z: Complex<f64>, roots: &[Complex<f64>; 10], degree: usize) -> u32 {
    let mut best = 0u32;
    let mut best_dist = f64::MAX;
    for (i, &r) in roots[..degree].iter().enumerate() {
        let d = (z - r).norm_sqr();
        if d < best_dist {
            best_dist = d;
            best = i as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use arith::Complex;

    fn z3_minus_1() -> NewtonMap {
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0;
        coeffs[3] = 1.0;
        NewtonMap::new(coeffs, 3)
    }

    // ── eval ──────────────────────────────────────────────────────────────────

    #[test]
    fn eval_degree1_z_minus_3_at_origin() {
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -3.0;
        coeffs[1] = 1.0;
        let map = NewtonMap::new(coeffs, 1);
        let (p, dp, ddp) = map.eval(Complex::new(0.0, 0.0));
        assert!((p.re - (-3.0)).abs() < 1e-15);
        assert!((dp.re - 1.0).abs() < 1e-15);
        assert_eq!(ddp.re, 0.0, "p''=0 for degree-1");
    }

    #[test]
    fn eval_z_cubed_minus_1_at_1() {
        // p(1)=0, p'(1)=3, p''(1)=6
        let map = z3_minus_1();
        let (p, dp, ddp) = map.eval(Complex::new(1.0, 0.0));
        assert!(p.re.abs() < 1e-14);
        assert!((dp.re - 3.0).abs() < 1e-14);
        assert!((ddp.re - 6.0).abs() < 1e-14, "p''(1) = {}", ddp.re);
    }

    #[test]
    fn eval_z_cubed_minus_1_at_origin() {
        // p''(z) = 6z → p''(0) = 0
        let map = z3_minus_1();
        let (_, _, ddp) = map.eval(Complex::new(0.0, 0.0));
        assert!(ddp.re.abs() < 1e-15, "p''(0) must be 0");
    }

    #[test]
    fn eval_sparse_z5_minus_1_at_1() {
        // p(1)=0, p'(1)=5, p''(1)=20
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0;
        coeffs[5] = 1.0;
        let map = NewtonMap::new(coeffs, 5);
        let (p, dp, ddp) = map.eval(Complex::new(1.0, 0.0));
        assert!(p.re.abs() < 1e-14);
        assert!((dp.re - 5.0).abs() < 1e-14);
        assert!((ddp.re - 20.0).abs() < 1e-13, "p''(1) for z⁵−1 = {}", ddp.re);
    }

    #[test]
    fn eval_degree10_at_z_equals_1() {
        // p(1)=11, p'(1)=55, p''(1)=330
        let coeffs = [1.0f64; 11];
        let map = NewtonMap::new(coeffs, 10);
        let (p, dp, ddp) = map.eval(Complex::new(1.0, 0.0));
        assert!((p.re - 11.0).abs() < 1e-13);
        assert!((dp.re - 55.0).abs() < 1e-13);
        assert!((ddp.re - 330.0).abs() < 1e-11, "p''(1) = {}", ddp.re);
    }

    // ── log_deriv distance estimate ───────────────────────────────────────────

    #[test]
    fn log_deriv_negative_for_deep_basin_point() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let result = iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100);
        match result {
            NewtonResult::Converged { log_deriv, .. } => {
                assert!(log_deriv < 0.0,
                    "log_deriv must be negative for a deep basin point, got {log_deriv}");
            }
            NewtonResult::Unresolved => panic!("z₀=2 must converge"),
        }
    }

    #[test]
    fn deeper_basin_has_more_negative_log_deriv() {
        // z=1.2 is physically closer to root=1 (deeper in the real basin) than z=2.
        // Both converge; z=1.2 should have a more negative log_deriv.
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let r_near = iterate_newton(&map, Complex::new(1.2, 0.0), &roots, 100);
        let r_far  = iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100);
        let (ld_near, ld_far) = match (r_near, r_far) {
            (NewtonResult::Converged { log_deriv: a, .. },
             NewtonResult::Converged { log_deriv: b, .. }) => (a, b),
            _ => panic!("both points must converge"),
        };
        assert!(ld_near < ld_far,
            "z=1.2 (closer to root) should have more negative log_deriv: {ld_near} vs {ld_far}");
    }

    // ── newton_tile_pixel ─────────────────────────────────────────────────────

    #[test]
    fn tile_pixel_converged_packs_log_deriv_into_r_channel() {
        let result = NewtonResult::Converged { root_index: 2, log_deriv: -7.5 };
        let px = newton_tile_pixel(result);
        assert!((px.smooth_t - (-7.5_f32)).abs() < 1e-6, "r = log_deriv, got {}", px.smooth_t);
        assert_eq!(px.orbit_min_r, 0.0);
        assert!((px.angle - 2.0_f32).abs() < 1e-6);
        assert_eq!(px.escaped, 1.0);
    }

    #[test]
    fn tile_pixel_unresolved_channels() {
        let px = newton_tile_pixel(NewtonResult::Unresolved);
        assert_eq!(px.smooth_t,    0.0);
        assert_eq!(px.orbit_min_r, 0.0);
        assert_eq!(px.angle,       0.0);
        assert_eq!(px.escaped,     0.0);
    }

    // ── iterate_newton ────────────────────────────────────────────────────────

    #[test]
    fn iterate_converges_from_deep_basin_point() {
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        let result = iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100);
        assert!(matches!(result, NewtonResult::Converged { .. }));
    }

    #[test]
    fn iterate_unresolved_for_divergent_start() {
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        let result = iterate_newton(&map, Complex::new(1e9, 0.0), &roots, 100);
        assert!(matches!(result, NewtonResult::Unresolved));
    }

    #[test]
    fn iterate_unresolved_when_max_iter_exceeded() {
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        let result = iterate_newton(&map, Complex::new(0.0, 0.0), &roots, 100);
        assert!(matches!(result, NewtonResult::Unresolved));
    }

    #[test]
    fn iterate_assigns_correct_root_index_with_real_roots() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let result = iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100);
        match result {
            NewtonResult::Converged { root_index, .. } => {
                assert_eq!(root_index, 1, "z₀=2 → root index 1 (root=1+0i, arg=0)");
            }
            NewtonResult::Unresolved => panic!("expected Converged"),
        }
    }

    // ── find_roots ────────────────────────────────────────────────────────────

    fn close_enough(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        (a - b).norm_sqr().sqrt() < tol
    }

    fn find_root_in_set(target: Complex<f64>, roots: &[Complex<f64>; 10], count: usize, tol: f64) -> bool {
        roots[..count].iter().any(|&r| close_enough(r, target, tol))
    }

    #[test]
    fn find_roots_z3_minus_1_finds_three_roots() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let sqrt3_2 = 3.0_f64.sqrt() / 2.0;
        assert!(find_root_in_set(Complex::new(1.0, 0.0), &roots, 3, 1e-10));
        assert!(find_root_in_set(Complex::new(-0.5, sqrt3_2), &roots, 3, 1e-10));
        assert!(find_root_in_set(Complex::new(-0.5, -sqrt3_2), &roots, 3, 1e-10));
    }

    #[test]
    fn find_roots_z4_minus_1_finds_four_roots() {
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0;
        coeffs[4] = 1.0;
        let map = NewtonMap::new(coeffs, 4);
        let roots = find_roots(&map);
        assert!(find_root_in_set(Complex::new( 1.0,  0.0), &roots, 4, 1e-10));
        assert!(find_root_in_set(Complex::new( 0.0,  1.0), &roots, 4, 1e-10));
        assert!(find_root_in_set(Complex::new(-1.0,  0.0), &roots, 4, 1e-10));
        assert!(find_root_in_set(Complex::new( 0.0, -1.0), &roots, 4, 1e-10));
    }

    #[test]
    fn find_roots_degree1_z_minus_3() {
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -3.0;
        coeffs[1] = 1.0;
        let map = NewtonMap::new(coeffs, 1);
        let roots = find_roots(&map);
        assert!(close_enough(roots[0], Complex::new(3.0, 0.0), 1e-10));
    }

    #[test]
    fn find_roots_sorted_by_arg_ascending() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let args: Vec<f64> = roots[..3].iter().map(|r| r.im.atan2(r.re)).collect();
        assert!(args[0] <= args[1] && args[1] <= args[2], "args: {:?}", args);
    }
}
