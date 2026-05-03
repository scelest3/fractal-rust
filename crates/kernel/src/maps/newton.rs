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
}

impl NewtonMap {
    pub fn new(coeffs: [f64; 11], degree: usize) -> Self {
        let mut dcoeffs = [0.0f64; 10];
        for k in 0..degree {
            dcoeffs[k] = (k + 1) as f64 * coeffs[k + 1];
        }
        Self { coeffs, degree, dcoeffs }
    }

    /// Evaluate `p(z)` and `p′(z)` simultaneously via Horner's method.
    pub fn eval(&self, z: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
        let mut p = Complex::new(self.coeffs[self.degree], 0.0);
        for k in (0..self.degree).rev() {
            p = p * z + Complex::new(self.coeffs[k], 0.0);
        }
        let dp_degree = self.degree - 1;
        let mut dp = Complex::new(self.dcoeffs[dp_degree], 0.0);
        for k in (0..dp_degree).rev() {
            dp = dp * z + Complex::new(self.dcoeffs[k], 0.0);
        }
        (p, dp)
    }
}

// ── NewtonResult ──────────────────────────────────────────────────────────────

pub enum NewtonResult {
    /// Orbit converged to a root.
    ///
    /// `convergence_iter` is the integer step count N. `log_p_norm` is
    /// ln|p(z_N)| at the convergence step. Together they drive the smooth
    /// iteration count in the shader:
    ///   smooth_f = N − log₂(−log|p| / −log ε)
    /// which is a continuous function of z₀, eliminating discrete rings.
    Converged { root_index: u32, convergence_iter: u32, log_p_norm: f32 },
    /// Diverged (`|z| > 1e8`) or reached `max_iter` without converging.
    Unresolved,
}

// ── iterate_newton ────────────────────────────────────────────────────────────

const CONVERGENCE_EPS: f64 = 1e-6;
const DIVERGENCE_RADIUS_SQ: f64 = 1e16; // |z| > 1e8

/// Iterate the Newton map from `z0` until convergence or failure.
pub fn iterate_newton(
    map: &NewtonMap,
    z0: Complex<f64>,
    roots: &[Complex<f64>; 10],
    max_iter: u32,
) -> NewtonResult {
    let mut z = z0;
    for iter in 0..max_iter {
        let (p, dp) = map.eval(z);
        let p_norm = p.norm_sqr().sqrt();

        if p_norm < CONVERGENCE_EPS {
            let root_index = nearest_root_index(z, roots, map.degree);
            let log_p_norm = p_norm.max(f64::MIN_POSITIVE).ln() as f32;
            return NewtonResult::Converged { root_index, convergence_iter: iter, log_p_norm };
        }

        if z.norm_sqr() > DIVERGENCE_RADIUS_SQ {
            return NewtonResult::Unresolved;
        }

        let dp_norm_sq = dp.norm_sqr();
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
            let (p, _) = map.eval(roots[i]);
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
/// Channel layout:
///   r = convergence_iter as f32 (integer step count N)
///   g = log_p_norm (ln|p(z_N)|; used with r to compute smooth_f in shader)
///   b = root_index as f32
///   a = 1.0 (Converged) | 0.0 (Unresolved)
pub fn newton_tile_pixel(result: NewtonResult) -> TilePixel {
    match result {
        NewtonResult::Converged { root_index, convergence_iter, log_p_norm } => TilePixel {
            smooth_t:    convergence_iter as f32,
            orbit_min_r: log_p_norm,
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
        coeffs[0] = -3.0; coeffs[1] = 1.0;
        let map = NewtonMap::new(coeffs, 1);
        let (p, dp) = map.eval(Complex::new(0.0, 0.0));
        assert!((p.re - (-3.0)).abs() < 1e-15);
        assert!((dp.re - 1.0).abs() < 1e-15);
    }

    #[test]
    fn eval_z_cubed_minus_1_at_1() {
        // p(1)=0, p'(1)=3
        let map = z3_minus_1();
        let (p, dp) = map.eval(Complex::new(1.0, 0.0));
        assert!(p.re.abs() < 1e-14);
        assert!((dp.re - 3.0).abs() < 1e-14);
    }

    #[test]
    fn eval_sparse_z5_minus_1_at_1() {
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0; coeffs[5] = 1.0;
        let map = NewtonMap::new(coeffs, 5);
        let (p, dp) = map.eval(Complex::new(1.0, 0.0));
        assert!(p.re.abs() < 1e-14);
        assert!((dp.re - 5.0).abs() < 1e-14);
    }

    #[test]
    fn eval_degree10_at_z_equals_1() {
        // p(1)=11, p'(1)=55
        let coeffs = [1.0f64; 11];
        let map = NewtonMap::new(coeffs, 10);
        let (p, dp) = map.eval(Complex::new(1.0, 0.0));
        assert!((p.re - 11.0).abs() < 1e-13);
        assert!((dp.re - 55.0).abs() < 1e-13);
    }

    // ── iterate_newton ────────────────────────────────────────────────────────

    #[test]
    fn iterate_converges_from_deep_basin_point() {
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        assert!(matches!(
            iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100),
            NewtonResult::Converged { .. }
        ));
    }

    #[test]
    fn iterate_unresolved_for_divergent_start() {
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        assert!(matches!(
            iterate_newton(&map, Complex::new(1e9, 0.0), &roots, 100),
            NewtonResult::Unresolved
        ));
    }

    #[test]
    fn iterate_unresolved_at_critical_point() {
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        assert!(matches!(
            iterate_newton(&map, Complex::new(0.0, 0.0), &roots, 100),
            NewtonResult::Unresolved
        ));
    }

    #[test]
    fn iterate_assigns_correct_root_index() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        match iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100) {
            NewtonResult::Converged { root_index, .. } =>
                assert_eq!(root_index, 1, "z₀=2 → root index 1 (root=1+0i, arg=0)"),
            NewtonResult::Unresolved => panic!("expected Converged"),
        }
    }

    #[test]
    fn log_p_norm_at_convergence_is_at_most_ln_eps() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        match iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100) {
            NewtonResult::Converged { log_p_norm, .. } => {
                let ln_eps = (CONVERGENCE_EPS.ln()) as f32;
                assert!(log_p_norm <= ln_eps + 0.01,
                    "log_p_norm {log_p_norm} should be ≤ ln(ε) ≈ {ln_eps}");
            }
            NewtonResult::Unresolved => panic!("expected Converged"),
        }
    }

    #[test]
    fn deeper_basin_converges_in_fewer_iterations() {
        // z=1.2 is closer to root=1 and should take fewer steps than z=5.
        // (z=5 starts further away so takes more Newton steps.)
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let n_near = match iterate_newton(&map, Complex::new(1.2, 0.0), &roots, 100) {
            NewtonResult::Converged { convergence_iter, .. } => convergence_iter,
            _ => panic!("must converge"),
        };
        let n_far = match iterate_newton(&map, Complex::new(5.0, 0.0), &roots, 100) {
            NewtonResult::Converged { convergence_iter, .. } => convergence_iter,
            _ => panic!("must converge"),
        };
        assert!(n_near <= n_far,
            "z=1.2 ({n_near} iters) should converge no slower than z=5 ({n_far} iters)");
    }

    // ── newton_tile_pixel ─────────────────────────────────────────────────────

    #[test]
    fn tile_pixel_converged_channels() {
        let result = NewtonResult::Converged {
            root_index: 2, convergence_iter: 7, log_p_norm: -18.4,
        };
        let px = newton_tile_pixel(result);
        assert!((px.smooth_t    - 7.0_f32).abs()   < 1e-6, "r = convergence_iter");
        assert!((px.orbit_min_r - (-18.4_f32)).abs() < 1e-4, "g = log_p_norm");
        assert!((px.angle - 2.0_f32).abs()          < 1e-6, "b = root_index");
        assert_eq!(px.escaped, 1.0, "a = 1 for Converged");
    }

    #[test]
    fn tile_pixel_unresolved_channels() {
        let px = newton_tile_pixel(NewtonResult::Unresolved);
        assert_eq!(px.smooth_t,    0.0);
        assert_eq!(px.orbit_min_r, 0.0);
        assert_eq!(px.angle,       0.0);
        assert_eq!(px.escaped,     0.0);
    }

    // ── find_roots ────────────────────────────────────────────────────────────

    fn close_enough(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        (a - b).norm_sqr().sqrt() < tol
    }

    fn find_root_in_set(target: Complex<f64>, roots: &[Complex<f64>; 10], n: usize, tol: f64) -> bool {
        roots[..n].iter().any(|&r| close_enough(r, target, tol))
    }

    #[test]
    fn find_roots_z3_minus_1_finds_three_roots() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let s = 3.0_f64.sqrt() / 2.0;
        assert!(find_root_in_set(Complex::new(1.0, 0.0),  &roots, 3, 1e-10));
        assert!(find_root_in_set(Complex::new(-0.5,  s),  &roots, 3, 1e-10));
        assert!(find_root_in_set(Complex::new(-0.5, -s),  &roots, 3, 1e-10));
    }

    #[test]
    fn find_roots_z4_minus_1_finds_four_roots() {
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0; coeffs[4] = 1.0;
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
        coeffs[0] = -3.0; coeffs[1] = 1.0;
        let map = NewtonMap::new(coeffs, 1);
        assert!(close_enough(find_roots(&map)[0], Complex::new(3.0, 0.0), 1e-10));
    }

    #[test]
    fn find_roots_sorted_by_arg_ascending() {
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let args: Vec<f64> = roots[..3].iter().map(|r| r.im.atan2(r.re)).collect();
        assert!(args[0] <= args[1] && args[1] <= args[2], "args: {args:?}");
    }
}
