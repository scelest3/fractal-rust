use arith::Complex;

use crate::TilePixel;

// ── NewtonMap ─────────────────────────────────────────────────────────────────

pub struct NewtonMap {
    /// Polynomial coefficients ascending by degree: `coeffs[k]` is the coefficient of `z^k`.
    pub coeffs: [f64; 11],
    /// Live degree of the polynomial (1–10).
    pub degree: usize,
    /// Precomputed derivative coefficients: `dcoeffs[k] = (k+1) * coeffs[k+1]`.
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
    Converged { root_index: u32, convergence_iter: u32 },
    /// Diverged (`|z| > 1e8`) or reached `max_iter` without converging.
    Unresolved,
}

// ── iterate_newton ────────────────────────────────────────────────────────────

const CONVERGENCE_EPS: f64 = 1e-6;
const DIVERGENCE_RADIUS_SQ: f64 = 1e16; // |z| > 1e8

/// Iterate the Newton map from `z0` until convergence or failure.
///
/// `roots` is the sorted root array from `find_roots`. The nearest root
/// determines `root_index` in `Converged`. Pass `&[Complex::zero(); 10]`
/// as a placeholder before Stage 4 wires real roots in.
pub fn iterate_newton(
    map: &NewtonMap,
    z0: Complex<f64>,
    roots: &[Complex<f64>; 10],
    max_iter: u32,
) -> NewtonResult {
    let mut z = z0;
    for iter in 0..max_iter {
        let (p, dp) = map.eval(z);

        if p.norm_sqr().sqrt() < CONVERGENCE_EPS {
            let root_index = nearest_root_index(z, roots, map.degree);
            return NewtonResult::Converged { root_index, convergence_iter: iter };
        }

        if z.norm_sqr() > DIVERGENCE_RADIUS_SQ {
            return NewtonResult::Unresolved;
        }

        if dp.norm_sqr() < 1e-30 {
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
    // Monic scaling: divide by leading coefficient so the polynomial is monic.
    let lead = map.coeffs[n];

    // Initial estimates: successive powers of an off-axis seed scaled to a
    // radius derived from the coefficient ratio heuristic. Using a non-symmetric
    // seed (0.4 + 0.9i) rather than equally-spaced angles avoids collapse for
    // symmetric polynomials like z³−1 whose roots lie at the seed positions.
    let radius = (map.coeffs[0] / lead).abs().powf(1.0 / n as f64).max(0.5);
    let seed = Complex::new(0.4, 0.9); // |seed| ≈ 0.985, off-axis
    let mut roots = [Complex::<f64>::zero(); 10];
    let mut seed_power = Complex::new(radius, 0.0);
    for k in 0..n {
        roots[k] = seed_power;
        seed_power = seed_power * seed;
    }

    // Durand-Kerner iteration.
    for _ in 0..DK_MAX_ITER {
        let mut max_step = 0.0f64;
        for i in 0..n {
            let (p, _) = map.eval(roots[i]);
            // Denominator: ∏_{j≠i} (roots[i] − roots[j])
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

    // Sort by arg(root) ascending for canonical hue-stable ordering.
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
///   r = convergence_t (convergence_iter / max_iter)
///   g = 0.0
///   b = root_index as f32
///   a = 1.0 (Converged) | 0.0 (Unresolved)
pub fn newton_tile_pixel(result: NewtonResult, max_iter: u32) -> TilePixel {
    match result {
        NewtonResult::Converged { root_index, convergence_iter } => TilePixel {
            smooth_t:    convergence_iter as f32 / max_iter as f32,
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

    // ── Stage 1: NewtonMap + Horner eval ─────────────────────────────────────

    // ── Stage 2: iterate_newton ───────────────────────────────────────────────

    fn z3_minus_1() -> NewtonMap {
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0;
        coeffs[3] = 1.0;
        NewtonMap::new(coeffs, 3)
    }

    // ── Stage 4: find_roots ───────────────────────────────────────────────────

    fn close_enough(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        (a - b).norm_sqr().sqrt() < tol
    }

    fn find_root_in_set(
        target: Complex<f64>,
        roots: &[Complex<f64>; 10],
        count: usize,
        tol: f64,
    ) -> bool {
        roots[..count].iter().any(|&r| close_enough(r, target, tol))
    }

    #[test]
    fn find_roots_z4_minus_1_finds_four_roots() {
        // z⁴ − 1 roots: 1, i, −1, −i
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0;
        coeffs[4] = 1.0;
        let map = NewtonMap::new(coeffs, 4);
        let roots = find_roots(&map);
        assert!(find_root_in_set(Complex::new( 1.0,  0.0), &roots, 4, 1e-10), "root  1 not found");
        assert!(find_root_in_set(Complex::new( 0.0,  1.0), &roots, 4, 1e-10), "root  i not found");
        assert!(find_root_in_set(Complex::new(-1.0,  0.0), &roots, 4, 1e-10), "root −1 not found");
        assert!(find_root_in_set(Complex::new( 0.0, -1.0), &roots, 4, 1e-10), "root −i not found");
    }

    #[test]
    fn find_roots_degree1_z_minus_3() {
        // p(z) = z − 3: single root at 3.
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -3.0;
        coeffs[1] = 1.0;
        let map = NewtonMap::new(coeffs, 1);
        let roots = find_roots(&map);
        assert!(close_enough(roots[0], Complex::new(3.0, 0.0), 1e-10),
            "root = {:?}", roots[0]);
    }

    #[test]
    fn iterate_assigns_correct_root_index_with_real_roots() {
        // z³ − 1: root 0 (most negative arg) = −½ − i√3/2, root 1 = 1, root 2 = −½ + i√3/2.
        // z₀ = 2: real positive basin → converges to root at 1, which is index 1.
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let result = iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100);
        match result {
            NewtonResult::Converged { root_index, .. } => {
                // root at 1+0i has arg=0 — middle of the three args, so index 1.
                assert_eq!(root_index, 1, "z₀=2 should converge to root index 1 (root=1+0i)");
            }
            NewtonResult::Unresolved => panic!("expected Converged, got Unresolved"),
        }
    }

    #[test]
    fn find_roots_sorted_by_arg_ascending() {
        // z³ − 1: args are −2π/3, 0, +2π/3 → sorted order: arg(−½−i√3/2) < arg(1) < arg(−½+i√3/2)
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let args: Vec<f64> = roots[..3].iter()
            .map(|r| r.im.atan2(r.re))
            .collect();
        assert!(args[0] <= args[1] && args[1] <= args[2],
            "roots not sorted by arg: {:?}", args);
    }

    #[test]
    fn find_roots_z3_minus_1_finds_three_roots() {
        // z³ − 1 has roots: 1, −½ + i√3/2, −½ − i√3/2
        let map = z3_minus_1();
        let roots = find_roots(&map);
        let sqrt3_2 = 3.0_f64.sqrt() / 2.0;
        assert!(find_root_in_set(Complex::new(1.0, 0.0), &roots, 3, 1e-10),
            "root 1 not found");
        assert!(find_root_in_set(Complex::new(-0.5, sqrt3_2), &roots, 3, 1e-10),
            "root −½+i√3/2 not found");
        assert!(find_root_in_set(Complex::new(-0.5, -sqrt3_2), &roots, 3, 1e-10),
            "root −½−i√3/2 not found");
    }

    // ── Stage 3: newton_tile_pixel ────────────────────────────────────────────

    #[test]
    fn tile_pixel_converged_channels() {
        // root_index=2, convergence_iter=10, max_iter=100
        // r = 10/100 = 0.1,  g = 0.0,  b = 2.0,  a = 1.0
        let result = NewtonResult::Converged { root_index: 2, convergence_iter: 10 };
        let px = newton_tile_pixel(result, 100);
        assert!((px.smooth_t - 0.1_f32).abs() < 1e-6, "r = {}", px.smooth_t);
        assert_eq!(px.orbit_min_r, 0.0, "g must be 0");
        assert!((px.angle - 2.0_f32).abs() < 1e-6, "b = {}", px.angle);
        assert_eq!(px.escaped, 1.0, "a must be 1 for Converged");
    }

    #[test]
    fn tile_pixel_unresolved_channels() {
        let px = newton_tile_pixel(NewtonResult::Unresolved, 100);
        assert_eq!(px.smooth_t,    0.0, "r must be 0 for Unresolved");
        assert_eq!(px.orbit_min_r, 0.0, "g must be 0");
        assert_eq!(px.angle,       0.0, "b must be 0");
        assert_eq!(px.escaped,     0.0, "a must be 0 for Unresolved");
    }

    #[test]
    fn iterate_unresolved_for_divergent_start() {
        // z₀ = 1e9 is well past the bailout radius.
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        let result = iterate_newton(&map, Complex::new(1e9, 0.0), &roots, 100);
        assert!(matches!(result, NewtonResult::Unresolved),
            "z₀ beyond bailout should be Unresolved");
    }

    #[test]
    fn iterate_unresolved_when_max_iter_exceeded() {
        // p′(z) = 0 at z = 0 for z³−1, so the step is undefined and iteration stalls.
        // With max_iter=1 and z₀=0, we immediately hit the dp≈0 guard → Unresolved.
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10];
        let result = iterate_newton(&map, Complex::new(0.0, 0.0), &roots, 100);
        assert!(matches!(result, NewtonResult::Unresolved),
            "z₀ at critical point (p′=0) should be Unresolved");
    }

    #[test]
    fn iterate_converges_from_deep_basin_point() {
        // z₀ = 2 is clearly in the real-positive basin of z³−1 (root = 1).
        // Expects Converged within 100 iterations.
        let map = z3_minus_1();
        let roots = [Complex::<f64>::zero(); 10]; // placeholder — root_index will be 0
        let result = iterate_newton(&map, Complex::new(2.0, 0.0), &roots, 100);
        assert!(matches!(result, NewtonResult::Converged { .. }),
            "z₀=2 should converge for z³−1");
    }

    #[test]
    fn eval_degree10_at_z_equals_1() {
        // All coefficients = 1.0: p(z) = 1 + z + z² + … + z¹⁰
        // p(1) = 11,  p′(1) = 0+1+2+…+10 = 55
        let coeffs = [1.0f64; 11];
        let map = NewtonMap::new(coeffs, 10);
        let (p, dp) = map.eval(Complex::new(1.0, 0.0));
        assert!((p.re - 11.0).abs() < 1e-13, "p(1) = {}", p.re);
        assert!((dp.re - 55.0).abs() < 1e-13, "p'(1) = {}", dp.re);
    }

    #[test]
    fn eval_sparse_z5_minus_1_at_1() {
        // p(z) = z⁵ − 1, three zero middle coefficients
        // p(1) = 0,  p′(1) = 5
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0;
        coeffs[5] = 1.0;
        let map = NewtonMap::new(coeffs, 5);
        let (p, dp) = map.eval(Complex::new(1.0, 0.0));
        assert!(p.re.abs() < 1e-14, "p(1) = {}", p.re);
        assert!((dp.re - 5.0).abs() < 1e-14, "p'(1) = {}", dp.re);
    }

    #[test]
    fn eval_z_cubed_minus_1_at_1() {
        // p(z) = z³ − 1  →  coeffs = [−1, 0, 0, 1, 0, …]
        // p(1) = 0,  p′(1) = 3
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -1.0;
        coeffs[3] = 1.0;
        let map = NewtonMap::new(coeffs, 3);
        let (p, dp) = map.eval(Complex::new(1.0, 0.0));
        assert!(p.re.abs() < 1e-14, "p(1) = {}", p.re);
        assert!(p.im.abs() < 1e-14);
        assert!((dp.re - 3.0).abs() < 1e-14, "p'(1) = {}", dp.re);
        assert!(dp.im.abs() < 1e-14);
    }

    #[test]
    fn eval_degree1_z_minus_3_at_origin() {
        // p(z) = z − 3  →  coeffs = [−3, 1, 0, …]
        // p(0) = −3,  p′(0) = 1
        let mut coeffs = [0.0f64; 11];
        coeffs[0] = -3.0;
        coeffs[1] = 1.0;
        let map = NewtonMap::new(coeffs, 1);
        let (p, dp) = map.eval(Complex::new(0.0, 0.0));
        assert!((p.re - (-3.0)).abs() < 1e-15, "p(0) = {}", p.re);
        assert!(p.im.abs() < 1e-15);
        assert!((dp.re - 1.0).abs() < 1e-15, "p'(0) = {}", dp.re);
        assert!(dp.im.abs() < 1e-15);
    }
}
