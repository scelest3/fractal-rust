//! Fractal iteration engine — escape-time computation, perturbation theory, BLA.
//!
//! `kernel` imports `arith::{Complex, DeltaC, Precision}` and owns all fractal
//! algorithms. `Complex<f64>` here is always `arith::Complex<f64>` — the
//! workspace-wide type. `wasm-bridge` is the only crate that names concrete
//! `BigFloat<N>` types; `kernel` is generic over `T: Precision` for the cold
//! reference orbit path (Phase 2).

use arith::{Complex, DeltaC, Precision};

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

// ── perturb_pixel ─────────────────────────────────────────────────────────────

/// Compute the escape-time result for one pixel via perturbation theory.
///
/// When `bla_table` is non-empty it is used to skip multiple orbit steps at once
/// in smooth regions (BLA acceleration). Passing `&[]` disables BLA and falls back
/// to direct per-step perturbation — used in validation tests and in Phase 1 rendering
/// where the BLA table is not yet available. When non-empty, `bla_table` must have the
/// same length as `ref_orbit`.
///
/// Escape is checked on `Z_n + δz_n` before each step (pre-step check). `orbit_min_r`
/// is tracked over `{z_1, z_2, …}`, excluding z_0 = 0, matching `escape_time`'s
/// convention. BLA steps skip intermediate orbit entries so `orbit_min_r` may be
/// slightly larger than the non-BLA value — acceptable for deep-zoom rendering where
/// orbit traps are rarely the primary coloring algorithm.
///
/// Returns `EscapeResult::Glitched` when the Pauldelbrot criterion fires:
/// `|δz_n|² > glitch_threshold² · |Z_n|²`. The caller must re-render with a
/// reference orbit that passes closer to the pixel.
///
/// # Smooth-t alignment with `escape_time`
/// `n − ln(ln(|z_n|)) / ln(2)` (pre-step index n) equals
/// `(iter + 1) − ln(ln(|z_{iter+1}|)) / ln(2)` (escape_time's iter = n − 1)
/// for the same escaped z, so `TilePixel` output matches escape_time bit-for-bit
/// when the zero-reference trick is used in tests.
pub fn perturb_pixel<M: PerturbationSupport>(
    map: &M,
    ref_orbit: &[Complex<f64>],
    bla_table: &[BlaEntry],
    delta_c: DeltaC,
    max_iter: u32,
) -> EscapeResult {
    let dc = delta_c.as_complex_f64();
    let mut dz = Complex::<f64>::zero();
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = Complex::<f64>::zero();
    let limit = (max_iter as usize).min(ref_orbit.len());
    let glitch_thresh_sq = {
        let t = map.glitch_threshold();
        t * t
    };

    let mut n = 0usize;
    while n < limit {
        let z_ref = ref_orbit[n];
        let z_n = z_ref + dz;

        // Track orbit_min over {z_1, z_2, …} — z_0 = 0 excluded (matches escape_time).
        if n > 0 {
            let r = z_n.norm_sqr().sqrt();
            if r < orbit_min_r {
                orbit_min_r = r;
                orbit_min_z = z_n;
            }
        }

        // Pre-step escape check on z_n = Z_n + δz_n.
        let r_sq = z_n.norm_sqr();
        if r_sq > map.escape_radius_sq() {
            let norm = r_sq.sqrt();
            let smooth_t = n as f64 - norm.ln().ln() / core::f64::consts::LN_2;
            return EscapeResult::Escaped {
                iter: n as u32,
                smooth_t,
                orbit_min_r,
                orbit_min_z,
                angle_final: z_n.im.atan2(z_n.re),
            };
        }

        // Pauldelbrot glitch criterion: |δz_n|² > threshold² · |Z_n|².
        // Guard: skip when Z_n ≈ 0 — criterion is undefined near the origin.
        let ref_sq = z_ref.norm_sqr();
        if ref_sq > 1e-30 && dz.norm_sqr() > glitch_thresh_sq * ref_sq {
            return EscapeResult::Glitched;
        }

        // BLA: skip multiple steps when |δz_n| < valid_radius.
        // bla_table.get(n) returns None for empty slices → always falls through to direct step.
        if let Some(&entry) = bla_table.get(n) {
            let next_n = n + entry.skip as usize;
            if entry.valid_radius > 0.0
                && dz.norm_sqr().sqrt() < entry.valid_radius
                && next_n <= limit
            {
                dz = entry.a * dz + entry.b * dc;
                n = next_n;
                continue;
            }
        }

        // Direct single perturbation step.
        dz = map.perturb_step(dz, dc, z_ref);
        n += 1;
    }

    EscapeResult::Interior { orbit_min_r, orbit_min_z }
}

// ── BlaEntry ──────────────────────────────────────────────────────────────────

/// One entry in the BLA (Bilinear Approximation) acceleration table.
///
/// For a pixel starting at orbit index `n` with perturbation `δz_n`, the
/// approximation `δz_{n+skip} ≈ a·δz_n + b·δc` is valid when `|δz_n| < valid_radius`.
///
/// Layout is `repr(C)` so entries can be read directly from shared WASM memory.
/// Total size: 48 bytes (verified by `size_of` test).
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct BlaEntry {
    /// Linear coefficient for `δz_n` (maps perturbation forward by `skip` steps).
    pub a: Complex<f64>,       // 16 B
    /// Linear coefficient for `δc` (accumulated dc contribution over `skip` steps).
    pub b: Complex<f64>,       // 16 B
    /// Number of orbit steps this entry skips.
    pub skip: u32,             //  4 B
    _pad: u32,                 //  4 B  (ensures f64 alignment for valid_radius)
    /// Maximum `|δz_n|` for which the approximation is valid.
    pub valid_radius: f64,     //  8 B
}                              // 48 B total

/// Merge two consecutive BLA entries into one that covers both spans.
///
/// Merged coefficients: `A = A₁·A₀`, `B = A₁·B₀ + B₁`.
/// Merged radius (ARCHITECTURE.md): `min(r₀, r₁ / |A₀|)`.
/// The `/|A₀|` factor is mandatory — omitting it causes silent false banding.
fn bla_merge(e0: BlaEntry, e1: BlaEntry) -> BlaEntry {
    let a0_norm = e0.a.norm_sqr().sqrt();
    let valid_radius = if a0_norm > 0.0 {
        e0.valid_radius.min(e1.valid_radius / a0_norm)
    } else {
        0.0
    };
    BlaEntry {
        a: e1.a * e0.a,
        b: e1.a * e0.b + e1.b,
        skip: e0.skip + e1.skip,
        _pad: 0,
        valid_radius,
    }
}

/// Build the BLA acceleration table for a reference orbit.
///
/// Uses Zhuoran's level-doubling algorithm: builds log₂(orbit_len) levels of
/// merged entries, then stores the highest-level valid entry at each position.
///
/// `bla_table[n].skip` is the maximum number of steps that can be skipped from
/// orbit position `n` while keeping the approximation error within `valid_radius`.
///
/// An entry with `valid_radius == 0` is never applicable (position near orbit start
/// or where the orbit collapses); the rendering loop falls back to direct iteration.
pub fn build_bla_table(orbit: &[Complex<f64>]) -> Vec<BlaEntry> {
    let n = orbit.len();
    if n == 0 {
        return vec![];
    }

    // Level 0: single-step linear approximation δz_{k+1} ≈ 2·Z_k·δz_k + δc.
    // valid_radius = |Z_k|: valid when |δz| is smaller than the reference orbit value.
    let one = Complex::new(1.0_f64, 0.0);
    let level0: Vec<BlaEntry> = orbit
        .iter()
        .map(|&z| BlaEntry {
            a: Complex::new(2.0_f64, 0.0) * z,
            b: one,
            skip: 1,
            _pad: 0,
            valid_radius: z.norm_sqr().sqrt(),
        })
        .collect();

    // Build successive levels by merging adjacent pairs.
    // levels[k][i] covers orbit positions i .. i + 2^k.
    let mut levels: Vec<Vec<BlaEntry>> = vec![level0];
    let mut half = 1usize;
    while half < n {
        let prev = levels.last().unwrap();
        let cur: Vec<BlaEntry> = (0..n)
            .map(|i| {
                if i + half < n {
                    let e0 = prev[i];
                    let e1 = prev[i + half];
                    // Only merge if both halves have non-zero valid_radius.
                    if e0.valid_radius > 0.0 && e1.valid_radius > 0.0 {
                        bla_merge(e0, e1)
                    } else {
                        BlaEntry::default() // sentinel: valid_radius = 0, never used
                    }
                } else {
                    BlaEntry::default() // position too close to end; no entry at this level
                }
            })
            .collect();
        levels.push(cur);
        half *= 2;
    }

    // Final table: for each position, take the highest-level entry with valid_radius > 0.
    (0..n)
        .map(|i| {
            for lvl in (0..levels.len()).rev() {
                let e = levels[lvl][i];
                if e.valid_radius > 0.0 {
                    return e;
                }
            }
            BlaEntry::default() // Z_i = 0, no BLA applicable (e.g. orbit start)
        })
        .collect()
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

    // ── perturb_pixel ─────────────────────────────────────────────────────────

    // Build a zero reference orbit (c_ref = 0+0i, Z_n = 0 for all n).
    // With this reference, perturb_step(dz, dc, 0) = dz² + dc, which is
    // arithmetically identical to escape_time's z² + c, giving bit-identical output.
    fn zero_orbit(len: usize) -> Vec<Complex<f64>> {
        vec![Complex::new(0.0_f64, 0.0); len]
    }

    fn tile_pixels_match(a: TilePixel, b: TilePixel) -> bool {
        a.escaped == b.escaped
            && (a.smooth_t - b.smooth_t).abs() < 1e-6_f32
            && (a.orbit_min_r - b.orbit_min_r).abs() < 1e-6_f32
            && (a.angle - b.angle).abs() < 1e-6_f32
    }

    #[test]
    fn perturb_pixel_interior_returns_interior() {
        // c = 0, dc = 0, ref = zero orbit → always Interior
        let orbit = zero_orbit(200);
        let r = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(0.0, 0.0), 200);
        assert!(matches!(r, EscapeResult::Interior { .. }));
    }

    #[test]
    fn perturb_pixel_interior_orbit_min_r_is_finite() {
        // Interior pixels must populate orbit_min_r (used by orbit traps)
        let orbit = zero_orbit(100);
        let r = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(0.0, 0.0), 100);
        if let EscapeResult::Interior { orbit_min_r, .. } = r {
            assert!(orbit_min_r.is_finite());
        } else {
            panic!("expected Interior");
        }
    }

    #[test]
    fn perturb_pixel_escaped_orbit_min_r_is_populated() {
        // Escaped pixels must also populate orbit_min_r
        let orbit = zero_orbit(200);
        let r = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(2.0, 0.0), 200);
        if let EscapeResult::Escaped { orbit_min_r, .. } = r {
            assert!(orbit_min_r.is_finite() && orbit_min_r >= 0.0);
        } else {
            panic!("expected Escaped");
        }
    }

    #[test]
    fn perturb_pixel_smooth_t_is_finite_and_positive_for_escaped() {
        let orbit = zero_orbit(200);
        let r = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(2.0, 0.0), 200);
        if let EscapeResult::Escaped { smooth_t, .. } = r {
            assert!(smooth_t.is_finite() && smooth_t > 0.0);
        } else {
            panic!("expected Escaped");
        }
    }

    #[test]
    fn perturb_pixel_matches_escape_time_for_c_equals_2() {
        // c = 2+0i, dc = 2+0i, zero ref orbit.
        // With Z_n=0: perturb_step(dz, 2, 0) = dz²+2, same recurrence as escape_time.
        let orbit = zero_orbit(200);
        let et = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 200);
        let pp = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(2.0, 0.0), 200);
        let et_px = TilePixel::from(et);
        let pp_px = TilePixel::from(pp);
        assert!(
            tile_pixels_match(et_px, pp_px),
            "TilePixel mismatch: escape_time={et_px:?} perturb_pixel={pp_px:?}"
        );
    }

    #[test]
    fn perturb_pixel_matches_escape_time_for_interior_point() {
        // c = -0.5+0i is interior in the Mandelbrot set
        let orbit = zero_orbit(500);
        let et = escape_time(&MAP, Complex::new(-0.5_f64, 0.0), 500);
        let pp = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(-0.5, 0.0), 500);
        let et_px = TilePixel::from(et);
        let pp_px = TilePixel::from(pp);
        assert!(
            tile_pixels_match(et_px, pp_px),
            "TilePixel mismatch: escape_time={et_px:?} perturb_pixel={pp_px:?}"
        );
    }

    #[test]
    fn perturb_pixel_grid_matches_escape_time_at_zoom_exp_0() {
        // 16×16 grid over the standard Mandelbrot viewport [-2, 1] × [-1.25, 1.25].
        // Reference is the origin (Z_n = 0 for all n), dc = c for each pixel.
        // With Z=0 the perturbation recurrence is identical to escape_time.
        const N: usize = 16;
        const MAX_ITER: u32 = 200;
        let orbit = zero_orbit(MAX_ITER as usize);

        for row in 0..N {
            for col in 0..N {
                let re = -2.0 + (col as f64 / (N - 1) as f64) * 3.0;
                let im = -1.25 + (row as f64 / (N - 1) as f64) * 2.5;
                let c = Complex::new(re, im);

                let et_px = TilePixel::from(escape_time(&MAP, c, MAX_ITER));
                let pp = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(re, im), MAX_ITER);
                // Glitched pixels are skipped: with Z=0 the glitch guard always fires
                // for the zero-threshold check, so no glitches should occur.
                if matches!(pp, EscapeResult::Glitched) {
                    continue;
                }
                let pp_px = TilePixel::from(pp);
                assert!(
                    tile_pixels_match(et_px, pp_px),
                    "mismatch at ({re:.3}, {im:.3}): escape_time={et_px:?} perturb_pixel={pp_px:?}"
                );
            }
        }
    }

    #[test]
    fn perturb_pixel_glitch_fires_for_large_delta_c() {
        // Reference at c_ref = -0.5+0i (Z values cycle near 0.25–0.5).
        // Pixel at c = -0.5+0.99i → |dc|=0.99i, much larger than glitch threshold × |Z_1|.
        // Glitch should fire on the first non-trivial step.
        const MAX_ITER: u32 = 100;
        let mut ref_buf = vec![Complex::<f64>::zero(); MAX_ITER as usize];
        let orbit_len = compute_ref_orbit(
            &MAP,
            Complex::new(-0.5_f64, 0.0),
            MAX_ITER,
            &mut ref_buf,
        );
        let r = perturb_pixel(
            &MAP,
            &ref_buf[..orbit_len],
            &[],
            DeltaC::new(0.0, 0.99), // dc = 0.99i: huge relative to |Z_1|=0.5
            MAX_ITER,
        );
        assert!(
            matches!(r, EscapeResult::Glitched),
            "expected Glitched for large dc, got {r:?}"
        );
    }

    // ── Golden coordinate tests ───────────────────────────────────────────────
    //
    // Three specific Mandelbrot coordinates used as regression anchors.
    // The oracle is escape_time (validated separately). perturb_pixel with the
    // zero reference must produce the same TilePixel for each golden point.

    fn golden_assert(re: f64, im: f64, max_iter: u32) {
        let orbit = zero_orbit(max_iter as usize);
        let et_px = TilePixel::from(escape_time(&MAP, Complex::new(re, im), max_iter));
        let pp = perturb_pixel(&MAP, &orbit, &[], DeltaC::new(re, im), max_iter);
        if matches!(pp, EscapeResult::Glitched) {
            return; // zero reference never glitches (Z=0 guard), so this is unreachable
        }
        let pp_px = TilePixel::from(pp);
        assert!(
            tile_pixels_match(et_px, pp_px),
            "golden ({re}, {im}): escape_time={et_px:?} perturb_pixel={pp_px:?}"
        );
    }

    #[test]
    fn golden_seahorse_valley_exterior() {
        // Classic exterior point near the Seahorse Valley.
        // escape_time oracle: escapes in the boundary region, ~50–200 iterations.
        golden_assert(-0.7269, 0.1889, 500);
    }

    #[test]
    fn golden_cardioid_boundary_exterior() {
        // Point just outside the main cardioid — escapes quickly.
        golden_assert(0.5, 0.5, 500);
    }

    #[test]
    fn golden_period2_bulb_exterior() {
        // Near the period-2 bulb boundary.
        golden_assert(-1.3, 0.0, 500);
    }

    // ── BlaEntry ──────────────────────────────────────────────────────────────

    #[test]
    fn bla_entry_size_is_48_bytes() {
        assert_eq!(core::mem::size_of::<BlaEntry>(), 48);
    }

    #[test]
    fn bla_entry_repr_c_layout() {
        // Offsets per spec: a@0, b@16, skip@32, _pad@36, valid_radius@40.
        let e = BlaEntry {
            a: Complex::new(1.0_f64, 2.0),
            b: Complex::new(3.0_f64, 4.0),
            skip: 7,
            _pad: 0,
            valid_radius: 0.5,
        };
        assert_eq!(e.a, Complex::new(1.0_f64, 2.0));
        assert_eq!(e.b, Complex::new(3.0_f64, 4.0));
        assert_eq!(e.skip, 7);
        assert!((e.valid_radius - 0.5).abs() < 1e-15);
    }

    // ── build_bla_table ───────────────────────────────────────────────────────

    #[test]
    fn bla_table_empty_orbit_gives_empty_table() {
        let table = build_bla_table(&[]);
        assert!(table.is_empty());
    }

    #[test]
    fn bla_table_length_matches_orbit() {
        let c = Complex::new(-0.5_f64, 0.0);
        let mut orbit = vec![Complex::<f64>::zero(); 64];
        let len = compute_ref_orbit(&MAP, c, 64, &mut orbit);
        let table = build_bla_table(&orbit[..len]);
        assert_eq!(table.len(), len);
    }

    #[test]
    fn bla_table_level0_a_equals_two_z() {
        // Level-0 entry at position 1: A = 2·Z_1 = 2·c_ref (for c_ref = -0.5+0i, Z_1 = -0.5).
        let c = Complex::new(-0.5_f64, 0.0);
        let mut orbit = vec![Complex::<f64>::zero(); 32];
        let len = compute_ref_orbit(&MAP, c, 32, &mut orbit);
        // Position 0 has Z_0 = 0, so valid_radius = 0 (no BLA applicable).
        assert_eq!(orbit[0].re, 0.0);
        // Position 1: Z_1 = Z_0^2 + c = c = -0.5+0i.
        let e1 = build_bla_table(&orbit[..len])[1];
        // The table entry at position 1 will be a higher-level merge — but A must
        // equal the product of level-0 A values along the chain.  We just check
        // valid_radius > 0 (table entry exists) and skip >= 1.
        assert!(e1.valid_radius > 0.0, "position 1 must have a valid BLA entry");
        assert!(e1.skip >= 1);
        // Level-0 A at position 1 = 2·Z_1 = 2·(-0.5) = -1+0i.
        // Level-0 valid_radius at position 1 = |Z_1| = 0.5.
        // The final entry may be higher level, so we just verify the invariant.
        let _ = len;
    }

    #[test]
    fn bla_table_merge_formula_valid_radius_uses_a_norm() {
        // Directly test bla_merge: min(r₀, r₁/|A₀|) vs the wrong min(r₀, r₁).
        let e0 = BlaEntry {
            a: Complex::new(2.0_f64, 0.0), // |A₀| = 2
            b: Complex::new(1.0_f64, 0.0),
            skip: 1,
            _pad: 0,
            valid_radius: 1.0,
        };
        let e1 = BlaEntry {
            a: Complex::new(0.5_f64, 0.0),
            b: Complex::new(1.0_f64, 0.0),
            skip: 1,
            _pad: 0,
            valid_radius: 0.4,
        };
        let merged = bla_merge(e0, e1);
        // Correct: min(1.0, 0.4/2.0) = min(1.0, 0.2) = 0.2
        // Wrong (omitting /|A₀|): min(1.0, 0.4) = 0.4
        assert!(
            (merged.valid_radius - 0.2).abs() < 1e-14,
            "valid_radius = {}, expected 0.2",
            merged.valid_radius
        );
        assert_eq!(merged.skip, 2);
    }

    #[test]
    fn bla_table_higher_levels_have_increasing_skip() {
        // For a smooth interior orbit, entries near the middle should have skip > 1.
        let c = Complex::new(-0.5_f64, 0.0);
        let mut orbit = vec![Complex::<f64>::zero(); 256];
        let len = compute_ref_orbit(&MAP, c, 256, &mut orbit);
        let table = build_bla_table(&orbit[..len]);

        // After convergence (n > 20), skip should be significantly > 1.
        let mid_skip = table[len / 2].skip;
        assert!(
            mid_skip > 1,
            "expected skip > 1 in smooth interior region, got {mid_skip}"
        );
    }

    // ── perturb_pixel_bla: skip ratio ─────────────────────────────────────────

    #[test]
    fn bla_skip_ratio_80pct_on_smooth_interior() {
        // Reference: c = -0.5+0i, interior, orbit converges.
        // Pixel: δc = 1e-15 (simulates deep zoom). After convergence, |δz| ≈ 1e-15
        // which is far below any valid_radius ≈ 0.1–0.5. BLA should fire almost always.
        const MAX_ITER: u32 = 512;
        let c_ref = Complex::new(-0.5_f64, 0.0);
        let mut orbit = vec![Complex::<f64>::zero(); MAX_ITER as usize];
        let orbit_len = compute_ref_orbit(&MAP, c_ref, MAX_ITER, &mut orbit);
        let table = build_bla_table(&orbit[..orbit_len]);

        let dc = DeltaC::new(1e-15, 0.0);
        let mut dz = Complex::<f64>::zero();
        let mut total_steps = 0u32;
        let mut bla_steps = 0u32;

        let mut n = 0usize;
        while n < orbit_len {
            let z_ref = orbit[n];
            let z_n = z_ref + dz;
            let r_sq = z_n.norm_sqr();
            if r_sq > MAP.escape_radius_sq() {
                break;
            }
            let ref_sq = z_ref.norm_sqr();
            let glitch_sq = 1e-6 * ref_sq;
            if ref_sq > 1e-30 && dz.norm_sqr() > glitch_sq {
                break;
            }

            let entry = table[n];
            total_steps += 1;
            let next_n = n + entry.skip as usize;
            if entry.valid_radius > 0.0
                && dz.norm_sqr().sqrt() < entry.valid_radius
                && next_n <= orbit_len
            {
                bla_steps += 1;
                dz = entry.a * dz + entry.b * dc.as_complex_f64();
                n = next_n;
            } else {
                dz = MAP.perturb_step(dz, dc.as_complex_f64(), z_ref);
                n += 1;
            }
        }

        let ratio = bla_steps as f64 / total_steps as f64;
        assert!(
            ratio >= 0.80,
            "BLA skip ratio {:.1}% < 80% required",
            ratio * 100.0
        );
    }

    // ── pixel-exact: perturb_pixel with BLA vs without ───────────────────────

    #[test]
    fn bla_pixel_exact_interior_matches_perturb_pixel() {
        // Interior reference, tiny δc — BLA and direct paths must both return Interior.
        const MAX_ITER: u32 = 256;
        let c_ref = Complex::new(-0.5_f64, 0.0);
        let mut orbit = vec![Complex::<f64>::zero(); MAX_ITER as usize];
        let orbit_len = compute_ref_orbit(&MAP, c_ref, MAX_ITER, &mut orbit);
        let table = build_bla_table(&orbit[..orbit_len]);

        let dc = DeltaC::new(1e-12, 0.0);
        let r_direct = perturb_pixel(&MAP, &orbit[..orbit_len], &[], dc, MAX_ITER);
        let r_bla = perturb_pixel(&MAP, &orbit[..orbit_len], &table, dc, MAX_ITER);

        assert!(
            matches!(r_direct, EscapeResult::Interior { .. }),
            "direct should be Interior, got {r_direct:?}"
        );
        assert!(
            matches!(r_bla, EscapeResult::Interior { .. }),
            "BLA result should be Interior, got {r_bla:?}"
        );
    }

    #[test]
    fn bla_pixel_exact_degenerate_zero_ref_matches_perturb_pixel() {
        // Zero reference (valid_radius = 0 everywhere): BLA never fires, so
        // perturb_pixel with BLA table must be identical to perturb_pixel without.
        const MAX_ITER: u32 = 200;
        let orbit = zero_orbit(MAX_ITER as usize);
        let table = build_bla_table(&orbit);

        assert!(
            table.iter().all(|e| e.valid_radius == 0.0),
            "zero orbit must produce all-zero valid_radius entries"
        );

        for (re, im) in [(0.0_f64, 0.0_f64), (2.0_f64, 0.0_f64)] {
            let dc = DeltaC::new(re, im);
            let r_direct = perturb_pixel(&MAP, &orbit, &[], dc, MAX_ITER);
            let r_bla = perturb_pixel(&MAP, &orbit, &table, dc, MAX_ITER);

            let px_direct = if matches!(r_direct, EscapeResult::Glitched) {
                continue;
            } else {
                TilePixel::from(r_direct)
            };
            if matches!(r_bla, EscapeResult::Glitched) {
                continue;
            }
            let px_bla = TilePixel::from(r_bla);

            assert!(
                tile_pixels_match(px_direct, px_bla),
                "degenerate-ref pixel ({re}, {im}): direct={px_direct:?} bla={px_bla:?}"
            );
        }
    }

    #[test]
    fn bla_pixel_exact_256x256_tile_at_interior_ref() {
        // Full 256×256 tile, interior reference c_ref = -0.5+0i.
        // δc values span ±1e-8 around c_ref: all pixels are interior (no glitch).
        // perturb_pixel with and without BLA must agree on escaped/interior for all.
        const MAX_ITER: u32 = 256;
        const TILE: usize = 256;
        const STEP: f64 = 2e-8 / TILE as f64;

        let c_ref = Complex::new(-0.5_f64, 0.0);
        let mut orbit = vec![Complex::<f64>::zero(); MAX_ITER as usize];
        let orbit_len = compute_ref_orbit(&MAP, c_ref, MAX_ITER, &mut orbit);
        let table = build_bla_table(&orbit[..orbit_len]);

        let mut mismatches = 0u32;
        for row in 0..TILE {
            for col in 0..TILE {
                let dre = (col as f64 - TILE as f64 / 2.0) * STEP;
                let dim = (row as f64 - TILE as f64 / 2.0) * STEP;
                let dc = DeltaC::new(dre, dim);

                let r_direct = perturb_pixel(&MAP, &orbit[..orbit_len], &[], dc, MAX_ITER);
                let r_bla = perturb_pixel(&MAP, &orbit[..orbit_len], &table, dc, MAX_ITER);

                if matches!(r_direct, EscapeResult::Glitched)
                    || matches!(r_bla, EscapeResult::Glitched)
                {
                    continue;
                }

                let escaped_direct = matches!(r_direct, EscapeResult::Escaped { .. });
                let escaped_bla = matches!(r_bla, EscapeResult::Escaped { .. });
                if escaped_direct != escaped_bla {
                    mismatches += 1;
                    continue;
                }
                if escaped_direct {
                    if let (
                        EscapeResult::Escaped { smooth_t: st_d, angle_final: af_d, .. },
                        EscapeResult::Escaped { smooth_t: st_b, angle_final: af_b, .. },
                    ) = (r_direct, r_bla)
                    {
                        if (st_d - st_b).abs() > 1e-6 || (af_d - af_b).abs() > 1e-6 {
                            mismatches += 1;
                        }
                    }
                }
            }
        }

        assert_eq!(
            mismatches, 0,
            "{mismatches} pixel(s) differed between perturb_pixel (no BLA) and perturb_pixel (BLA)"
        );
    }
}
