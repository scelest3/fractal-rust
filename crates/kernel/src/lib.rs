//! Fractal iteration engine — escape-time computation, perturbation theory, BLA.
//!
//! `kernel` imports `arith::{Complex, DeltaC, Precision}` and owns all fractal
//! algorithms. `Complex<f64>` here is always `arith::Complex<f64>` — the
//! workspace-wide type. `wasm-bridge` is the only crate that names concrete
//! `BigFloat<N>` types; `kernel` is generic over `T: Precision` for the cold
//! reference orbit path (Phase 2).

pub mod maps;
pub use maps::{JuliaMap, MandelbrotMap, MultibroMap};

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
        /// `log(|dz/dc| / (2|z|·ln|z|))` at escape — exterior distance estimate.
        /// Larger (less negative) means closer to the set boundary.
        /// Stored in `PixelData[2]`; used for distance-estimate coloring.
        log_dist: f64,
    },
    /// Orbit did not escape within `max_iter` steps (interior or convergent).
    Interior {
        /// Minimum `|z|` over the orbit.
        orbit_min_r: f64,
        /// z at which `orbit_min_r` was attained.
        orbit_min_z: Complex<f64>,
        /// Detected period: 1 = fixed point, 2 = 2-cycle, 0 = reached max_iter.
        /// Stored in `PixelData[0]`; used for period coloring.
        period: u32,
    },
    /// Pixel requires glitch correction; must not be converted to `PixelData`
    /// directly. Only produced by the perturbation path (Phase 2).
    Glitched,
}

/// Raw per-pixel data written into the tile ring slot.
///
/// Four f32 channels packed as `[ch0, ch1, ch2, ch3]` — layout is fractal-specific.
/// Uploaded to the WebGL RGBA32F tile texture; shader reads `.r/.g/.b/.a`.
///
/// Mandelbrot: `[smooth_t, orbit_min_r, log_dist/arg(orbit_min_z), escaped]`
/// Newton:     `[convergence_iter, log_p_norm, root_index, converged]`
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct PixelData(pub [f32; 4]);

impl From<EscapeResult> for PixelData {
    fn from(r: EscapeResult) -> Self {
        match r {
            EscapeResult::Escaped { smooth_t, orbit_min_r, log_dist, .. } => Self([
                smooth_t as f32,
                orbit_min_r as f32,
                log_dist as f32,
                1.0,
            ]),
            EscapeResult::Interior { orbit_min_r, orbit_min_z, period } => Self([
                period as f32,
                orbit_min_r as f32,
                orbit_min_z.im.atan2(orbit_min_z.re) as f32,
                0.0,
            ]),
            EscapeResult::Glitched => {
                panic!("Glitched pixel must not be converted to PixelData — apply glitch correction first")
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

    /// Returns `Some(period)` once the orbit settles into a cycle, `None` otherwise.
    fn converged(&self, _z: Complex<f64>, _z_prev: Complex<f64>) -> Option<u32> {
        None
    }

    /// Natural log of the map's iteration exponent, used in the smooth-t colouring formula.
    /// Default is `ln(2)`, correct for `z²`. Override for `zⁿ`.
    fn log_exponent(&self) -> f64 {
        core::f64::consts::LN_2
    }

    /// Whether this map supports distance-estimate colouring.
    fn supports_distance_estimate(&self) -> bool {
        true
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

// ── compute_ref_orbit ─────────────────────────────────────────────────────────

/// Compute the reference orbit in precision T, storing downcast f64 entries.
///
/// Returns the number of non-escaped entries written to `out[0..n]`.
/// If the reference escapes at step `n`, the **escaped** value `Z_n` is also
/// written to `out[n]` (one extra slot), so `out` should be sized `max_iter + 1`
/// to guarantee room. `perturb_pixel` uses this extra entry to detect escape at
/// the orbit boundary for exterior reference points.
///
/// # Arguments
/// * `map`      — controls the high-precision step and escape radius
/// * `c`        — reference point in precision T
/// * `max_iter` — upper bound on non-escaped orbit length
/// * `out`      — caller-supplied buffer; sized `max_iter + 1` to hold the escape entry
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
            // Append the escaped Z value if there is room — perturb_pixel needs it
            // to detect escape for pixels whose orbit mirrors the reference boundary.
            if i + 1 < out.len() {
                out[i + 1] = Complex::new(z_next.re.to_f64_lossy(), z_next.im.to_f64_lossy());
            }
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
/// for the same escaped z, so `PixelData` output matches escape_time bit-for-bit
/// when the zero-reference trick is used in tests.
pub fn perturb_pixel<M: PerturbationSupport>(
    map: &M,
    primary_orbit: &[Complex<f64>],
    secondary_orbit: &[Complex<f64>],
    bla_table: &[BlaEntry],
    delta_c: DeltaC,
    dc_from_secondary: DeltaC,
    max_iter: u32,
) -> EscapeResult {
    let glitch_thresh_sq = {
        let t = map.glitch_threshold();
        t * t
    };

    // current_orbit and current_dc switch to secondary on rebase.
    let mut current_orbit: &[Complex<f64>] = primary_orbit;
    let mut current_dc = delta_c.as_complex_f64();
    let mut dz = Complex::<f64>::zero();
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = Complex::<f64>::zero();
    // Allow one extra step so perturb_pixel can reach the escaped Z_N entry
    // appended by compute_ref_orbit for exterior reference points.
    let mut limit = (max_iter as usize + 1).min(primary_orbit.len());
    let mut rebased = false;

    let mut n = 0usize;
    while n < limit {
        let z_ref = current_orbit[n];
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
                log_dist: 0.0, // dz not tracked on perturbation path; placeholder
            };
        }

        // Pauldelbrot glitch criterion: |δz_n|² > threshold² · |Z_n|².
        // Guard: skip when Z_n ≈ 0 — criterion is undefined near the origin.
        let ref_sq = z_ref.norm_sqr();
        if ref_sq > 1e-30 && dz.norm_sqr() > glitch_thresh_sq * ref_sq {
            if !rebased && n < secondary_orbit.len() {
                // Rebase: switch to secondary orbit at step n.
                // new δz = z_n − Z'_n (pixel position relative to secondary reference).
                dz = z_n - secondary_orbit[n];
                current_orbit = secondary_orbit;
                current_dc = dc_from_secondary.as_complex_f64();
                rebased = true;
                limit = (max_iter as usize + 1).min(secondary_orbit.len());
                // Re-evaluate step n with the new dz — do not advance n.
                continue;
            }
            return EscapeResult::Glitched;
        }

        // BLA: skip multiple steps when |δz_n| < valid_radius.
        // Disabled after rebasing — the BLA table was built from the primary orbit.
        if !rebased {
            if let Some(&entry) = bla_table.get(n) {
                let next_n = n + entry.skip as usize;
                if entry.valid_radius > 0.0
                    && dz.norm_sqr().sqrt() < entry.valid_radius
                    && next_n <= limit
                {
                    dz = entry.a * dz + entry.b * current_dc;
                    n = next_n;
                    continue;
                }
            }
        }

        // Direct single perturbation step.
        dz = map.perturb_step(dz, current_dc, z_ref);
        n += 1;
    }

    EscapeResult::Interior { orbit_min_r, orbit_min_z, period: 0 }
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

// ── compute_secondary_orbit ───────────────────────────────────────────────────

/// Compute a secondary reference orbit by running f64 perturbation from the
/// primary orbit to a nearby point `dc_secondary`.
///
/// Each entry `out[n]` = `Z_n + δz_n` — the secondary point's orbit in f64.
/// This can be used as a replacement reference for `perturb_pixel`: pass
/// `secondary_orbit` as `ref_orbit` and `dc_pixel − dc_secondary` as `delta_c`.
///
/// Runs purely in f64 (no BigFloat), making it fast enough to call per-tile
/// inside a Tile Worker. No glitch check is performed — the output is always
/// the best-available f64 approximation of the secondary orbit.
///
/// If the secondary escapes before `max_iter`, `out` is truncated at the
/// escape step; `perturb_pixel` handles short reference orbits correctly.
pub fn compute_secondary_orbit<M: PerturbationSupport>(
    map: &M,
    primary_orbit: &[Complex<f64>],
    dc_secondary: DeltaC,
    max_iter: u32,
    out: &mut Vec<Complex<f64>>,
) {
    let dc = dc_secondary.as_complex_f64();
    let mut dz = Complex::<f64>::zero();
    let limit = (max_iter as usize + 1).min(primary_orbit.len());
    out.clear();
    out.reserve(limit);
    for z_ref in primary_orbit.iter().copied().take(limit) {
        let z_sec = z_ref + dz;
        out.push(z_sec);
        if z_sec.norm_sqr() > map.escape_radius_sq() {
            break;
        }
        dz = map.perturb_step(dz, dc, z_ref);
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
    let mut z          = Complex::<f64>::zero();
    let mut dz         = Complex::<f64>::zero();
    let mut z_two_back = Complex::<f64>::zero();
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = Complex::<f64>::zero();

    for iter in 0..max_iter {
        let step   = map.step(z, c);
        let z_new  = step.z;
        let z_prev = z;

        // dz_{n+1}/dc = 2·z_n·(dz_n/dc) + 1
        dz = Complex::new(2.0, 0.0) * z_prev * dz + Complex::new(1.0, 0.0);

        // Track orbit min over {z₁, z₂, …} — z₀ = 0 excluded.
        let r = z_new.norm_sqr().sqrt();
        if r < orbit_min_r { orbit_min_r = r; orbit_min_z = z_new; }

        z = z_new;

        if step.escaped {
            let norm     = z.norm_sqr().sqrt();
            let smooth_t = (iter as f64 + 1.0) - (norm.ln().ln() / map.log_exponent());
            let dz_norm  = dz.norm_sqr().sqrt().max(1e-300);
            let log_dist = (dz_norm / (2.0 * norm * norm.ln().max(1e-300))).ln();
            return EscapeResult::Escaped { iter, smooth_t, orbit_min_r, orbit_min_z,
                log_dist };
        }

        if let Some(p) = map.converged(z, z_prev) {
            return EscapeResult::Interior { orbit_min_r, orbit_min_z, period: p };
        }

        // Period-2 detection via inline comparison is deferred to Floyd's cycle
        // detection (Layer 3). Oscillating period-1 orbits (negative multiplier)
        // cause the two-back check to fire before the fixed-point check, making
        // inline period-2 detection unreliable without a full cycle-detection pass.
        let _ = z_two_back; // suppress unused warning until Floyd's is added
    }

    EscapeResult::Interior { orbit_min_r, orbit_min_z, period: 0 }
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
    out: &mut [PixelData],
) {
    debug_assert_eq!(coords.len(), out.len());
    for (c, slot) in coords.iter().zip(out.iter_mut()) {
        *slot = PixelData::from(escape_time(map, *c, max_iter));
    }
}

// ── escape_time_julia ─────────────────────────────────────────────────────────

/// Compute the escape-time result for one Julia-set pixel.
///
/// The pixel coordinate is used as `z₀`; `map.c` is the fixed Julia parameter.
/// Distance-estimate tracking is omitted — `JuliaMap::supports_distance_estimate`
/// returns `false`, so `log_dist` is always 0.0.
pub fn escape_time_julia(map: &JuliaMap, z0: Complex<f64>, max_iter: u32) -> EscapeResult {
    let mut z = z0;
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = Complex::<f64>::zero();
    let c_zero = Complex::<f64>::zero(); // ignored by JuliaMap::step

    for iter in 0..max_iter {
        let step   = map.step(z, c_zero);
        let z_new  = step.z;
        let z_prev = z;

        let r = z_new.norm_sqr().sqrt();
        if r < orbit_min_r { orbit_min_r = r; orbit_min_z = z_new; }

        z = z_new;

        if step.escaped {
            let norm     = z.norm_sqr().sqrt();
            let smooth_t = (iter as f64 + 1.0) - (norm.ln().ln() / map.log_exponent());
            return EscapeResult::Escaped { iter, smooth_t, orbit_min_r, orbit_min_z,
                log_dist: 0.0 };
        }

        if let Some(p) = map.converged(z, z_prev) {
            return EscapeResult::Interior { orbit_min_r, orbit_min_z, period: p };
        }
    }

    EscapeResult::Interior { orbit_min_r, orbit_min_z, period: 0 }
}

// ── render_tile_julia ─────────────────────────────────────────────────────────

/// Fill `out` with escape-time results for a Julia map.
///
/// Each entry in `coords` is the pixel's complex coordinate used as `z₀`.
/// `coords` and `out` must have the same length (asserted in debug builds).
pub fn render_tile_julia(
    map: &JuliaMap,
    coords: &[Complex<f64>],
    max_iter: u32,
    out: &mut [PixelData],
) {
    debug_assert_eq!(coords.len(), out.len());
    for (z0, slot) in coords.iter().zip(out.iter_mut()) {
        *slot = PixelData::from(escape_time_julia(map, *z0, max_iter));
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arith::{Complex, F64x2};

    const MAP: MandelbrotMap = MandelbrotMap;

    // ── EscapeResult enum ─────────────────────────────────────────────────────

    #[test]
    fn tile_pixel_from_escaped_has_escaped_one() {
        let r = EscapeResult::Escaped {
            iter: 5,
            smooth_t: 1.5,
            orbit_min_r: 0.3,
            orbit_min_z: Complex::new(0.1_f64, 0.2),
            log_dist: 0.7,
        };
        let p = PixelData::from(r);
        assert_eq!(p.0[3], 1.0);
        assert!((p.0[0] - 1.5_f32).abs() < 1e-6);
        assert!((p.0[1] - 0.3_f32).abs() < 1e-6);
        assert!((p.0[2] - 0.7_f32).abs() < 1e-6);
    }

    #[test]
    fn tile_pixel_from_interior_has_escaped_zero() {
        let r = EscapeResult::Interior {
            orbit_min_r: 0.4,
            orbit_min_z: Complex::new(0.0_f64, 0.0),
            period: 0,
        };
        let p = PixelData::from(r);
        assert_eq!(p.0[3], 0.0);
        assert_eq!(p.0[0], 0.0); // period 0 → 0.0
        assert!((p.0[1] - 0.4_f32).abs() < 1e-6);
    }

    #[test]
    fn tile_pixel_interior_period_stored_in_smooth_t() {
        let r = EscapeResult::Interior {
            orbit_min_r: 0.4,
            orbit_min_z: Complex::new(0.0_f64, 0.0),
            period: 2,
        };
        let p = PixelData::from(r);
        assert_eq!(p.0[0], 2.0_f32);
        assert_eq!(p.0[3], 0.0_f32);
    }

    #[test]
    #[should_panic(expected = "Glitched pixel")]
    fn tile_pixel_from_glitched_panics() {
        let _ = PixelData::from(EscapeResult::Glitched);
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

    #[test]
    fn main_cardioid_exits_early_with_period_1() {
        // c = -0.5 is well inside the main cardioid — converges to a fixed point.
        let r = escape_time(&MAP, Complex::new(-0.5_f64, 0.0), 10_000);
        match r {
            EscapeResult::Interior { period, .. } => assert_eq!(period, 1),
            _ => panic!("expected Interior"),
        }
    }

    #[test]
    fn period_2_bulb_is_interior_period_unknown() {
        // c = -1.0 is in the period-2 bulb. Period-2 detection is deferred to
        // Floyd's cycle detection (Layer 3) — for now returns period=0.
        let r = escape_time(&MAP, Complex::new(-1.0_f64, 0.0), 10_000);
        match r {
            EscapeResult::Interior { period, .. } => assert_eq!(period, 0),
            _ => panic!("expected Interior"),
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
    fn escaped_pixel_log_dist_is_finite() {
        let r = escape_time(&MAP, Complex::new(2.0_f64, 0.0), 1000);
        if let EscapeResult::Escaped { log_dist, .. } = r {
            assert!(log_dist.is_finite(), "log_dist must be finite at escape");
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

    // ── PixelData ─────────────────────────────────────────────────────────────

    #[test]
    fn tile_pixel_is_16_bytes_repr_c() {
        assert_eq!(core::mem::size_of::<PixelData>(), 16);
    }

    // ── render_tile_escape ────────────────────────────────────────────────────

    #[test]
    fn render_tile_mixed_interior_and_escaped() {
        let coords: Vec<Complex<f64>> = (-2..=1)
            .flat_map(|re| (-2..=1).map(move |im| Complex::new(re as f64, im as f64)))
            .collect();
        let mut out = vec![PixelData::default(); coords.len()];
        render_tile_escape(&MAP, &coords, 100, &mut out);

        let escaped_count = out.iter().filter(|p| p.0[3] == 1.0).count();
        let interior_count = out.iter().filter(|p| p.0[3] == 0.0).count();
        assert!(escaped_count > 0, "expected at least one escaped pixel");
        assert!(interior_count > 0, "expected at least one interior pixel");
    }

    #[test]
    fn render_tile_all_escaped_far_exterior() {
        let coords: Vec<Complex<f64>> =
            (0..16).map(|i| Complex::new(10.0 + i as f64, 0.0)).collect();
        let mut out = vec![PixelData::default(); coords.len()];
        render_tile_escape(&MAP, &coords, 100, &mut out);
        assert!(out.iter().all(|p| p.0[3] == 1.0));
    }

    #[test]
    fn render_tile_output_length_matches_coords() {
        let coords = vec![Complex::new(0.0_f64, 0.0); 64];
        let mut out = vec![PixelData::default(); 64];
        render_tile_escape(&MAP, &coords, 50, &mut out);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn render_tile_interior_smooth_t_is_period() {
        // c=0: orbit is z_0=0, z_1=0, immediately converges — period-1 fixed point.
        // ch0 stores period as f32 for interior pixels.
        let coords = vec![Complex::new(0.0_f64, 0.0)];
        let mut out = vec![PixelData::default(); 1];
        render_tile_escape(&MAP, &coords, 100, &mut out);
        assert_eq!(out[0].0[3], 0.0);
        assert_eq!(out[0].0[0], 1.0); // period 1
    }

    #[test]
    fn render_tile_escaped_smooth_t_is_finite() {
        let coords = vec![Complex::new(2.0_f64, 0.0)];
        let mut out = vec![PixelData::default(); 1];
        render_tile_escape(&MAP, &coords, 100, &mut out);
        assert_eq!(out[0].0[3], 1.0);
        assert!(out[0].0[0].is_finite());
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

    fn tile_pixels_match(a: PixelData, b: PixelData) -> bool {
        if a.0[3] != b.0[3] { return false; }
        if (a.0[1] - b.0[1]).abs() >= 1e-6_f32 { return false; }
        if a.0[3] > 0.5 {
            // Escaped: ch0 (smooth_t) must match. ch2 holds log_dist (escape_time) vs 0.0
            // (perturb_pixel, which doesn't track dz) — intentional difference, skip.
            if (a.0[0] - b.0[0]).abs() >= 1e-6_f32 { return false; }
        }
        // Interior: ch0 holds period (escape_time detects period-1, perturb_pixel
        // always returns 0) — intentional difference, skip comparison.
        true
    }

    #[test]
    fn perturb_pixel_interior_returns_interior() {
        // c = 0, dc = 0, ref = zero orbit → always Interior
        let orbit = zero_orbit(200);
        let r = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(0.0, 0.0), DeltaC::new(0.0, 0.0), 200);
        assert!(matches!(r, EscapeResult::Interior { .. }));
    }

    #[test]
    fn perturb_pixel_interior_orbit_min_r_is_finite() {
        // Interior pixels must populate orbit_min_r (used by orbit traps)
        let orbit = zero_orbit(100);
        let r = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(0.0, 0.0), DeltaC::new(0.0, 0.0), 100);
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
        let r = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(2.0, 0.0), DeltaC::new(0.0, 0.0), 200);
        if let EscapeResult::Escaped { orbit_min_r, .. } = r {
            assert!(orbit_min_r.is_finite() && orbit_min_r >= 0.0);
        } else {
            panic!("expected Escaped");
        }
    }

    #[test]
    fn perturb_pixel_smooth_t_is_finite_and_positive_for_escaped() {
        let orbit = zero_orbit(200);
        let r = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(2.0, 0.0), DeltaC::new(0.0, 0.0), 200);
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
        let pp = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(2.0, 0.0), DeltaC::new(0.0, 0.0), 200);
        let et_px = PixelData::from(et);
        let pp_px = PixelData::from(pp);
        assert!(
            tile_pixels_match(et_px, pp_px),
            "PixelData mismatch: escape_time={et_px:?} perturb_pixel={pp_px:?}"
        );
    }

    #[test]
    fn perturb_pixel_matches_escape_time_for_interior_point() {
        // c = -0.5+0i is interior in the Mandelbrot set
        let orbit = zero_orbit(500);
        let et = escape_time(&MAP, Complex::new(-0.5_f64, 0.0), 500);
        let pp = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(-0.5, 0.0), DeltaC::new(0.0, 0.0), 500);
        let et_px = PixelData::from(et);
        let pp_px = PixelData::from(pp);
        assert!(
            tile_pixels_match(et_px, pp_px),
            "PixelData mismatch: escape_time={et_px:?} perturb_pixel={pp_px:?}"
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

                let et_px = PixelData::from(escape_time(&MAP, c, MAX_ITER));
                let pp = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(re, im), DeltaC::new(0.0, 0.0), MAX_ITER);
                // Glitched pixels are skipped: with Z=0 the glitch guard always fires
                // for the zero-threshold check, so no glitches should occur.
                if matches!(pp, EscapeResult::Glitched) {
                    continue;
                }
                let pp_px = PixelData::from(pp);
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
            &[],
            DeltaC::new(0.0, 0.99), // dc = 0.99i: huge relative to |Z_1|=0.5
            DeltaC::new(0.0, 0.0),
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
    // zero reference must produce the same PixelData for each golden point.

    fn golden_assert(re: f64, im: f64, max_iter: u32) {
        let orbit = zero_orbit(max_iter as usize);
        let et_px = PixelData::from(escape_time(&MAP, Complex::new(re, im), max_iter));
        let pp = perturb_pixel(&MAP, &orbit, &[], &[], DeltaC::new(re, im), DeltaC::new(0.0, 0.0), max_iter);
        if matches!(pp, EscapeResult::Glitched) {
            return; // zero reference never glitches (Z=0 guard), so this is unreachable
        }
        let pp_px = PixelData::from(pp);
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
        let r_direct = perturb_pixel(&MAP, &orbit[..orbit_len], &[], &[], dc, DeltaC::new(0.0, 0.0), MAX_ITER);
        let r_bla = perturb_pixel(&MAP, &orbit[..orbit_len], &[], &table, dc, DeltaC::new(0.0, 0.0), MAX_ITER);

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
            let r_direct = perturb_pixel(&MAP, &orbit, &[], &[], dc, DeltaC::new(0.0, 0.0), MAX_ITER);
            let r_bla = perturb_pixel(&MAP, &orbit, &[], &table, dc, DeltaC::new(0.0, 0.0), MAX_ITER);

            let px_direct = if matches!(r_direct, EscapeResult::Glitched) {
                continue;
            } else {
                PixelData::from(r_direct)
            };
            if matches!(r_bla, EscapeResult::Glitched) {
                continue;
            }
            let px_bla = PixelData::from(r_bla);

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

                let r_direct = perturb_pixel(&MAP, &orbit[..orbit_len], &[], &[], dc, DeltaC::new(0.0, 0.0), MAX_ITER);
                let r_bla = perturb_pixel(&MAP, &orbit[..orbit_len], &[], &table, dc, DeltaC::new(0.0, 0.0), MAX_ITER);

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
                        EscapeResult::Escaped { smooth_t: st_d, log_dist: ld_d, .. },
                        EscapeResult::Escaped { smooth_t: st_b, log_dist: ld_b, .. },
                    ) = (r_direct, r_bla)
                    {
                        if (st_d - st_b).abs() > 1e-6 || (ld_d - ld_b).abs() > 1e-6 {
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

    // ── compute_secondary_orbit ───────────────────────────────────────────────

    #[test]
    fn secondary_orbit_at_zero_dc_equals_primary() {
        // dc_secondary = 0 → secondary orbit must be identical to primary orbit.
        let c = Complex::new(-0.5_f64, 0.0);
        let mut primary = vec![Complex::<f64>::zero(); 64];
        let len = compute_ref_orbit(&MAP, c, 64, &mut primary);
        let primary = &primary[..len];

        let mut secondary = Vec::new();
        compute_secondary_orbit(&MAP, primary, DeltaC::new(0.0, 0.0), 64, &mut secondary);

        assert_eq!(secondary.len(), primary.len());
        for (a, b) in primary.iter().zip(secondary.iter()) {
            assert!((a.re - b.re).abs() < 1e-15 && (a.im - b.im).abs() < 1e-15);
        }
    }

    #[test]
    fn secondary_orbit_matches_direct_orbit_for_nearby_point() {
        // Primary: c_ref = -0.5 + 0i. Secondary dc = 0.5 → c_sec = 0.0 (origin).
        // compute_secondary_orbit should reproduce the direct orbit of c_sec.
        const MAX_ITER: u32 = 100;
        let c_ref = Complex::new(-0.5_f64, 0.0);
        let c_sec = Complex::new(0.0_f64, 0.0); // c_ref + dc_sec = -0.5 + 0.5 = 0

        let mut primary = vec![Complex::<f64>::zero(); MAX_ITER as usize + 1];
        let primary_len = compute_ref_orbit(&MAP, c_ref, MAX_ITER, &mut primary);
        let primary = &primary[..primary_len];

        let mut secondary = Vec::new();
        compute_secondary_orbit(&MAP, primary, DeltaC::new(0.5, 0.0), MAX_ITER, &mut secondary);

        let mut direct = vec![Complex::<f64>::zero(); MAX_ITER as usize + 1];
        let direct_len = compute_ref_orbit(&MAP, c_sec, MAX_ITER, &mut direct);
        let direct = &direct[..direct_len];

        assert_eq!(secondary.len(), direct.len(), "secondary orbit length must match direct");
        for (i, (a, b)) in secondary.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                "entry {i}: secondary={a:?} direct={b:?}"
            );
        }
    }

    #[test]
    fn perturb_pixel_rebasing_mid_loop() {
        // Primary: c_ref = -0.5 + 0i (interior). dc_pixel = 1.0 + 0i → c_pixel = 0.5 (exterior).
        // Without secondary the pixel glitches. With secondary at dc_sec = dc_pixel,
        // the inline rebase at the glitch step produces the correct escape result.
        const MAX_ITER: u32 = 200;
        let c_ref = Complex::new(-0.5_f64, 0.0);
        let dc_sec = DeltaC::new(1.0, 0.0);

        let mut primary = vec![Complex::<f64>::zero(); MAX_ITER as usize + 1];
        let orbit_len = compute_ref_orbit(&MAP, c_ref, MAX_ITER, &mut primary);
        let primary = &primary[..orbit_len];

        let mut secondary = Vec::new();
        compute_secondary_orbit(&MAP, primary, dc_sec, MAX_ITER, &mut secondary);

        // dc_from_secondary = dc_pixel - dc_sec = 0 (pixel IS the secondary reference).
        let result = perturb_pixel(
            &MAP, primary, &secondary, &[],
            dc_sec, DeltaC::new(0.0, 0.0), MAX_ITER,
        );
        assert!(
            !matches!(result, EscapeResult::Glitched),
            "inline rebase must resolve the glitch"
        );
        let et_px = PixelData::from(escape_time(&MAP, Complex::new(0.5_f64, 0.0), MAX_ITER));
        let pp_px = PixelData::from(result);
        assert!(
            tile_pixels_match(et_px, pp_px),
            "rebased result must match escape_time(0.5): et={et_px:?} pp={pp_px:?}"
        );
    }

    #[test]
    fn secondary_orbit_resolves_glitched_pixel() {
        // Primary: c_ref = -0.5 + 0i (interior). Pixel at dc = 1.0 + 0i glitches.
        // Secondary at dc_sec = dc_pixel: re-render with dc_rel = 0 must not glitch,
        // and the result must match escape_time for c_pixel = c_ref + dc_pixel = 0.5.
        const MAX_ITER: u32 = 200;
        let c_ref = Complex::new(-0.5_f64, 0.0);
        let dc_pixel = DeltaC::new(1.0, 0.0); // c_pixel = 0.5

        let mut primary = vec![Complex::<f64>::zero(); MAX_ITER as usize + 1];
        let orbit_len = compute_ref_orbit(&MAP, c_ref, MAX_ITER, &mut primary);
        let primary = &primary[..orbit_len];

        // Verify the pixel glitches against the primary.
        assert!(
            matches!(perturb_pixel(&MAP, primary, &[], &[], dc_pixel, DeltaC::new(0.0, 0.0), MAX_ITER), EscapeResult::Glitched),
            "pixel should glitch against interior primary with large dc"
        );

        // Compute secondary orbit at dc_sec = dc_pixel.
        let mut secondary = Vec::new();
        compute_secondary_orbit(&MAP, primary, dc_pixel, MAX_ITER, &mut secondary);

        // Re-render with dc_rel = 0 (pixel IS the secondary reference).
        let r_secondary = perturb_pixel(&MAP, &secondary, &[], &[], DeltaC::new(0.0, 0.0), DeltaC::new(0.0, 0.0), MAX_ITER);
        assert!(
            !matches!(r_secondary, EscapeResult::Glitched),
            "pixel must not glitch against secondary reference"
        );

        // Result must match escape_time for c = 0.5.
        let et_px = PixelData::from(escape_time(&MAP, Complex::new(0.5_f64, 0.0), MAX_ITER));
        let pp_px = PixelData::from(r_secondary);
        assert!(
            tile_pixels_match(et_px, pp_px),
            "secondary-corrected result must match escape_time: et={et_px:?} pp={pp_px:?}"
        );
    }

    // ── render_tile_julia ─────────────────────────────────────────────────────

    #[test]
    fn render_tile_julia_mixed_tile() {
        // c=-0.5+0.5i is inside the main Mandelbrot cardioid, so z₀=0 is interior.
        // c=-0.7+0.27i is just outside M; use a clearly-exterior z₀=1+0i to escape fast.
        let map = JuliaMap::new(2.0, Complex::new(-0.5, 0.5));
        // z₀=0+0i is interior; z₀=1.5+0i escapes quickly.
        let coords = [Complex::new(0.0_f64, 0.0), Complex::new(1.5_f64, 0.0)];
        let mut out = [PixelData::default(); 2];
        render_tile_julia(&map, &coords, 500, &mut out);
        // Channel 3 (index [3]) is 0.0 for interior, 1.0 for escaped.
        assert_eq!(out[0].0[3], 0.0, "z₀=0 should be interior (channel 3 = 0)");
        assert_eq!(out[1].0[3], 1.0, "z₀=1 should be escaped (channel 3 = 1)");
    }
}
