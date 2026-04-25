//! Extended-precision arithmetic — no_std, no heap.
//!
//! All types are `Copy` and live on the stack. `BigFloat<N>` is a fixed-size
//! array of `u64` limbs; `N` is chosen automatically by `required_limbs`.
#![no_std]

/// Returns the minimum number of `BigFloat` limbs needed for `zoom_exp`.
///
/// `zoom_exp` is log₁₀ of the magnification depth (e.g. `100.0` for 10⁻¹⁰⁰).
/// Each limb contributes ~15.9 decimal digits. An 8-digit safety margin is
/// included. Panics in debug builds if `zoom_exp` exceeds the v1 cap of 300.
pub fn required_limbs(zoom_exp: f64) -> usize {
    debug_assert!(
        zoom_exp <= 300.0,
        "zoom_exp {zoom_exp} exceeds v1 cap of 300"
    );
    let digits = zoom_exp + 8.0;
    // Integer ceiling of digits / 15.9, minimum 2.
    let n = (digits as usize) / 15 + 1;
    if n < 2 {
        2
    } else {
        n
    }
}

/// Opaque per-pixel perturbation coordinate.
///
/// Internals are a plain `f64` pair in v1. The type is opaque so that v2 can
/// upgrade to a rescaled extended-precision representation without changing any
/// call sites in `kernel`.
#[derive(Copy, Clone, Debug)]
pub struct DeltaC {
    re: f64,
    im: f64,
}

impl DeltaC {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn as_f64_pair(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

/// Stub for the double-double type (106 mantissa bits).
/// Full implementation in Phase 2.
#[derive(Copy, Clone, Debug, Default)]
pub struct F64x2(pub f64, pub f64);

/// Stub for the fixed-size N-limb software float.
/// Full implementation in Phase 2.
#[derive(Copy, Clone, Debug)]
pub struct BigFloat<const N: usize> {
    pub limbs: [u64; N],
}

impl<const N: usize> Default for BigFloat<N> {
    fn default() -> Self {
        Self { limbs: [0; N] }
    }
}
