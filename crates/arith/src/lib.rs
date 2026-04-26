//! Extended-precision arithmetic — no_std, no heap.
//!
//! `arith` is the workspace-wide home for numeric primitives. `Complex<T>` is
//! defined here and used by every other crate — there is no parallel complex
//! type in the codebase.
//!
//! **Crate boundary rule:** arith owns numbers; kernel owns fractal algorithms;
//! wasm-bridge owns precision dispatch. kernel imports only
//! `arith::{Complex, DeltaC, Precision}`.
#![no_std]

use core::ops::{Add, Mul, Neg, Sub};

// ── Precision trait ────────────────────────────────────────────────────────────

/// Unified scalar contract for all precision tiers.
///
/// Implemented by `f64` (Phase 1), `F64x2` (Phase 2), and `BigFloat<N>` (Phase 2).
/// `kernel` is generic over `T: Precision` for reference orbit computation;
/// the per-pixel perturbation hot path always uses the concrete `f64` impl.
pub trait Precision:
    Copy
    + Clone
    + core::fmt::Debug
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    /// Construct from f64 — for constant-init and test paths. Not called in
    /// the hot loop.
    fn from_f64(v: f64) -> Self;
    /// Lossily downcast to f64 — used only at precision boundaries (e.g.
    /// writing a BigFloat orbit entry into the f64 ring buffer).
    fn to_f64_lossy(self) -> f64;
    /// For a scalar T, returns T² (the scalar's contribution to |z|² when
    /// computing Complex::norm_sqr = re.norm_sqr() + im.norm_sqr()).
    fn norm_sqr(self) -> Self;
}

impl Precision for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn to_f64_lossy(self) -> f64 {
        self
    }
    #[inline(always)]
    fn norm_sqr(self) -> Self {
        self * self
    }
}

// ── Complex<T> ────────────────────────────────────────────────────────────────

/// Generic complex number over any `Precision` tier.
///
/// `repr(C)` so that `&[Complex<f64>]` can be cast directly from shared WASM
/// memory orbit bytes — zero copy. The orbit buffer format is two f64 values
/// per entry (re then im), 16 bytes total, which matches this layout exactly.
///
/// This is the *only* complex type in the workspace. All crates import this.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Complex<T: Precision> {
    pub re: T,
    pub im: T,
}

impl<T: Precision> Complex<T> {
    #[inline(always)]
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    /// |z|² = re² + im² — stays in precision T.
    #[inline(always)]
    pub fn norm_sqr(self) -> T {
        self.re.norm_sqr() + self.im.norm_sqr()
    }

    /// Complex conjugate: (re, im) → (re, -im).
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl<T: Precision + PartialEq> PartialEq for Complex<T> {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl<T: Precision> Add for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: Precision> Sub for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: Precision> Mul for Complex<T> {
    type Output = Self;
    /// (a+bi)(c+di) = (ac−bd) + (ad+bc)i
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl<T: Precision> Neg for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

/// `2.0 * z_ref` in the perturbation hot loop: `δ = 2·Z_ref·δ + δ² + ΔC`.
/// Only defined for `f64` scalar — the perturbation loop is always f64.
impl Mul<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    #[inline(always)]
    fn mul(self, rhs: Complex<f64>) -> Complex<f64> {
        Complex::new(self * rhs.re, self * rhs.im)
    }
}

// ── DeltaC ────────────────────────────────────────────────────────────────────

/// Opaque per-pixel perturbation coordinate (ΔC).
///
/// v1 internals: plain `f64` pair. The type is opaque so v2 can upgrade to a
/// rescaled extended-precision representation without changing any call site in
/// `kernel`. `as_complex_f64()` is the only sanctioned extraction.
#[derive(Copy, Clone, Debug)]
pub struct DeltaC {
    re: f64,
    im: f64,
}

impl DeltaC {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Extract as `Complex<f64>` for the perturbation hot loop.
    /// Preferred over `as_f64_pair` at all kernel call sites.
    #[inline(always)]
    pub fn as_complex_f64(self) -> Complex<f64> {
        Complex::new(self.re, self.im)
    }

    /// Raw f64 pair — retained for wasm-bridge serialization paths.
    #[inline(always)]
    pub fn as_f64_pair(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

// ── Precision tier selection ───────────────────────────────────────────────────

/// Returns the minimum `BigFloat<N>` limb count for the given zoom depth.
///
/// `zoom_exp` is log₁₀ of magnification (e.g. `100.0` for 10⁻¹⁰⁰ depth).
/// Each limb contributes ~15.9 decimal digits; an 8-digit safety margin is
/// included. Panics in debug builds if `zoom_exp` exceeds the v1 cap of 300.
pub fn required_limbs(zoom_exp: f64) -> usize {
    debug_assert!(
        zoom_exp <= 300.0,
        "zoom_exp {zoom_exp} exceeds v1 cap of 300"
    );
    // Integer ceiling of (zoom_exp + 8) / 15, minimum 2.
    // 15 digits per limb (conservative vs ~15.9) ensures the safety margin
    // absorbs the fractional truncation without a floating-point ceil call.
    let digits = (zoom_exp as usize) + 8;
    let n = digits.div_ceil(15);
    if n < 2 { 2 } else { n }
}

// ── Phase 2 stubs ─────────────────────────────────────────────────────────────

/// Double-double precision (~106 mantissa bits). Phase 2.
///
/// Full `Precision` impl — including Knuth TwoSum / Veltkamp splitting — ships
/// in Phase 2. The type is public now so Phase 1 code can name it in stubs.
#[derive(Copy, Clone, Debug, Default)]
pub struct F64x2(pub f64, pub f64);

/// Fixed-size N-limb software float. Phase 2.
///
/// Full `Precision` impl — including carry-chain arithmetic — ships in Phase 2.
/// Only N = 2, 4, 8 are monomorphized in the shipped WASM module.
#[derive(Copy, Clone, Debug)]
pub struct BigFloat<const N: usize> {
    pub limbs: [u64; N],
}

impl<const N: usize> Default for BigFloat<N> {
    fn default() -> Self {
        Self { limbs: [0; N] }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Precision for f64 ────────────────────────────────────────────────────

    #[test]
    fn f64_precision_identity_elements() {
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
    }

    #[test]
    fn f64_precision_round_trip() {
        assert_eq!(f64::from_f64(3.14).to_f64_lossy(), 3.14);
    }

    #[test]
    fn f64_precision_norm_sqr() {
        assert_eq!(3.0_f64.norm_sqr(), 9.0);
        assert_eq!((-4.0_f64).norm_sqr(), 16.0);
    }

    // ── Complex<f64> construction ─────────────────────────────────────────────

    #[test]
    fn complex_new_stores_components() {
        let z = Complex::new(3.0_f64, 4.0_f64);
        assert_eq!(z.re, 3.0);
        assert_eq!(z.im, 4.0);
    }

    #[test]
    fn complex_zero_is_additive_identity() {
        let z = Complex::new(1.0_f64, 2.0_f64);
        assert_eq!(z + Complex::zero(), z);
    }

    #[test]
    fn complex_default_is_zero() {
        let z: Complex<f64> = Complex::default();
        assert_eq!(z, Complex::zero());
    }

    // ── norm_sqr ──────────────────────────────────────────────────────────────

    #[test]
    fn norm_sqr_classic_345() {
        // |3 + 4i|² = 25
        let z = Complex::new(3.0_f64, 4.0_f64);
        assert_eq!(z.norm_sqr(), 25.0);
    }

    #[test]
    fn norm_sqr_pure_real() {
        let z = Complex::new(5.0_f64, 0.0_f64);
        assert_eq!(z.norm_sqr(), 25.0);
    }

    #[test]
    fn norm_sqr_pure_imaginary() {
        let z = Complex::new(0.0_f64, 5.0_f64);
        assert_eq!(z.norm_sqr(), 25.0);
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[test]
    fn complex_add() {
        let a = Complex::new(1.0_f64, 2.0_f64);
        let b = Complex::new(3.0_f64, 4.0_f64);
        assert_eq!(a + b, Complex::new(4.0, 6.0));
    }

    #[test]
    fn complex_sub() {
        let a = Complex::new(5.0_f64, 7.0_f64);
        let b = Complex::new(2.0_f64, 3.0_f64);
        assert_eq!(a - b, Complex::new(3.0, 4.0));
    }

    #[test]
    fn complex_mul_foil() {
        // (1+2i)(3+4i) = (3−8) + (4+6)i = −5 + 10i
        let a = Complex::new(1.0_f64, 2.0_f64);
        let b = Complex::new(3.0_f64, 4.0_f64);
        let p = a * b;
        assert_eq!(p.re, -5.0);
        assert_eq!(p.im, 10.0);
    }

    #[test]
    fn complex_mul_i_squared_is_minus_one() {
        let i = Complex::new(0.0_f64, 1.0_f64);
        let result = i * i;
        assert_eq!(result.re, -1.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn complex_neg() {
        let z = Complex::new(3.0_f64, -4.0_f64);
        assert_eq!(-z, Complex::new(-3.0, 4.0));
    }

    // ── conj ─────────────────────────────────────────────────────────────────

    #[test]
    fn conj_negates_imaginary() {
        let z = Complex::new(3.0_f64, 4.0_f64);
        assert_eq!(z.conj(), Complex::new(3.0, -4.0));
    }

    #[test]
    fn mul_by_conj_gives_norm_sqr() {
        // z * conj(z) = |z|² + 0i
        let z = Complex::new(3.0_f64, 4.0_f64);
        let product = z * z.conj();
        assert_eq!(product.re, z.norm_sqr());
        assert_eq!(product.im, 0.0);
    }

    // ── f64 scalar multiply ────────────────────────────────────────────────────

    #[test]
    fn scalar_mul_scales_both_components() {
        let z = Complex::new(3.0_f64, 4.0_f64);
        let scaled = 2.0_f64 * z;
        assert_eq!(scaled, Complex::new(6.0, 8.0));
    }

    #[test]
    fn scalar_mul_zero_gives_zero() {
        let z = Complex::new(3.0_f64, 4.0_f64);
        assert_eq!(0.0_f64 * z, Complex::zero());
    }

    // ── DeltaC ────────────────────────────────────────────────────────────────

    #[test]
    fn delta_c_round_trip_via_complex() {
        let dc = DeltaC::new(1.5, -2.5);
        let c = dc.as_complex_f64();
        assert_eq!(c.re, 1.5);
        assert_eq!(c.im, -2.5);
    }

    #[test]
    fn delta_c_as_f64_pair_matches_complex() {
        let dc = DeltaC::new(1.5, -2.5);
        let (re, im) = dc.as_f64_pair();
        let c = dc.as_complex_f64();
        assert_eq!(re, c.re);
        assert_eq!(im, c.im);
    }

    // ── Mandelbrot iteration sanity check ─────────────────────────────────────

    #[test]
    fn mandelbrot_interior_stays_bounded() {
        // c = 0 → z stays at 0 forever
        let c = Complex::new(0.0_f64, 0.0_f64);
        let mut z = Complex::zero();
        for _ in 0..100 {
            z = z * z + c;
        }
        assert!(z.norm_sqr() <= 4.0);
    }

    #[test]
    fn mandelbrot_exterior_escapes() {
        // c = 2 → escapes on first step
        let c = Complex::new(2.0_f64, 0.0_f64);
        let mut z = Complex::zero();
        let mut escaped = false;
        for _ in 0..100 {
            z = z * z + c;
            if z.norm_sqr() > 4.0 {
                escaped = true;
                break;
            }
        }
        assert!(escaped);
    }

    // ── Memory layout ─────────────────────────────────────────────────────────

    #[test]
    fn complex_f64_is_16_bytes() {
        assert_eq!(core::mem::size_of::<Complex<f64>>(), 16);
    }

    #[test]
    fn complex_f64_repr_c_field_offsets() {
        // re is at offset 0, im at offset 8 — matches orbit buffer wire format
        let z = Complex::new(1.0_f64, 2.0_f64);
        let ptr = &z as *const Complex<f64> as *const f64;
        unsafe {
            assert_eq!(*ptr, 1.0);
            assert_eq!(*ptr.add(1), 2.0);
        }
    }

    // ── required_limbs ────────────────────────────────────────────────────────

    #[test]
    fn required_limbs_minimum_is_two() {
        assert_eq!(required_limbs(0.0), 2);
        assert_eq!(required_limbs(5.0), 2);
    }

    #[test]
    fn required_limbs_grows_with_depth() {
        let n50 = required_limbs(50.0);
        let n100 = required_limbs(100.0);
        let n200 = required_limbs(200.0);
        assert!(n50 < n100);
        assert!(n100 < n200);
    }
}
