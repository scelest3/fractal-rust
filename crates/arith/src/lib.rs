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

use core::ops::{Add, Div, Mul, Neg, Sub};

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
impl Div for Complex<f64> {
    type Output = Self;
    /// `a / b = a · conj(b) / |b|²`
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sqr();
        let numer = self * rhs.conj();
        Complex::new(numer.re / denom, numer.im / denom)
    }
}

impl Div<f64> for Complex<f64> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Complex::new(self.re / rhs, self.im / rhs)
    }
}

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

// ── F64x2 double-double arithmetic ────────────────────────────────────────────

/// Double-double precision (~106 mantissa bits).
///
/// Represents the exact value `hi + lo` where `|lo| ≤ ½ ulp(hi)`.
/// Uses Knuth TwoSum for addition and Veltkamp splitting for multiplication.
/// No compiler barriers needed — WASM and native targets are strictly IEEE 754,
/// and .cargo/config.toml carries no fast-math flags.
#[derive(Copy, Clone, Debug, Default)]
pub struct F64x2(pub f64, pub f64);

/// Knuth TwoSum: returns `(s, e)` such that `a + b == s + e` exactly.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Veltkamp split: splits `a` into `(hi, lo)` each with ≤ 26 mantissa bits,
/// so that `hi + lo == a` exactly and `hi * lo` has no rounding error.
#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    let t = 134_217_729.0_f64 * a; // 2^27 + 1
    let hi = t - (t - a);
    let lo = a - hi;
    (hi, lo)
}

/// TwoProd: returns `(p, e)` such that `a * b == p + e` exactly.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let e = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo;
    (p, e)
}

impl F64x2 {
    /// Renormalize so that `|lo| ≤ ½ ulp(hi)`.
    #[inline(always)]
    fn renorm(hi: f64, lo: f64) -> Self {
        let (h, l) = two_sum(hi, lo);
        F64x2(h, l)
    }
}

impl Precision for F64x2 {
    #[inline(always)]
    fn zero() -> Self { F64x2(0.0, 0.0) }
    #[inline(always)]
    fn one() -> Self { F64x2(1.0, 0.0) }
    #[inline(always)]
    fn from_f64(v: f64) -> Self { F64x2(v, 0.0) }
    #[inline(always)]
    fn to_f64_lossy(self) -> f64 { self.0 }
    #[inline(always)]
    fn norm_sqr(self) -> Self { self * self }
}

impl Add for F64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.0, rhs.0);
        F64x2::renorm(s, e + self.1 + rhs.1)
    }
}

impl Sub for F64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { self + (-rhs) }
}

impl Mul for F64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = two_prod(self.0, rhs.0);
        // Cross terms: self.0*rhs.1 and self.1*rhs.0 are O(eps²); self.1*rhs.1 dropped.
        F64x2::renorm(p, e + self.0 * rhs.1 + self.1 * rhs.0)
    }
}

impl Neg for F64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { F64x2(-self.0, -self.1) }
}

// ── BigFloat<N> fixed-point arithmetic ───────────────────────────────────────

/// Fixed-point signed real in `[−8, 8]`, N × 64-bit two's complement limbs.
///
/// **Not a general-purpose float.** The `[−8, 8]` range is sufficient for all
/// Mandelbrot reference-orbit values (`|z| < 2` before escape) and intermediate
/// `z² + c` computations (magnitude up to 6). To generalise to arbitrary reals,
/// add an explicit `exponent: i32` field and adjust carry-chain arithmetic to
/// account for the radix-point shift.
///
/// Representation: `value = (two's-complement N×64-bit integer) × 2^{−(N−1)×64−60}`.
/// `limbs[0]` is the most-significant word; its MSB is the sign bit.
/// The binary point sits between bits 63 and 60 of `limbs[0]` (4 integer bits).
///
/// Only `N = 2, 4, 8` are monomorphized in the shipped WASM module.
#[derive(Copy, Clone, Debug)]
pub struct BigFloat<const N: usize> {
    pub limbs: [u64; N],
}

impl<const N: usize> Default for BigFloat<N> {
    fn default() -> Self { Self { limbs: [0; N] } }
}

// ── Internal multi-word helpers (big-endian u64 slices) ───────────────────────

/// Multi-word two's complement negation in place.
fn negate_words(w: &mut [u64]) {
    let mut carry: u64 = 1;
    for i in (0..w.len()).rev() {
        let (v, c) = (!w[i]).overflowing_add(carry);
        w[i] = v;
        carry = c as u64;
    }
}

/// Unsigned schoolbook N×N multiplication, producing the N-word fixed-point
/// result after a right-shift of `(N−1)×64 + 60` bits.
///
/// Uses a `[u64; 16]` scratch buffer (supports N ≤ 8).
fn mul_unsigned_fixed(a: &[u64], b: &[u64], out: &mut [u64]) {
    let n = a.len();
    debug_assert!(n <= 8 && n == b.len() && n == out.len());
    let mut p = [0u64; 16]; // 2N little-endian product words

    // Schoolbook: accumulate a_le[i] × b_le[j] into p[i+j] (little-endian indexing).
    for i in 0..n {
        let ai = a[n - 1 - i] as u128;
        for j in 0..n {
            let partial = ai * (b[n - 1 - j] as u128);
            let lo = partial as u64;
            let hi = (partial >> 64) as u64;
            let (s, c) = p[i + j].overflowing_add(lo);
            p[i + j] = s;
            let (s2, c2) = p[i + j + 1].overflowing_add(hi.wrapping_add(c as u64));
            p[i + j + 1] = s2;
            // Propagate carry.
            let mut carry = c2 as u64;
            let mut k = i + j + 2;
            while carry > 0 && k < 2 * n {
                let (s, c) = p[k].overflowing_add(carry);
                p[k] = s;
                carry = c as u64;
                k += 1;
            }
        }
    }

    // Round to nearest: the "round bit" is bit 59 of p[n-1] (LE).
    // Add 2^{(n-1)*64+59} = (1<<59) to p[n-1] before extracting.
    let (rv, rc) = p[n - 1].overflowing_add(1u64 << 59);
    p[n - 1] = rv;
    if rc {
        let mut carry = 1u64;
        let mut k = n;
        while carry > 0 && k < 2 * n {
            let (v, c) = p[k].overflowing_add(carry);
            p[k] = v;
            carry = c as u64;
            k += 1;
        }
    }

    // Extract: right-shift by (n−1)×64 + 60 bits.
    // In big-endian output index j: out[j] = (p[2n-2-j] >> 60) | (p[2n-1-j] << 4).
    for j in 0..n {
        let lo = p[2 * n - 2 - j] >> 60;
        let hi = p[2 * n - 1 - j] << 4;
        out[j] = lo | hi;
    }
}

// ── Precision impl ─────────────────────────────────────────────────────────────

impl<const N: usize> Precision for BigFloat<N> {
    fn zero() -> Self { Self::default() }

    fn one() -> Self {
        // 1.0 = bit 60 of limbs[0] set (since 2^60 × 2^{-60} = 1.0).
        let mut r = Self::default();
        r.limbs[0] = 1u64 << 60;
        r
    }

    fn from_f64(v: f64) -> Self {
        let mut result = Self::default();
        if v == 0.0 || v.is_nan() { return result; }

        let bits = v.to_bits();
        let sign_negative = bits >> 63 != 0;
        let exp_biased = ((bits >> 52) & 0x7ff) as i32;
        let exp = exp_biased - 1023;
        // Mantissa: 53 bits with implicit leading 1 for normalized values.
        let mantissa: u64 = if exp_biased == 0 {
            (bits & 0x000f_ffff_ffff_ffff) << 1
        } else {
            (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
        };

        // The 53-bit mantissa LSB sits at bit `shift` from the fixed-point LSB.
        // shift = exp - 52 + frac_bits = exp - 52 + (N-1)*64 + 60 = exp + (N-1)*64 + 8.
        let shift = exp + (N as i32 - 1) * 64 + 8;
        if shift < 0 || shift >= (N as i32) * 64 { return result; }

        let word_from_lsb = (shift / 64) as usize; // 0 = limbs[N-1]
        let bit_in_word = (shift % 64) as u32;
        let word_idx = N - 1 - word_from_lsb; // big-endian index

        if bit_in_word + 53 <= 64 {
            result.limbs[word_idx] = mantissa << bit_in_word;
        } else {
            result.limbs[word_idx] = mantissa << bit_in_word;
            if word_idx > 0 {
                result.limbs[word_idx - 1] = mantissa >> (64 - bit_in_word);
            }
        }

        if sign_negative { negate_words(&mut result.limbs); }
        result
    }

    fn to_f64_lossy(self) -> f64 {
        // High word contributes (limbs[0] as i64) × 2^{-60}.
        // Second word contributes (limbs[1] as u64) × 2^{-60-64} = × 2^{-124}.
        // f64 only has 53 bits so the second word adds at most 1 ULP of correction.
        let hi = (self.limbs[0] as i64 as f64) * f64::from_bits(0x3C30_0000_0000_0000); // 2^{-60}
        if N < 2 { return hi; }
        let lo = (self.limbs[1] as f64) * f64::from_bits(0x3830_0000_0000_0000); // 2^{-124}
        hi + lo
    }

    fn norm_sqr(self) -> Self { self * self }
}

impl<const N: usize> Add for BigFloat<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = Self::default();
        let mut carry: u64 = 0;
        for i in (0..N).rev() {
            let (s1, c1) = self.limbs[i].overflowing_add(rhs.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            out.limbs[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        out
    }
}

impl<const N: usize> Sub for BigFloat<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { self + (-rhs) }
}

impl<const N: usize> Mul for BigFloat<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a_neg = self.limbs[0] >> 63 != 0;
        let b_neg = rhs.limbs[0] >> 63 != 0;
        let result_neg = a_neg ^ b_neg;

        let mut abs_a = self;
        let mut abs_b = rhs;
        if a_neg { negate_words(&mut abs_a.limbs); }
        if b_neg { negate_words(&mut abs_b.limbs); }

        let mut out = Self::default();
        mul_unsigned_fixed(&abs_a.limbs, &abs_b.limbs, &mut out.limbs);
        if result_neg { negate_words(&mut out.limbs); }
        out
    }
}

impl<const N: usize> Neg for BigFloat<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut r = self;
        negate_words(&mut r.limbs);
        r
    }
}

// ── BigFloat<N> decimal parsing ───────────────────────────────────────────────

impl<const N: usize> BigFloat<N> {
    /// Parse a decimal string (e.g. `"-1.234567890123456789"`) into `BigFloat<N>`.
    ///
    /// The value must lie in `[−8, 8]`. Only approximately `N × 19` significant
    /// decimal digits of precision are captured — digits beyond that are silently
    /// truncated (they fall below the representable precision of `N` limbs).
    ///
    /// Returns `None` if the string is not a valid decimal or the integer part
    /// falls outside `[−7, 7]` (leaving headroom for intermediate arithmetic).
    pub fn from_decimal_str(s: &str) -> Option<Self> {
        debug_assert!(N <= 8, "only N ≤ 8 supported");
        let s = s.trim();
        let (negative, s) = if let Some(rest) = s.strip_prefix('-') {
            (true, rest)
        } else if let Some(rest) = s.strip_prefix('+') {
            (false, rest)
        } else {
            (false, s)
        };

        let (int_str, frac_str) = match s.find('.') {
            Some(dot) => (&s[..dot], &s[dot + 1..]),
            None => (s, ""),
        };

        let int_val: u64 = int_str.parse().ok()?;
        if int_val > 7 { return None; }

        let mut result = Self::default();
        result.limbs[0] = int_val << 60; // integer part

        if !frac_str.is_empty() {
            // Accumulate fractional digits as a big-endian N-word integer K.
            let max_digits = N * 19; // ≈ N×64 bits / log2(10)
            let mut k = [0u64; 8];
            let mut digit_count: usize = 0;

            for c in frac_str.chars() {
                if digit_count >= max_digits { break; }
                let d = c.to_digit(10)? as u64;
                // k = k × 10 + d
                let mut carry = d as u128;
                for i in (0..N).rev() {
                    let v = (k[i] as u128) * 10 + carry;
                    k[i] = v as u64;
                    carry = v >> 64;
                }
                // Overflow beyond N words is discarded (below our precision).
                digit_count += 1;
            }

            if digit_count > 0 {
                // Place K in a 2N-word buffer and scale to K × 2^{frac_bits}.
                // Placing K in buf[0..N] gives K × 2^{N×64}.
                // Right-shifting by 4 gives K × 2^{N×64−4} = K × 2^{frac_bits}.
                let mut buf = [0u64; 16];
                buf[..N].copy_from_slice(&k[..N]);

                // Big-endian right-shift by 4 bits.
                let mut carry_bits: u64 = 0;
                for word in buf[..2 * N].iter_mut() {
                    let next = *word & 0xF;
                    *word = (carry_bits << 60) | (*word >> 4);
                    carry_bits = next;
                }

                // Divide by 10, digit_count times.
                for _ in 0..digit_count {
                    let mut rem: u128 = 0;
                    for word in buf[..2 * N].iter_mut() {
                        let d = (rem << 64) | (*word as u128);
                        *word = (d / 10) as u64;
                        rem = d % 10;
                    }
                }

                // The fractional fixed-point value is now in buf[N..2N].
                let mut carry: u64 = 0;
                for i in (0..N).rev() {
                    let (s1, c1) = result.limbs[i].overflowing_add(buf[N + i]);
                    let (s2, c2) = s1.overflowing_add(carry);
                    result.limbs[i] = s2;
                    carry = (c1 as u64) + (c2 as u64);
                }
            }
        }

        if negative { negate_words(&mut result.limbs); }
        Some(result)
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

    // ── complex division ──────────────────────────────────────────────────────

    #[test]
    fn complex_div_complex_basic() {
        // (4 + 2i) / (1 + i) = (4+2i)(1-i) / |1+i|² = (6 - 2i) / 2 = 3 - i
        let a = Complex::new(4.0_f64, 2.0);
        let b = Complex::new(1.0_f64, 1.0);
        let q = a / b;
        assert!((q.re - 3.0).abs() < 1e-14, "re = {}", q.re);
        assert!((q.im - (-1.0)).abs() < 1e-14, "im = {}", q.im);
    }

    #[test]
    fn complex_div_self_gives_one() {
        let z = Complex::new(3.0_f64, 4.0);
        let q = z / z;
        assert!((q.re - 1.0).abs() < 1e-14, "re = {}", q.re);
        assert!(q.im.abs() < 1e-14, "im = {}", q.im);
    }

    #[test]
    fn complex_div_f64_scalar() {
        // (3 + 4i) / 5 = 0.6 + 0.8i
        let z = Complex::new(3.0_f64, 4.0);
        let q = z / 5.0_f64;
        assert!((q.re - 0.6).abs() < 1e-14, "re = {}", q.re);
        assert!((q.im - 0.8).abs() < 1e-14, "im = {}", q.im);
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

    // ── F64x2 ────────────────────────────────────────────────────────────────

    #[test]
    fn f64x2_identity_elements() {
        assert_eq!(F64x2::zero().to_f64_lossy(), 0.0);
        assert_eq!(F64x2::one().to_f64_lossy(), 1.0);
    }

    // ── MPFR oracle helpers ───────────────────────────────────────────────────

    /// Compute `(a_hi + a_lo) + (b_hi + b_lo)` at 128-bit precision via MPFR,
    /// return the result as f64.  Used only to build expected values for tests.
    #[cfg(test)]
    fn mpfr_add(a_hi: f64, a_lo: f64, b_hi: f64, b_lo: f64) -> f64 {
        use rug::Float;
        let a = Float::with_val(128, a_hi) + Float::with_val(128, a_lo);
        let b = Float::with_val(128, b_hi) + Float::with_val(128, b_lo);
        (a + b).to_f64()
    }

    #[cfg(test)]
    fn mpfr_mul(a_hi: f64, a_lo: f64, b_hi: f64, b_lo: f64) -> f64 {
        use rug::Float;
        let a = Float::with_val(128, a_hi) + Float::with_val(128, a_lo);
        let b = Float::with_val(128, b_hi) + Float::with_val(128, b_lo);
        (a * b).to_f64()
    }

    // ── F64x2 oracle tests ────────────────────────────────────────────────────

    #[test]
    fn f64x2_add_exact_for_non_overlapping() {
        // 1.0 + 2^-53: these values don't overlap in f64, so the sum and its
        // error can be represented exactly.  F64x2 must capture both halves.
        let a = F64x2(1.0, 0.0);
        let b = F64x2(f64::EPSILON / 2.0, 0.0); // 2^-53
        let r = a + b;
        // hi should be 1.0 + 2^-53 (rounds to 1.0 in plain f64),
        // but lo must be non-zero so the sum is exact.
        assert_eq!(r.0 + r.1, 1.0 + f64::EPSILON / 2.0,
            "double-double sum must be exact for non-overlapping values");
        // The low word must carry the residual — it shouldn't vanish.
        assert_ne!(r.1, 0.0, "lo word must capture the addition error");
    }

    #[test]
    fn complex_f64x2_add_and_mul_via_generic_impl() {
        // Exercises the generic Complex<T: Precision> impl with T = F64x2.
        // This is the type the reference orbit loop will use.
        let a = Complex::new(F64x2(1.0, 0.0), F64x2(2.0, 0.0));
        let b = Complex::new(F64x2(3.0, 0.0), F64x2(4.0, 0.0));

        let sum = a + b;
        assert_eq!(sum.re.to_f64_lossy(), 4.0);
        assert_eq!(sum.im.to_f64_lossy(), 6.0);

        // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let prod = a * b;
        assert_eq!(prod.re.to_f64_lossy(), -5.0);
        assert_eq!(prod.im.to_f64_lossy(), 10.0);
    }

    #[test]
    fn complex_f64x2_norm_sqr_matches_mpfr() {
        use rug::Float;
        // |3 + 4i|² = 25 — checks the full path through Complex::norm_sqr
        let z = Complex::new(F64x2(3.0, 0.0), F64x2(4.0, 0.0));
        let result = z.norm_sqr().to_f64_lossy();
        let expected = {
            let re = Float::with_val(128, 3.0_f64);
            let im = Float::with_val(128, 4.0_f64);
            (re.clone() * &re + im.clone() * &im).to_f64()
        };
        assert_eq!(result, expected);
    }

    #[test]
    fn f64x2_norm_sqr_matches_mpfr() {
        use rug::Float;
        // norm_sqr(x) = x * x — for a scalar F64x2 this is self * self.
        let cases: &[f64] = &[
            core::f64::consts::PI,
            core::f64::consts::SQRT_2,
            1.0 + f64::EPSILON,
            0.1,
        ];
        for &v in cases {
            let x = F64x2(v, 0.0);
            let result = x.norm_sqr().to_f64_lossy();
            let expected = (Float::with_val(128, v) * Float::with_val(128, v)).to_f64();
            assert_eq!(result, expected, "F64x2::norm_sqr mismatch for {v}");
        }
    }

    #[test]
    fn f64x2_norm_sqr_returns_f64x2_not_lossy() {
        // Squaring a value near 1.0 + epsilon should have a non-zero lo component
        // if norm_sqr preserves full double-double precision.
        let x = F64x2(1.0 + f64::EPSILON, 0.0);
        let sq = x.norm_sqr();
        // The hi word is f64(x*x); if lo is non-zero, we have full precision.
        assert_ne!(sq.1, 0.0, "norm_sqr lo word should be non-zero for full precision");
    }

    #[test]
    fn f64x2_neg_flips_both_components() {
        let x = F64x2(3.0, 1.0e-17);
        let n = -x;
        assert_eq!(n.0, -3.0);
        assert_eq!(n.1, -1.0e-17);
    }

    #[test]
    fn f64x2_sub_matches_mpfr_oracle() {
        use rug::Float;
        let cases: &[(f64, f64, f64, f64)] = &[
            (core::f64::consts::PI, 0.0, core::f64::consts::PI, 0.0),
            (1.0, 0.0, f64::EPSILON / 2.0, 0.0),
            (core::f64::consts::SQRT_2, 0.0, 1.0, 0.0),
        ];
        for &(ahi, alo, bhi, blo) in cases {
            let result = (F64x2(ahi, alo) - F64x2(bhi, blo)).to_f64_lossy();
            let a = Float::with_val(128, ahi) + Float::with_val(128, alo);
            let b = Float::with_val(128, bhi) + Float::with_val(128, blo);
            let expected = (a - b).to_f64();
            assert_eq!(result, expected,
                "F64x2 sub mismatch for ({ahi}+{alo}) - ({bhi}+{blo})");
        }
    }

    #[test]
    fn f64x2_mul_matches_mpfr_oracle() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (core::f64::consts::PI, 0.0, core::f64::consts::PI, 0.0),
            (1.0 + f64::EPSILON, 0.0, 1.0 - f64::EPSILON, 0.0),
            (0.1, 0.0, 0.3, 0.0),
            (1.234_567_890_123_456_7, 0.0, 9.876_543_210_987_654, 0.0),
        ];
        for &(ahi, alo, bhi, blo) in cases {
            let result = (F64x2(ahi, alo) * F64x2(bhi, blo)).to_f64_lossy();
            let expected = mpfr_mul(ahi, alo, bhi, blo);
            assert_eq!(result, expected,
                "F64x2 mul mismatch for ({ahi}+{alo}) * ({bhi}+{blo})");
        }
    }

    #[test]
    fn f64x2_add_matches_mpfr_oracle() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (1.0, 0.0, core::f64::consts::PI, 0.0),
            (0.1, 0.0, 0.2, 0.0),
            (1.0e15, 0.0, 1.0, 1.0e-15),
            (-core::f64::consts::SQRT_2, 0.0, core::f64::consts::SQRT_2, 0.0),
        ];
        for &(ahi, alo, bhi, blo) in cases {
            let result = (F64x2(ahi, alo) + F64x2(bhi, blo)).to_f64_lossy();
            let expected = mpfr_add(ahi, alo, bhi, blo);
            assert_eq!(result, expected,
                "F64x2 add mismatch for ({ahi}+{alo}) + ({bhi}+{blo})");
        }
    }

    #[test]
    fn f64x2_from_f64_round_trip() {
        for v in [0.0_f64, 1.0, -1.0, 0.1, core::f64::consts::PI, f64::MAX / 2.0] {
            let x = F64x2::from_f64(v);
            assert_eq!(x.to_f64_lossy(), v);
            assert_eq!(x.1, 0.0, "lo must be 0 for exact f64 construction");
        }
    }

    // ── BigFloat<N> helpers ───────────────────────────────────────────────────

    /// Convert f64 value to MPFR Float at `prec` bits.
    #[cfg(test)]
    fn mpfr(v: f64, prec: u32) -> rug::Float {
        rug::Float::with_val(prec, v)
    }

    /// Evaluate `op` on two f64 values at `prec`-bit precision, return f64.
    #[cfg(test)]
    fn mpfr_op<F>(a: f64, b: f64, prec: u32, op: F) -> f64
    where
        F: Fn(rug::Float, rug::Float) -> rug::Float,
    {
        op(mpfr(a, prec), mpfr(b, prec)).to_f64()
    }

    // ── BigFloat<N>: identity ─────────────────────────────────────────────────

    #[test]
    fn bigfloat_zero_is_zero() {
        assert_eq!(BigFloat::<2>::zero().to_f64_lossy(), 0.0);
        assert_eq!(BigFloat::<4>::zero().to_f64_lossy(), 0.0);
    }

    #[test]
    fn bigfloat_one_is_one() {
        assert_eq!(BigFloat::<2>::one().to_f64_lossy(), 1.0);
        assert_eq!(BigFloat::<4>::one().to_f64_lossy(), 1.0);
    }

    // ── BigFloat<N>: from_f64 round-trip ─────────────────────────────────────

    #[test]
    fn bigfloat_from_f64_round_trip_exact_values() {
        for &v in &[0.0_f64, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 7.0, -7.0] {
            let r2 = BigFloat::<2>::from_f64(v).to_f64_lossy();
            let r4 = BigFloat::<4>::from_f64(v).to_f64_lossy();
            assert_eq!(r2, v, "BigFloat<2> from_f64 round-trip failed for {v}");
            assert_eq!(r4, v, "BigFloat<4> from_f64 round-trip failed for {v}");
        }
    }

    #[test]
    fn bigfloat_from_f64_round_trip_irrational() {
        for &v in &[core::f64::consts::PI, core::f64::consts::SQRT_2, 0.1, 0.3] {
            let r2 = BigFloat::<2>::from_f64(v).to_f64_lossy();
            let r4 = BigFloat::<4>::from_f64(v).to_f64_lossy();
            assert_eq!(r2, v, "BigFloat<2> round-trip failed for {v}");
            assert_eq!(r4, v, "BigFloat<4> round-trip failed for {v}");
        }
    }

    // ── BigFloat<N>: addition ─────────────────────────────────────────────────

    #[test]
    fn bigfloat_add_matches_mpfr_n2() {
        let cases: &[(f64, f64)] = &[
            (1.0, 2.0),
            (core::f64::consts::PI, 1.0),
            (0.1, 0.2),
            (-1.5, 2.5),
            (7.0, -7.0),
        ];
        for &(a, b) in cases {
            let result = (BigFloat::<2>::from_f64(a) + BigFloat::<2>::from_f64(b)).to_f64_lossy();
            let expected = mpfr_op(a, b, 256, |x, y| x + y);
            assert_eq!(result, expected, "BigFloat<2> add: {a} + {b}");
        }
    }

    #[test]
    fn bigfloat_add_matches_mpfr_n4() {
        let cases: &[(f64, f64)] = &[
            (core::f64::consts::PI, core::f64::consts::SQRT_2),
            (-2.0, 1.0),
            (0.1, 0.2),
        ];
        for &(a, b) in cases {
            let result = (BigFloat::<4>::from_f64(a) + BigFloat::<4>::from_f64(b)).to_f64_lossy();
            let expected = mpfr_op(a, b, 256, |x, y| x + y);
            assert_eq!(result, expected, "BigFloat<4> add: {a} + {b}");
        }
    }

    // ── BigFloat<N>: multiplication ───────────────────────────────────────────

    #[test]
    fn bigfloat_mul_matches_mpfr_n2() {
        // All products must stay within [-8, 8].
        let cases: &[(f64, f64)] = &[
            (1.0, 2.0),                                           // 2.0
            (core::f64::consts::PI / 4.0, core::f64::consts::PI / 4.0), // ≈ 0.617
            (0.1, 0.3),                                           // 0.03
            (-1.5, 2.0),                                          // -3.0
            (1.0 + f64::EPSILON, 1.0 - f64::EPSILON),            // ≈ 1.0
        ];
        for &(a, b) in cases {
            let result = (BigFloat::<2>::from_f64(a) * BigFloat::<2>::from_f64(b)).to_f64_lossy();
            let expected = mpfr_op(a, b, 256, |x, y| x * y);
            assert_eq!(result, expected, "BigFloat<2> mul: {a} * {b}");
        }
    }

    #[test]
    fn bigfloat_mul_matches_mpfr_n4() {
        // All products must stay within [-8, 8].
        let cases: &[(f64, f64)] = &[
            (core::f64::consts::SQRT_2, 1.0),   // ≈ 1.41
            (-2.0, 1.5),                          // -3.0
            (0.1, 0.7),                           // 0.07
        ];
        for &(a, b) in cases {
            let result = (BigFloat::<4>::from_f64(a) * BigFloat::<4>::from_f64(b)).to_f64_lossy();
            let expected = mpfr_op(a, b, 256, |x, y| x * y);
            assert_eq!(result, expected, "BigFloat<4> mul: {a} * {b}");
        }
    }

    // ── BigFloat<N>: negation ─────────────────────────────────────────────────

    #[test]
    fn bigfloat_neg_and_double_neg() {
        for &v in &[1.0_f64, -2.5, core::f64::consts::PI] {
            let x = BigFloat::<2>::from_f64(v);
            assert_eq!((-x).to_f64_lossy(), -v, "neg failed for {v}");
            assert_eq!((-(-x)).to_f64_lossy(), v, "double-neg failed for {v}");
        }
    }

    // ── BigFloat<N>: norm_sqr ─────────────────────────────────────────────────

    #[test]
    fn bigfloat_norm_sqr_matches_mpfr() {
        // All squares must stay within [-8, 8]: values must satisfy |v| ≤ √8 ≈ 2.83.
        for &v in &[core::f64::consts::PI / 4.0, 1.5_f64, -2.0_f64, 0.1_f64] {
            let sq2 = BigFloat::<2>::from_f64(v).norm_sqr().to_f64_lossy();
            let sq4 = BigFloat::<4>::from_f64(v).norm_sqr().to_f64_lossy();
            let expected = mpfr_op(v, v, 256, |x, y| x * y);
            assert_eq!(sq2, expected, "BigFloat<2> norm_sqr failed for {v}");
            assert_eq!(sq4, expected, "BigFloat<4> norm_sqr failed for {v}");
        }
    }

    // ── BigFloat<N>: from_decimal_str ─────────────────────────────────────────

    #[test]
    fn bigfloat_decimal_str_zero_and_one() {
        assert_eq!(BigFloat::<2>::from_decimal_str("0.0").unwrap().to_f64_lossy(), 0.0);
        assert_eq!(BigFloat::<2>::from_decimal_str("1.0").unwrap().to_f64_lossy(), 1.0);
        assert_eq!(BigFloat::<2>::from_decimal_str("-1.0").unwrap().to_f64_lossy(), -1.0);
    }

    #[test]
    fn bigfloat_decimal_str_exact_halves() {
        let r = BigFloat::<2>::from_decimal_str("0.5").unwrap().to_f64_lossy();
        assert_eq!(r, 0.5);
        let r = BigFloat::<2>::from_decimal_str("-2.5").unwrap().to_f64_lossy();
        assert_eq!(r, -2.5);
    }

    #[test]
    fn bigfloat_decimal_str_matches_mpfr_high_precision() {
        // Feigenbaum point — a known deep-zoom Mandelbrot coordinate.
        let s = "-1.401155189341197197084495757821858742";
        let result = BigFloat::<4>::from_decimal_str(s).unwrap().to_f64_lossy();
        let expected = {
            use rug::Float;
            let parsed = Float::parse(s).unwrap();
            Float::with_val(256, parsed).to_f64()
        };
        assert_eq!(result, expected,
            "BigFloat<4> from_decimal_str mismatch for Feigenbaum point");
    }

    // ── Complex<BigFloat<N>>: end-to-end ──────────────────────────────────────

    #[test]
    fn complex_bigfloat4_add_and_mul_match_f64_path() {
        // Values chosen so all products stay within [-8, 8]:
        //   (0.5 + 0.3i)(1.0 + 0.5i) = (0.5 - 0.15) + (0.25 + 0.3)i = 0.35 + 0.55i
        let a_f = Complex::new(0.5_f64, 0.3_f64);
        let b_f = Complex::new(1.0_f64, 0.5_f64);
        let a_b = Complex::new(BigFloat::<4>::from_f64(0.5), BigFloat::<4>::from_f64(0.3));
        let b_b = Complex::new(BigFloat::<4>::from_f64(1.0), BigFloat::<4>::from_f64(0.5));

        let sum_f = a_f + b_f;
        let sum_b = a_b + b_b;
        assert_eq!(sum_b.re.to_f64_lossy(), sum_f.re);
        assert_eq!(sum_b.im.to_f64_lossy(), sum_f.im);

        let prod_f = a_f * b_f;
        let prod_b = a_b * b_b;
        assert_eq!(prod_b.re.to_f64_lossy(), prod_f.re);
        assert_eq!(prod_b.im.to_f64_lossy(), prod_f.im);
    }

    // ── BigFloat<N>: proptest arithmetic properties ───────────────────────────

    proptest::proptest! {
        #[test]
        fn bigfloat4_add_matches_mpfr_proptest(
            a in -2.0_f64..2.0_f64,
            b in -2.0_f64..2.0_f64,
        ) {
            let result = (BigFloat::<4>::from_f64(a) + BigFloat::<4>::from_f64(b)).to_f64_lossy();
            let expected = mpfr_op(a, b, 256, |x, y| x + y);
            proptest::prop_assert_eq!(result, expected);
        }

        #[test]
        fn bigfloat4_mul_matches_mpfr_proptest(
            a in -2.0_f64..2.0_f64,
            b in -2.0_f64..2.0_f64,
        ) {
            let result = (BigFloat::<4>::from_f64(a) * BigFloat::<4>::from_f64(b)).to_f64_lossy();
            let expected = mpfr_op(a, b, 256, |x, y| x * y);
            // BigFloat truncates; MPFR rounds to nearest. Allow ≤ 1 ULP difference
            // at the f64 projection level — irrelevant for orbit computation.
            let ulp = expected.abs() * f64::EPSILON;
            proptest::prop_assert!(
                result == expected || (result - expected).abs() <= ulp,
                "result={result}, expected={expected}, diff={}", (result - expected).abs()
            );
        }
    }
}
