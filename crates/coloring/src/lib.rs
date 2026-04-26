//! Coloring algorithms — LUT generation and EscapeResult → RGBA mapping.
//!
//! Phase 1 scope: produce a default 4096-entry RGBA f32 LUT for GPU upload.
//! The GPU fragment shader does the actual coloring (sampling the LUT by
//! `smooth_t`). No per-pixel Rust coloring pass runs in the hot loop.
//!
//! Full palette editor and EscapeResult-based algorithms ship in Phase 4.

/// Number of entries in the LUT. Always 4096.
pub const LUT_SIZE: usize = 4096;

/// Gradient control points in RGBA f32, evenly spaced in [0, 1).
///
/// Five stops create four linear segments. The last stop equals the first,
/// making the gradient cyclic — `smooth_t` can wrap around continuously
/// without a color discontinuity.
///
/// Palette: deep-blue → cyan → white → gold → deep-blue.
const STOPS: &[[f32; 4]] = &[
    [0.0, 0.0, 0.5, 1.0], // t = 0.00  deep blue
    [0.0, 0.8, 1.0, 1.0], // t = 0.25  cyan
    [1.0, 1.0, 1.0, 1.0], // t = 0.50  white
    [1.0, 0.6, 0.0, 1.0], // t = 0.75  gold
    [0.0, 0.0, 0.5, 1.0], // t = 1.00  deep blue (cyclic wrap)
];

/// 4096-entry RGBA f32 LUT uploaded to the GPU as a 1D texture.
///
/// The fragment shader samples this by `smooth_t mod 1.0` to produce smooth,
/// band-free coloring. Interior pixels (escaped = 0.0) are rendered black by
/// the shader directly; the LUT is only applied to escaped pixels.
pub struct Lut {
    data: Vec<f32>, // length always LUT_SIZE * 4
}

impl Lut {
    /// Build the default cyclic gradient LUT.
    ///
    /// Uses `t = i / LUT_SIZE` (not `i / (LUT_SIZE - 1)`) so that indices 0
    /// and 2048 land exactly on gradient stops — making boundary tests exact.
    pub fn default_gradient() -> Self {
        let mut data = vec![0.0f32; LUT_SIZE * 4];
        let n_seg = (STOPS.len() - 1) as f32; // 4 segments

        for i in 0..LUT_SIZE {
            let t = i as f32 / LUT_SIZE as f32;
            let pos = t * n_seg;
            let seg = (pos as usize).min(STOPS.len() - 2);
            let local_t = pos - seg as f32;

            let a = &STOPS[seg];
            let b = &STOPS[seg + 1];

            for ch in 0..4 {
                data[i * 4 + ch] = a[ch] + (b[ch] - a[ch]) * local_t;
            }
        }

        Self { data }
    }

    /// Raw f32 slice for `gl.texImage2D` or wasm-bindgen `Float32Array`.
    /// Length is always `LUT_SIZE * 4 = 16384`.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// RGBA channels at the given LUT index. Panics if `index >= LUT_SIZE`.
    pub fn entry(&self, index: usize) -> [f32; 4] {
        assert!(index < LUT_SIZE, "LUT index {index} out of range");
        let base = index * 4;
        [
            self.data[base],
            self.data[base + 1],
            self.data[base + 2],
            self.data[base + 3],
        ]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn entry_approx_eq(got: [f32; 4], want: [f32; 4], tol: f32) -> bool {
        got.iter().zip(want.iter()).all(|(a, b)| approx_eq(*a, *b, tol))
    }

    // ── Size ─────────────────────────────────────────────────────────────────

    #[test]
    fn lut_has_correct_length() {
        let lut = Lut::default_gradient();
        assert_eq!(lut.as_slice().len(), LUT_SIZE * 4);
    }

    // ── Boundary values (exact) ───────────────────────────────────────────────

    #[test]
    fn index_0_is_deep_blue() {
        // t = 0/4096 = 0.0 → exactly STOPS[0]
        let lut = Lut::default_gradient();
        let e = lut.entry(0);
        assert_eq!(e, [0.0, 0.0, 0.5, 1.0], "index 0 should be deep blue");
    }

    #[test]
    fn index_2048_is_white() {
        // t = 2048/4096 = 0.5 → exactly STOPS[2]
        let lut = Lut::default_gradient();
        let e = lut.entry(2048);
        assert!(
            entry_approx_eq(e, [1.0, 1.0, 1.0, 1.0], 1e-6),
            "index 2048 should be white, got {e:?}"
        );
    }

    #[test]
    fn index_4095_approaches_deep_blue() {
        // t = 4095/4096 ≈ 0.99976 — one step before the cyclic wrap.
        // The blue channel should be dominant; red/green near zero.
        let lut = Lut::default_gradient();
        let e = lut.entry(4095);
        assert!(e[2] > 0.4, "blue channel at 4095 should be > 0.4, got {}", e[2]);
        assert!(e[0] < 0.01, "red channel at 4095 should be < 0.01, got {}", e[0]);
        assert_eq!(e[3], 1.0, "alpha at 4095 should be 1.0");
    }

    // ── Interior stop values (exact) ─────────────────────────────────────────

    #[test]
    fn index_1024_is_cyan() {
        // t = 1024/4096 = 0.25 → exactly STOPS[1]
        let lut = Lut::default_gradient();
        let e = lut.entry(1024);
        assert!(
            entry_approx_eq(e, [0.0, 0.8, 1.0, 1.0], 1e-6),
            "index 1024 should be cyan, got {e:?}"
        );
    }

    #[test]
    fn index_3072_is_gold() {
        // t = 3072/4096 = 0.75 → exactly STOPS[3]
        let lut = Lut::default_gradient();
        let e = lut.entry(3072);
        assert!(
            entry_approx_eq(e, [1.0, 0.6, 0.0, 1.0], 1e-6),
            "index 3072 should be gold, got {e:?}"
        );
    }

    // ── Alpha channel ─────────────────────────────────────────────────────────

    #[test]
    fn all_alpha_values_are_one() {
        let lut = Lut::default_gradient();
        let slice = lut.as_slice();
        for i in 0..LUT_SIZE {
            let alpha = slice[i * 4 + 3];
            assert_eq!(alpha, 1.0, "alpha at index {i} should be 1.0, got {alpha}");
        }
    }

    // ── Gradient is monotonically smooth in each segment ──────────────────────

    #[test]
    fn gradient_is_continuous_at_segment_boundaries() {
        let lut = Lut::default_gradient();
        // Each of the four segment boundaries (indices 0, 1024, 2048, 3072)
        // should match the expected stop value exactly.
        let stops = [(0, STOPS[0]), (1024, STOPS[1]), (2048, STOPS[2]), (3072, STOPS[3])];
        for (idx, expected) in stops {
            let e = lut.entry(idx);
            assert!(
                entry_approx_eq(e, expected, 1e-6),
                "at index {idx}: expected {expected:?}, got {e:?}"
            );
        }
    }
}
