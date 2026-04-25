//! Coloring algorithms: `EscapeResult` → RGBA f32 via a LUT.
//!
//! The LUT is always 4 096 entries × RGBA f32. It is uploaded to a WebGL
//! `lut1D` texture and sampled by the fragment shader. Full implementation
//! in Phase 4.

/// 4 096-entry RGBA f32 LUT uploaded to the GPU as a 1D texture.
pub struct Lut {
    entries: Box<[f32; 4096 * 4]>,
}

impl Lut {
    pub fn zeroed() -> Self {
        Self {
            entries: Box::new([0.0f32; 4096 * 4]),
        }
    }

    /// Returns the raw f32 slice for uploading via `gl.texImage2D`.
    pub fn as_slice(&self) -> &[f32] {
        self.entries.as_ref()
    }
}
