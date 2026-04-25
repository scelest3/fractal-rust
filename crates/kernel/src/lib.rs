//! Fractal iteration engine — escape-time computation, perturbation theory, BLA.
//!
//! All public functions accept byte slices into shared `WebAssembly.Memory`
//! so they are testable natively with stack-allocated buffers.
//! Full implementation in Phases 1–2.

/// Raw per-pixel output written into the tile ring slot.
/// Four f32 channels: smooth_t, orbit_min_r, angle, escaped (1.0 or 0.0).
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct TilePixel {
    pub smooth_t: f32,
    pub orbit_min_r: f32,
    pub angle: f32,
    pub escaped: f32,
}

/// Escape-time result produced per pixel by the perturbation loop.
#[derive(Copy, Clone, Debug, Default)]
pub struct EscapeResult {
    pub iter: u32,
    pub escaped: bool,
    pub smooth_t: f64,
    pub orbit_min_r: f64,
    pub angle_final: f64,
}

/// Which fractal map to iterate. Stub — extended in Phase 1/3.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FractalKind {
    Mandelbrot,
    Newton,
}
