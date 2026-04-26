#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use wasm_bridge::{TileJob, render_tile_impl};
use kernel::TilePixel;

// ── render_tile_impl: mixed interior and escaped pixels ───────────────────
//
// Tile covers [-2, 2) × [-2, 2) at pixel_step = 4/256 ≈ 0.015625, which
// straddles the Mandelbrot set boundary. Expect both escaped and interior.

#[wasm_bindgen_test]
fn render_tile_produces_mixed_pixels() {
    let pixel_step = 4.0 / 256.0;
    let job = TileJob {
        delta_re: -2.0,
        delta_im: -2.0,
        pixel_step,
        max_iter: 100,
        slot_index: 0,
    };

    let mut out = vec![TilePixel::default(); 256 * 256];
    render_tile_impl(&job, &mut out);

    let escaped = out.iter().filter(|p| p.escaped == 1.0).count();
    let interior = out.iter().filter(|p| p.escaped == 0.0).count();

    assert!(escaped > 0, "expected at least one escaped pixel, got 0");
    assert!(interior > 0, "expected at least one interior pixel, got 0");
}

// ── render_tile_impl: all-exterior tile is fully escaped ─────────────────

#[wasm_bindgen_test]
fn render_tile_far_exterior_all_escaped() {
    let job = TileJob {
        delta_re: 10.0, // far outside the set
        delta_im: 10.0,
        pixel_step: 0.01,
        max_iter: 100,
        slot_index: 0,
    };

    let mut out = vec![TilePixel::default(); 256 * 256];
    render_tile_impl(&job, &mut out);

    assert!(
        out.iter().all(|p| p.escaped == 1.0),
        "all pixels far outside the Mandelbrot set must be escaped"
    );
}

// ── render_tile_impl: glitch_count is 0 in Phase 1 ───────────────────────

#[wasm_bindgen_test]
fn render_tile_glitch_count_is_zero() {
    let job = TileJob {
        delta_re: -2.0,
        delta_im: -2.0,
        pixel_step: 4.0 / 256.0,
        max_iter: 100,
        slot_index: 0,
    };

    let mut out = vec![TilePixel::default(); 256 * 256];
    let result = render_tile_impl(&job, &mut out);

    assert_eq!(result.glitch_count, 0);
}
