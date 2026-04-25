#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use wasm_bridge::{bytes_to_pages, layout};

// No run_in_browser restriction: tests run in Node.js locally (--node) and
// in Chrome on CI (--chrome --headless). Add run_in_browser once tests
// require SharedArrayBuffer or other browser-only APIs.

// ── Offset ordering ────────────────────────────────────────────────────────

/// Offsets must be strictly increasing in the region order defined in
/// ARCHITECTURE.md §5.2: primary_orbit < bla_table < secondary_orbit <
/// slot_state < tile_ring, and total_bytes must exceed tile_ring_offset.
#[wasm_bindgen_test]
fn layout_offsets_are_strictly_increasing() {
    let m = layout(4, 1000);

    assert_eq!(
        m.primary_orbit_offset, 0,
        "primary orbit must start at byte 0"
    );
    assert!(
        m.bla_table_offset > m.primary_orbit_offset,
        "BLA table must follow primary orbit"
    );
    assert!(
        m.secondary_orbit_offset > m.bla_table_offset,
        "secondary orbits must follow BLA table"
    );
    assert!(
        m.slot_state_offset > m.secondary_orbit_offset,
        "slot state must follow secondary orbits"
    );
    assert!(
        m.tile_ring_offset > m.slot_state_offset,
        "tile ring must follow slot state"
    );
    assert!(
        m.total_bytes > m.tile_ring_offset,
        "total_bytes must exceed tile_ring_offset"
    );
}

// ── Exact offset values (n_workers=4, max_iter=1000) ──────────────────────
//
// Primary orbit  : offset=0, size = 8 + 1000*32 = 32 008 B
// BLA table      : offset=32 008, size = 1000*48 = 48 000 B
// Secondary ×3   : offset=80 008, size = 3*(8 + 1000*16) = 48 024 B
// Slot state     : offset=128 032, size = 4*4*4 = 64 B
// Tile ring      : offset=128 096, size = 4*4*256*256*16 = 16 777 216 B
// Total          : 16 905 312 B

#[wasm_bindgen_test]
fn layout_exact_offsets_4_workers_1000_iter() {
    let m = layout(4, 1000);

    assert_eq!(m.primary_orbit_offset, 0);
    assert_eq!(m.bla_table_offset, 32_008);
    assert_eq!(m.secondary_orbit_offset, 80_008);
    assert_eq!(m.slot_state_offset, 128_032);
    assert_eq!(m.tile_ring_offset, 128_096);
    assert_eq!(m.tile_slot_bytes, 256 * 256 * 4 * 4);
    assert_eq!(m.total_bytes, 16_905_312);
}

// ── Tile ring scales linearly with n_workers ───────────────────────────────
//
// The tile ring is `4 × n_workers × TILE_SLOT_BYTES`. Doubling n_workers
// must double the tile ring region size (total_bytes - tile_ring_offset).

#[wasm_bindgen_test]
fn tile_ring_size_scales_with_n_workers() {
    let m4 = layout(4, 500);
    let m8 = layout(8, 500);

    let ring_size_4 = m4.total_bytes - m4.tile_ring_offset;
    let ring_size_8 = m8.total_bytes - m8.tile_ring_offset;

    assert_eq!(
        ring_size_8,
        ring_size_4 * 2,
        "tile ring size must double when n_workers doubles (got {ring_size_4} and {ring_size_8})"
    );
}

// ── tile_slot_bytes is constant regardless of inputs ──────────────────────

#[wasm_bindgen_test]
fn tile_slot_bytes_is_constant() {
    assert_eq!(layout(2, 100).tile_slot_bytes, 256 * 256 * 4 * 4);
    assert_eq!(layout(16, 100_000).tile_slot_bytes, 256 * 256 * 4 * 4);
}

// ── bytes_to_pages rounds up correctly ────────────────────────────────────

#[wasm_bindgen_test]
fn bytes_to_pages_rounds_up() {
    assert_eq!(bytes_to_pages(0), 0);
    assert_eq!(bytes_to_pages(1), 1);
    assert_eq!(bytes_to_pages(65_536), 1);
    assert_eq!(bytes_to_pages(65_537), 2);

    // layout() total_bytes should convert to a sane page count.
    let total = layout(8, 10_000).total_bytes;
    let pages = bytes_to_pages(total);
    assert!(pages > 0);
    assert!(pages as u64 * 65_536 >= total as u64);
}
