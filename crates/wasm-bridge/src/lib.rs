//! wasm-bindgen surface — the only crate that depends on wasm-bindgen.
//!
//! Exposes a small, stable Worker-facing API. Contains no math; all logic
//! lives in the inner crates. Full implementation across Phases 0–5.
use wasm_bindgen::prelude::*;

/// Byte offsets for every region in the shared `WebAssembly.Memory`.
///
/// Returned by `layout()` once per Worker at startup. TypeScript reads these
/// offsets rather than hardcoding any memory arithmetic.
#[wasm_bindgen]
pub struct MemoryLayout {
    pub primary_orbit_offset: u32,
    pub bla_table_offset: u32,
    pub secondary_orbit_offset: u32,
    pub slot_state_offset: u32,
    pub tile_ring_offset: u32,
    pub total_bytes: u32,
}

/// Compute byte offsets for all shared memory regions.
///
/// Must be called once per Worker after WASM instantiation. All offsets are
/// measured from byte 0 of the shared `WebAssembly.Memory`.
///
/// Stub — real offset arithmetic implemented in issue #4.
#[wasm_bindgen]
pub fn layout(_n_workers: u32, _max_iter: u32) -> MemoryLayout {
    MemoryLayout {
        primary_orbit_offset: 0,
        bla_table_offset: 0,
        secondary_orbit_offset: 0,
        slot_state_offset: 0,
        tile_ring_offset: 0,
        total_bytes: 0,
    }
}
