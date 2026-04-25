//! wasm-bindgen surface — the only crate that depends on wasm-bindgen.
//!
//! Exposes a small, stable Worker-facing API. Contains no math; all logic
//! lives in the inner crates. Full implementation across Phases 0–5.
use wasm_bindgen::prelude::*;

// ── Memory layout constants ────────────────────────────────────────────────

/// Bytes per primary orbit entry: Complex<f64> pair = 2 × 8 × 2 = 32 B.
/// This is the maximum stride, accommodating Phoenix (two complex values).
const PRIMARY_ORBIT_ENTRY_BYTES: u32 = 32;

/// Bytes per BLA table entry: `repr(C)` BlaEntry = 48 B.
/// Layout: a (16 B) + b (16 B) + skip (4 B) + _pad (4 B) + valid_radius (8 B).
const BLA_ENTRY_BYTES: u32 = 48;

/// Bytes per secondary orbit entry: single Complex<f64> = 16 B.
const SECONDARY_ORBIT_ENTRY_BYTES: u32 = 16;

/// Number of secondary orbit slots.
const SECONDARY_ORBIT_SLOTS: u32 = 3;

/// Bytes per orbit slot header: `{ entry_stride: u32, orbit_len: u32 }`.
const ORBIT_HEADER_BYTES: u32 = 8;

/// Number of tile ring slots per worker (absorbs burst latency).
const TILE_RING_SLOTS_PER_WORKER: u32 = 4;

/// Bytes per tile slot: 256 × 256 pixels × 4 channels × 4 bytes (f32).
const TILE_SLOT_BYTES: u32 = 256 * 256 * 4 * 4;

/// Bytes per slot state entry: one i32 (EMPTY=0 / WRITING=1 / READY=2).
const SLOT_STATE_ENTRY_BYTES: u32 = 4;

// ── Public API ─────────────────────────────────────────────────────────────

/// Byte offsets for every region in the shared `WebAssembly.Memory`.
///
/// Returned by `layout()` once per Worker at startup. TypeScript reads these
/// offsets rather than hardcoding any memory arithmetic.
#[wasm_bindgen]
pub struct MemoryLayout {
    /// Start of the primary reference orbit slot (includes 8-byte header).
    pub primary_orbit_offset: u32,
    /// Start of the BLA acceleration table.
    pub bla_table_offset: u32,
    /// Start of the first secondary orbit slot (3 slots follow contiguously).
    pub secondary_orbit_offset: u32,
    /// Start of the slot state `Int32Array` (EMPTY/WRITING/READY per ring slot).
    pub slot_state_offset: u32,
    /// Start of the tile pixel ring buffer.
    pub tile_ring_offset: u32,
    /// Bytes per tile ring slot (constant: 256 × 256 × 4 × 4 = 1 048 576 B).
    pub tile_slot_bytes: u32,
    /// Total shared memory size in bytes. Round up to `WebAssembly.Memory` pages.
    pub total_bytes: u32,
}

/// Compute byte offsets for all shared memory regions.
///
/// Must be called once per Worker after instantiation. All offsets are
/// measured from byte 0 of the shared `WebAssembly.Memory`.
///
/// # Arguments
/// * `n_workers` — number of Tile Workers (`hardwareConcurrency - 2`, min 2)
/// * `max_iter`  — maximum iteration count for the session
#[wasm_bindgen]
pub fn layout(n_workers: u32, max_iter: u32) -> MemoryLayout {
    // Region 1: primary orbit slot
    let primary_orbit_offset = 0u32;
    let primary_orbit_size = ORBIT_HEADER_BYTES + max_iter * PRIMARY_ORBIT_ENTRY_BYTES;

    // Region 2: BLA table
    let bla_table_offset = primary_orbit_offset + primary_orbit_size;
    let bla_table_size = max_iter * BLA_ENTRY_BYTES;

    // Region 3: secondary orbit slots (3 × header + entries)
    let secondary_orbit_offset = bla_table_offset + bla_table_size;
    let one_secondary_size = ORBIT_HEADER_BYTES + max_iter * SECONDARY_ORBIT_ENTRY_BYTES;
    let secondary_orbit_size = SECONDARY_ORBIT_SLOTS * one_secondary_size;

    // Region 4: slot state array (one i32 per ring slot)
    let total_ring_slots = TILE_RING_SLOTS_PER_WORKER * n_workers;
    let slot_state_offset = secondary_orbit_offset + secondary_orbit_size;
    let slot_state_size = total_ring_slots * SLOT_STATE_ENTRY_BYTES;

    // Region 5: tile pixel ring
    let tile_ring_offset = slot_state_offset + slot_state_size;
    let tile_ring_size = total_ring_slots * TILE_SLOT_BYTES;

    let total_bytes = tile_ring_offset + tile_ring_size;

    MemoryLayout {
        primary_orbit_offset,
        bla_table_offset,
        secondary_orbit_offset,
        slot_state_offset,
        tile_ring_offset,
        tile_slot_bytes: TILE_SLOT_BYTES,
        total_bytes,
    }
}

/// Convert a `total_bytes` value to the number of 64 KiB `WebAssembly.Memory` pages.
#[wasm_bindgen]
pub fn bytes_to_pages(total_bytes: u32) -> u32 {
    total_bytes.div_ceil(65536)
}
