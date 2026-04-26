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

/// Allocate a tile-sized buffer (256×256×4 f32) on the WASM heap.
///
/// Returns the byte offset of the allocation in WASM linear memory.
/// The caller must free it with `free_tile_buf(ptr)` after copying the data.
/// Used by Tile Workers that need a safe scratch region before shared memory
/// is available (Phase 0). Phase 1+ will write directly into the shared SAB.
#[wasm_bindgen]
pub fn alloc_tile_buf() -> u32 {
    let buf: Box<[u8]> = vec![0u8; 256 * 256 * 4 * 4].into_boxed_slice();
    Box::into_raw(buf) as *mut u8 as u32
}

/// Free a buffer previously returned by `alloc_tile_buf`.
#[wasm_bindgen]
pub fn free_tile_buf(ptr: u32) {
    unsafe {
        let slice_ptr = core::ptr::slice_from_raw_parts_mut(ptr as *mut u8, 256 * 256 * 4 * 4);
        drop(Box::from_raw(slice_ptr));
    }
}

/// Write a solid-color 256×256 RGBA f32 tile into the shared memory ring slot
/// at `slot_byte_offset`.
///
/// Called by Tile Workers for the Phase 0 hello-world. The offset must be
/// 4-byte-aligned; all offsets produced by `layout()` satisfy this.
///
/// # Safety
/// `slot_byte_offset` must be a valid offset into WASM linear memory with at
/// least `256 * 256 * 4 * 4` bytes available. The caller (the Tile Worker JS
/// shell) is responsible for passing the correct value from `layout()`.
#[wasm_bindgen]
pub fn write_solid_tile(slot_byte_offset: u32, r: f32, g: f32, b: f32, a: f32) {
    const PIXELS: usize = 256 * 256;
    // Safety: slot_byte_offset is a valid 4-byte-aligned byte offset into WASM
    // linear memory, which is the shared WebAssembly.Memory. All offsets from
    // layout() are multiples of 4. The Tile Worker acquires the ring slot via
    // Atomics.compareExchange before calling this function.
    unsafe {
        let ptr = slot_byte_offset as *mut f32;
        for i in 0..PIXELS {
            *ptr.add(i * 4) = r;
            *ptr.add(i * 4 + 1) = g;
            *ptr.add(i * 4 + 2) = b;
            *ptr.add(i * 4 + 3) = a;
        }
    }
}

/// Build the 4096-entry RGBA f32 LUT and return it as a `Float32Array`.
///
/// Uploaded once at session startup to the WebGL `lut1D` texture.
/// The fragment shader samples it by `smooth_t` to apply smooth coloring.
///
/// Phase 1: `palette` is ignored — always returns the default gradient.
/// Phase 4 will parse palette JSON and produce a custom gradient.
#[wasm_bindgen]
pub fn build_lut(_palette: JsValue) -> Box<[f32]> {
    coloring::Lut::default_gradient()
        .as_slice()
        .to_vec()
        .into_boxed_slice()
}
