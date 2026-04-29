//! wasm-bindgen surface — the only crate that depends on wasm-bindgen.
//!
//! Exposes a small, stable Worker-facing API. Contains no math; all logic
//! lives in the inner crates. Full implementation across Phases 0–5.
use std::cell::RefCell;
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

// ── render_tile ────────────────────────────────────────────────────────────

const TILE_SIZE: usize = 256;
const TILE_PIXELS: usize = TILE_SIZE * TILE_SIZE;

/// Parameters for one tile render job, passed from the Tile Worker.
///
/// `delta_re` / `delta_im` are the fractal coordinates of the tile's
/// top-left pixel center. `pixel_step` is fractal units per pixel (same
/// for both axes). `slot_index` selects the ring buffer slot to write into.
#[wasm_bindgen]
pub struct TileJob {
    pub delta_re: f64,
    pub delta_im: f64,
    pub pixel_step: f64,
    pub max_iter: u32,
    pub slot_index: u32,
}

/// Result returned to the Tile Worker after rendering.
#[wasm_bindgen]
pub struct TileResult {
    /// Number of glitched pixels detected (always 0 in Phase 1).
    pub glitch_count: u32,
}

/// Render a 256×256 tile into the WASM heap buffer at `ptr`.
///
/// `ptr` must be a byte offset returned by `alloc_tile_buf()`. After this
/// call, the Tile Worker copies the data from WASM linear memory into the
/// shared SAB slot using `Uint8Array.prototype.set`.
///
/// This is the Phase 1 entry point: it avoids the Phase 2 shared-memory path
/// (which requires `-Z build-std` nightly to inject a `SharedArrayBuffer` as
/// WASM linear memory) while still producing real escape-time tile data.
#[wasm_bindgen]
pub fn render_tile_to_ptr(
    delta_re: f64,
    delta_im: f64,
    pixel_step: f64,
    max_iter: u32,
    ptr: u32,
) {
    let job = TileJob { delta_re, delta_im, pixel_step, max_iter, slot_index: 0 };
    // Safety: ptr is from alloc_tile_buf() — a valid heap allocation of
    // exactly TILE_PIXELS * size_of::<TilePixel>() bytes.
    let out =
        unsafe { core::slice::from_raw_parts_mut(ptr as *mut kernel::TilePixel, TILE_PIXELS) };
    render_tile_impl(&job, out);
}

/// Inner implementation: render one 256×256 tile into a caller-supplied
/// `TilePixel` slice. Separated from the wasm export so integration tests
/// can pass a heap-allocated buffer without requiring shared memory layout.
pub fn render_tile_impl(job: &TileJob, out: &mut [kernel::TilePixel]) -> TileResult {
    debug_assert_eq!(out.len(), TILE_PIXELS);

    let mut coords = Vec::with_capacity(TILE_PIXELS);
    for row in 0..TILE_SIZE {
        for col in 0..TILE_SIZE {
            coords.push(arith::Complex::new(
                job.delta_re + col as f64 * job.pixel_step,
                job.delta_im + row as f64 * job.pixel_step,
            ));
        }
    }

    kernel::render_tile_escape(&kernel::MandelbrotMap, &coords, job.max_iter, out);
    TileResult { glitch_count: 0 }
}

/// Render a 256×256 Mandelbrot tile into the shared memory ring slot.
///
/// Computes per-pixel `Complex<f64>` coordinates from `job`, calls
/// `kernel::render_tile_escape`, and writes `TilePixel` data directly
/// into WASM linear memory at `tile_ring_offset + slot_index * tile_slot_bytes`.
///
/// # Safety
/// The caller (Tile Worker) must have acquired the ring slot via
/// `Atomics.compareExchange` before calling this function, ensuring no
/// concurrent writes. The slot byte range must be within the WASM linear
/// memory that backs the shared `WebAssembly.Memory`.
#[wasm_bindgen]
pub fn render_tile(job: &TileJob, layout: &MemoryLayout) -> TileResult {
    let byte_offset =
        (layout.tile_ring_offset + job.slot_index * layout.tile_slot_bytes) as usize;
    let ptr = byte_offset as *mut kernel::TilePixel;
    // Safety: byte_offset is within the shared WebAssembly.Memory — the Tile
    // Worker verified the slot is EMPTY before calling and holds WRITING state.
    let out = unsafe { core::slice::from_raw_parts_mut(ptr, TILE_PIXELS) };
    render_tile_impl(job, out)
}

// ── Phase 2: reference orbit + perturbation rendering ─────────────────────

// Thread-local storage shared between the orbit-compute path (Orbit Worker)
// and the perturbation-render path (Tile Workers). Each WASM instance — one
// per Web Worker — has its own copy, so no cross-worker contention.
thread_local! {
    static ORBIT: RefCell<Vec<arith::Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static BLA:   RefCell<Vec<kernel::BlaEntry>>    = const { RefCell::new(Vec::new()) };
}

/// Compute the reference orbit and BLA table for a deep-zoom render.
///
/// Selects `BigFloat<N>` precision via `required_limbs(zoom_exp)`, parses
/// `cx`/`cy` as BigDecimal strings, runs `kernel::compute_ref_orbit`, then
/// `kernel::build_bla_table`. Results are stored in thread-local buffers;
/// read them with `orbit_data_ptr` / `orbit_data_len` / `bla_data_ptr`.
///
/// Called by the Orbit Worker after each zoom-settle event.
#[wasm_bindgen]
pub fn compute_reference_orbit(cx: &str, cy: &str, zoom_exp: f64, max_iter: u32) {
    let (orbit, bla) = match arith::required_limbs(zoom_exp) {
        n if n <= 2 => orbit_inner::<2>(cx, cy, max_iter),
        n if n <= 4 => orbit_inner::<4>(cx, cy, max_iter),
        n if n <= 8 => orbit_inner::<8>(cx, cy, max_iter),
        n => panic!("required_limbs={n} exceeds Phase 2 max (N=8); zoom_exp={zoom_exp}"),
    };
    ORBIT.with(|o| *o.borrow_mut() = orbit);
    BLA.with(|b| *b.borrow_mut() = bla);
}

fn orbit_inner<const N: usize>(
    cx: &str,
    cy: &str,
    max_iter: u32,
) -> (Vec<arith::Complex<f64>>, Vec<kernel::BlaEntry>) {
    let c = arith::Complex::new(
        arith::BigFloat::<N>::from_decimal_str(cx).unwrap_or_default(),
        arith::BigFloat::<N>::from_decimal_str(cy).unwrap_or_default(),
    );
    // Allocate max_iter + 1 so compute_ref_orbit can append the escaped Z_N entry.
    let mut buf = vec![arith::Complex::<f64>::zero(); max_iter as usize + 1];
    let orbit_len = kernel::compute_ref_orbit(&kernel::MandelbrotMap, c, max_iter, &mut buf);

    // BLA is built from the non-escaped entries only (orbit_len).
    let bla = kernel::build_bla_table(&buf[..orbit_len]);

    // The orbit sent to perturb_pixel includes the escaped Z_N entry (orbit_len + 1)
    // when the reference is exterior, so perturb_pixel can detect escape at the boundary.
    // For interior references orbit_len == max_iter, and buf[orbit_len] is a zero entry
    // that is never reached (limit = min(max_iter+1, orbit_len+1) = max_iter+1, but the
    // interior loop will exhaust max_iter steps and return Interior before reaching it).
    let effective_len = if orbit_len < max_iter as usize { orbit_len + 1 } else { orbit_len };
    let orbit = buf[..effective_len].to_vec();
    (orbit, bla)
}

/// Byte offset of the orbit buffer in WASM linear memory.
/// Valid immediately after `compute_reference_orbit` returns.
#[wasm_bindgen]
pub fn orbit_data_ptr() -> u32 {
    ORBIT.with(|o| o.borrow().as_ptr() as u32)
}

/// Number of `Complex<f64>` entries in the orbit (= valid iteration steps).
#[wasm_bindgen]
pub fn orbit_data_len() -> u32 {
    ORBIT.with(|o| o.borrow().len() as u32)
}

/// Byte offset of the BLA table in WASM linear memory.
#[wasm_bindgen]
pub fn bla_data_ptr() -> u32 {
    BLA.with(|b| b.borrow().as_ptr() as u32)
}

/// Number of `BlaEntry` entries in the BLA table.
///
/// May be less than `orbit_data_len()` by 1 for exterior reference points
/// (the extra orbit entry is the escaped Z_N; it has no corresponding BLA entry).
#[wasm_bindgen]
pub fn bla_data_len() -> u32 {
    BLA.with(|b| b.borrow().len() as u32)
}

/// Install a reference orbit and BLA table into WASM thread-local storage.
///
/// Called by Tile Workers when they receive an `orbit_ready` message.
/// `orbit_f64` is a flat `[re₀, im₀, re₁, im₁, …]` array (2 f64 per entry).
/// `bla_bytes` is the raw byte representation of `BlaEntry` structs (48 B each).
///
/// This copy (from orbitSab → WASM heap) happens once per orbit change,
/// not once per tile render.
#[wasm_bindgen]
pub fn install_orbit(orbit_f64: &[f64], bla_bytes: &[u8]) {
    let orbit: Vec<arith::Complex<f64>> = orbit_f64
        .chunks_exact(2)
        .map(|c| arith::Complex::new(c[0], c[1]))
        .collect();

    let entry_size = core::mem::size_of::<kernel::BlaEntry>(); // 48
    let bla_len = bla_bytes.len() / entry_size;
    let mut bla = vec![kernel::BlaEntry::default(); bla_len];
    if bla_len > 0 {
        // SAFETY: BlaEntry is repr(C); bytes originate from the same Rust build
        // running in the Orbit Worker's WASM instance. We write into a properly
        // aligned Vec allocation via copy_nonoverlapping (source alignment unconstrained).
        unsafe {
            core::ptr::copy_nonoverlapping(
                bla_bytes.as_ptr(),
                bla.as_mut_ptr() as *mut u8,
                bla_len * entry_size,
            );
        }
    }

    ORBIT.with(|o| *o.borrow_mut() = orbit);
    BLA.with(|b| *b.borrow_mut() = bla);
}

/// Render a 256×256 Mandelbrot tile using F64x2 (double-double) arithmetic.
///
/// Covers zoom_exp ≈ 14–20 where f64 absolute coordinates lose per-pixel
/// precision but F64x2 (~30 significant digits) is exact.
///
/// `cx_ref` / `cy_ref` are the f64 viewport center. `delta_re` / `delta_im`
/// are the tile top-left's integer-offset from center (`(tile_px - W/2) × step`).
/// Splitting center + offset and combining in F64x2 avoids catastrophic
/// cancellation without requiring a high-precision string parse per tile.
#[wasm_bindgen]
pub fn render_tile_f64x2(
    cx_ref: f64,
    cy_ref: f64,
    delta_re: f64,
    delta_im: f64,
    pixel_step: f64,
    max_iter: u32,
    out_ptr: u32,
) {
    use arith::{F64x2, Precision};
    // Safety: out_ptr is from alloc_tile_buf() — a valid TILE_PIXELS-sized heap buffer.
    let out = unsafe {
        core::slice::from_raw_parts_mut(out_ptr as *mut kernel::TilePixel, TILE_PIXELS)
    };
    let escape_r_sq = 4.0_f64;
    let cx0 = F64x2::from_f64(cx_ref);
    let cy0 = F64x2::from_f64(cy_ref);
    for row in 0..TILE_SIZE {
        for col in 0..TILE_SIZE {
            let cr = cx0 + F64x2::from_f64(delta_re + col as f64 * pixel_step);
            let ci = cy0 + F64x2::from_f64(delta_im + row as f64 * pixel_step);
            out[row * TILE_SIZE + col] = mandelbrot_f64x2(cr, ci, max_iter, escape_r_sq);
        }
    }
}

fn mandelbrot_f64x2(cr: arith::F64x2, ci: arith::F64x2, max_iter: u32, escape_r_sq: f64) -> kernel::TilePixel {
    use arith::{F64x2, Precision};
    let mut zr = F64x2::zero();
    let mut zi = F64x2::zero();
    let mut orbit_min_r = f64::MAX;
    let mut orbit_min_z = arith::Complex::<f64>::zero();

    for iter in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let norm_sq = (zr2 + zi2).to_f64_lossy();

        let r = norm_sq.sqrt();
        if iter > 0 && r < orbit_min_r {
            orbit_min_r = r;
            orbit_min_z = arith::Complex::new(zr.to_f64_lossy(), zi.to_f64_lossy());
        }

        if norm_sq > escape_r_sq {
            let smooth_t = (iter as f64 + 1.0) - norm_sq.sqrt().ln().ln() / core::f64::consts::LN_2;
            let zr_f64 = zr.to_f64_lossy();
            let zi_f64 = zi.to_f64_lossy();
            return kernel::TilePixel::from(kernel::EscapeResult::Escaped {
                iter,
                smooth_t,
                orbit_min_r,
                orbit_min_z,
                angle_final: zi_f64.atan2(zr_f64),
            });
        }

        // z ← z² + c
        let zr_new = zr2 - zi2 + cr;
        zi = F64x2::from_f64(2.0) * zr * zi + ci;
        zr = zr_new;
    }

    kernel::TilePixel::from(kernel::EscapeResult::Interior { orbit_min_r, orbit_min_z })
}

/// Render a 256×256 perturbation tile using the installed reference orbit.
///
/// `cx_ref` / `cy_ref` are the f64 approximation of the reference point
/// (viewport center). They are used only for the final escape_time fallback.
///
/// `delta_re` / `delta_im` are the fractal offsets of the tile's top-left
/// pixel from the reference point, computed as `(tile_pixel − W/2) × pixel_step`.
///
/// Glitch correction is two-pass:
///   Pass 1 — perturbation against the primary reference orbit.
///   Pass 2 — pixels that glitched in pass 1 are re-rendered against a secondary
///             orbit computed from the first glitch's δc (f64 perturbation, fast).
///   Fallback — pixels that still glitch after pass 2 fall back to escape_time
///              with absolute f64 coordinates (correct up to zoom_exp ≈ 14).
///
/// Returns the number of pixels that required the escape_time fallback.
#[wasm_bindgen]
pub fn render_tile_perturb(
    cx_ref: f64,
    cy_ref: f64,
    delta_re: f64,
    delta_im: f64,
    pixel_step: f64,
    max_iter: u32,
    out_ptr: u32,
) -> u32 {
    ORBIT.with(|orbit_cell| {
        BLA.with(|bla_cell| {
            let orbit = orbit_cell.borrow();
            let bla = bla_cell.borrow();
            if orbit.is_empty() {
                return 0;
            }
            // Safety: out_ptr is from alloc_tile_buf() — a valid TILE_PIXELS-sized heap buffer.
            let out = unsafe {
                core::slice::from_raw_parts_mut(out_ptr as *mut kernel::TilePixel, TILE_PIXELS)
            };

            // Precompute secondary orbit at the tile center δc.
            // The center pixel is a representative point for the tile; rebasing to it
            // corrects exterior pixels that glitch against the primary mid-loop.
            let center_dc_re = delta_re + 128.0 * pixel_step;
            let center_dc_im = delta_im + 128.0 * pixel_step;
            let mut secondary = Vec::new();
            kernel::compute_secondary_orbit(
                &kernel::MandelbrotMap,
                &orbit,
                arith::DeltaC::new(center_dc_re, center_dc_im),
                max_iter,
                &mut secondary,
            );

            // Single pass with inline rebasing — no post-hoc second pass needed.
            let mut fallback_count = 0u32;
            for row in 0..TILE_SIZE {
                for col in 0..TILE_SIZE {
                    let dc_re = delta_re + col as f64 * pixel_step;
                    let dc_im = delta_im + row as f64 * pixel_step;
                    let dc_rel = arith::DeltaC::new(dc_re - center_dc_re, dc_im - center_dc_im);
                    let result = kernel::perturb_pixel(
                        &kernel::MandelbrotMap,
                        &orbit,
                        &secondary,
                        &bla,
                        arith::DeltaC::new(dc_re, dc_im),
                        dc_rel,
                        max_iter,
                    );
                    out[row * TILE_SIZE + col] = match result {
                        kernel::EscapeResult::Glitched => {
                            fallback_count += 1;
                            let c_abs = arith::Complex::new(cx_ref + dc_re, cy_ref + dc_im);
                            kernel::TilePixel::from(kernel::escape_time(
                                &kernel::MandelbrotMap,
                                c_abs,
                                max_iter,
                            ))
                        }
                        other => kernel::TilePixel::from(other),
                    };
                }
            }

            fallback_count
        })
    })
}
