/**
 * Shared memory layout calculation — mirrors crates/wasm-bridge/src/lib.rs.
 *
 * Used on the main thread to size the WebAssembly.Memory before any WASM
 * instance exists. The wasm-pack integration tests (issue #5) verify that
 * the Rust implementation produces identical values.
 */

// ── Constants (must stay in sync with wasm-bridge/src/lib.rs) ─────────────

const PRIMARY_ORBIT_ENTRY_BYTES = 32;
const BLA_ENTRY_BYTES = 48;
const SECONDARY_ORBIT_ENTRY_BYTES = 16;
const SECONDARY_ORBIT_SLOTS = 3;
const ORBIT_HEADER_BYTES = 8;
const TILE_RING_SLOTS_PER_WORKER = 4;
export const TILE_SLOT_BYTES = 256 * 256 * 4 * 4; // 1 048 576 B
const SLOT_STATE_ENTRY_BYTES = 4;
const WASM_PAGE_BYTES = 65536;

// ── Types ──────────────────────────────────────────────────────────────────

export interface MemoryLayout {
  primaryOrbitOffset: number;
  blaTableOffset: number;
  secondaryOrbitOffset: number;
  slotStateOffset: number;
  tileRingOffset: number;
  tileSlotBytes: number;
  totalBytes: number;
  /** Number of 64 KiB WebAssembly.Memory pages required. */
  pageCount: number;
}

// ── Layout computation ─────────────────────────────────────────────────────

export function computeLayout(nWorkers: number, maxIter: number): MemoryLayout {
  const primaryOrbitOffset = 0;
  const primaryOrbitSize = ORBIT_HEADER_BYTES + maxIter * PRIMARY_ORBIT_ENTRY_BYTES;

  const blaTableOffset = primaryOrbitOffset + primaryOrbitSize;
  const blaTableSize = maxIter * BLA_ENTRY_BYTES;

  const secondaryOrbitOffset = blaTableOffset + blaTableSize;
  const oneSecondarySize = ORBIT_HEADER_BYTES + maxIter * SECONDARY_ORBIT_ENTRY_BYTES;
  const secondaryOrbitSize = SECONDARY_ORBIT_SLOTS * oneSecondarySize;

  const totalRingSlots = TILE_RING_SLOTS_PER_WORKER * nWorkers;
  const slotStateOffset = secondaryOrbitOffset + secondaryOrbitSize;
  const slotStateSize = totalRingSlots * SLOT_STATE_ENTRY_BYTES;

  const tileRingOffset = slotStateOffset + slotStateSize;
  const tileRingSize = totalRingSlots * TILE_SLOT_BYTES;

  const totalBytes = tileRingOffset + tileRingSize;
  const pageCount = Math.ceil(totalBytes / WASM_PAGE_BYTES);

  return {
    primaryOrbitOffset,
    blaTableOffset,
    secondaryOrbitOffset,
    slotStateOffset,
    tileRingOffset,
    tileSlotBytes: TILE_SLOT_BYTES,
    totalBytes,
    pageCount,
  };
}
