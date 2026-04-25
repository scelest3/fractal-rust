//! Tile priority queue — dispatches `TileJob`s to Tile Workers in priority order.
//!
//! Scheduler owns no I/O. The Scheduler Worker wraps this queue in a message
//! loop; the queue logic itself is fully testable with `cargo test`.
//! Full implementation in Phase 1.

/// A single unit of render work dispatched to a Tile Worker.
#[derive(Copy, Clone, Debug)]
pub struct TileJob {
    pub tile_x: i32,
    pub tile_y: i32,
    /// Iteration depth level: 0 = 64 iter, 1 = 512 iter, 2 = max iter.
    pub iter_level: u8,
    pub max_iter: u32,
    /// Monotonically increasing ID of the reference orbit to use.
    pub ref_orbit_id: u64,
    pub priority: f32,
}

/// Stub priority queue. Full BinaryHeap-backed implementation in Phase 1.
pub struct TileQueue {
    jobs: Vec<TileJob>,
}

impl TileQueue {
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    pub fn push(&mut self, job: TileJob) {
        self.jobs.push(job);
    }

    pub fn pop_next(&mut self, current_orbit_id: u64) -> Option<TileJob> {
        let pos = self
            .jobs
            .iter()
            .position(|j| j.ref_orbit_id == current_orbit_id)?;
        Some(self.jobs.swap_remove(pos))
    }

    pub fn invalidate(&mut self, old_orbit_id: u64) {
        self.jobs.retain(|j| j.ref_orbit_id != old_orbit_id);
    }
}

impl Default for TileQueue {
    fn default() -> Self {
        Self::new()
    }
}
