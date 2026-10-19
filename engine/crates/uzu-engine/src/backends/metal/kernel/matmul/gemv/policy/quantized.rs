use metal::MTLGPUFamily;

use super::{DEFAULT_RESULTS_PER_SIMDGROUP, GemvTile};
use crate::backends::metal::context::LARGE_MIN_GPU_CORES;

const QUANT_N_BUCKET_MAXES: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKET_MAXES: [u32; 3] = [512, 2048, 8192];

/// Lane-sliced tiles split each quantization group across reduction lanes.
/// A lane stages 8 bytes: 16 W4 values or 8 W8 values.
const fn lane_tile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
    bits: u32,
    group: u32,
) -> GemvTile {
    let pack_factor = match bits {
        4 => 8,
        8 => 4,
        _ => 1,
    };
    GemvTile::quantized(num_simdgroups, results_per_simdgroup, 1, 32, group / (2 * pack_factor))
}

const fn lane_default(
    bits: u32,
    group: u32,
) -> GemvTile {
    lane_tile(8, DEFAULT_RESULTS_PER_SIMDGROUP, bits, group)
}

pub fn gathered_tile(
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
) -> Option<GemvTile> {
    if !matches!(bits, 4 | 8) || !matches!(group, 32 | 64) || !(1..=8).contains(&m) {
        return None;
    }
    let tile = lane_default(bits, group);
    (n >= tile.rows_per_lane()).then_some(tile)
}

fn table_bucket_index(
    value: u32,
    bucket_maxes: &[u32],
) -> usize {
    bucket_maxes.partition_point(|&max| value > max)
}

fn lane_policy(
    gpu_core_count: u32,
    apple_gpu_family: MTLGPUFamily,
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
) -> GemvTile {
    let is_large_gpu = gpu_core_count >= LARGE_MIN_GPU_CORES;
    if m != 1 || bits != 4 {
        return lane_default(bits, group);
    }

    let k_bucket = table_bucket_index(k, &QUANT_K_BUCKET_MAXES);
    let n_bucket = table_bucket_index(n, &QUANT_N_BUCKET_MAXES);
    let (num_simdgroups, results_per_simdgroup) = match (is_large_gpu, apple_gpu_family, k_bucket, n_bucket) {
        (true, _, 0, 1) => (4, 2),
        (true, _, 1, 0) => (2, 1),
        (true, _, 1, 1..=3) => (2, 2),
        (true, _, 1, 4) => (2, 1),
        (true, _, 1, 5) => (2, 4),
        (true, _, 2, 1) => (4, 2),
        (true, _, 3, 1) => (2, 2),
        (false, MTLGPUFamily::Apple9, 0, 1) => (4, 4),
        (false, MTLGPUFamily::Apple9, 1, 0) => (4, 2),
        (false, MTLGPUFamily::Apple9, 1, 1) => (2, 2),
        (false, MTLGPUFamily::Apple9, 1, 2) => (4, 2),
        (false, MTLGPUFamily::Apple9, 1, 4) => (2, 2),
        (false, MTLGPUFamily::Apple9, 1, 5) => (4, 2),
        (false, MTLGPUFamily::Apple9, 2, 1) => (4, 2),
        (false, MTLGPUFamily::Apple9, 3, 1) => (2, 2),
        (false, MTLGPUFamily::Apple8, 0, 1) => (4, 4),
        (false, MTLGPUFamily::Apple8, 1, _) | (false, MTLGPUFamily::Apple8, 2, 1) => (8, 2),
        (false, family, 0, 1) if family < MTLGPUFamily::Apple8 => (4, 8),
        (false, family, 1, 0..=3) if family < MTLGPUFamily::Apple8 => (8, 2),
        _ => (8, 4),
    };
    let selected = lane_tile(num_simdgroups, results_per_simdgroup, bits, group);
    if n < selected.results_per_simdgroup {
        lane_default(bits, group)
    } else {
        selected
    }
}

/// MLX-style (scale and bias) 4-bit, group 64 weights with two input rows: one threadgroup reads
/// each weight once for both rows (the tile family the tuned zero-point routes use). Measured on
/// Dory decode shapes on an M5 Pro: 1.02-1.07x the one-row time, against 1.25-1.33x with
/// `lane_default`, which reads the weights once per row.
pub fn scale_bias_batch_tile(
    bits: u32,
    group: u32,
    m: u32,
) -> Option<GemvTile> {
    #[cfg(test)]
    if TILE_OVERRIDE.with(std::cell::Cell::get).is_some() {
        return None;
    }
    (bits == 4 && group == 64 && m == 2).then(|| GemvTile::quantized_output_tile(2, 16, m, 8, 1))
}

#[cfg(test)]
thread_local! {
    /// Test-only: forces the tile of 4-bit quantized GEMVs, for tile sweeps (`dory_bench.rs`).
    pub(crate) static TILE_OVERRIDE: std::cell::Cell<Option<GemvTile>> = const { std::cell::Cell::new(None) };
}

pub fn select(
    gpu_core_count: u32,
    apple_gpu_family: MTLGPUFamily,
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
    k: u32,
    bf16_io: bool,
) -> Option<GemvTile> {
    if !matches!(bits, 4 | 8) || !matches!(group, 32 | 64) || !(1..=8).contains(&m) {
        return None;
    }
    #[cfg(test)]
    if let Some(tile) = TILE_OVERRIDE.with(std::cell::Cell::get).filter(|_| bits == 4) {
        return Some(tile);
    }

    if !bf16_io {
        let tile = lane_default(bits, group);
        return (m <= 4 && n >= tile.rows_per_lane()).then_some(tile);
    }

    let tile = match m {
        1..=4 => lane_policy(gpu_core_count, apple_gpu_family, m, n, k, bits, group),
        5..=8 if n <= 64 => lane_default(bits, group),
        _ => return None,
    };
    // Input rows are tiled independently; partial tiles clamp loads and stores.
    if n < tile.rows_per_lane() {
        return None;
    }
    Some(tile)
}
