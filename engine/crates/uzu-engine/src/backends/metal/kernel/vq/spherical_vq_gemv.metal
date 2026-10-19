#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define SVQ_SIMDGROUPS 8
#define SVQ_ROWS_PER_SIMDGROUP 4
#define SVQ_CODEBOOK_ENTRIES 256

// Experimental spherical vector-quantized GEMV (2.5 bits per weight, docs/dory_hybrid_loop_experiment_plan.md
// has the speed track). Each 8-weight vector is
//   signs(bits) * (min + code * step) * codebook[direction]
// with unit directions in the positive orthant (256 FP16 entries), a 2-bit radius code per vector and one
// FP16 (min, step) pair per 16 vectors (128 weights). Planes over [rows, columns / 8] vectors, row-major:
//   signs: a byte per vector, bit i negates element i; directions: a byte per vector;
//   radius_codes: 2 bits per vector, 16 per 32-bit word; radius_grid: (min, step) per 16 vectors.
// A lane takes two adjacent vectors per step; a simdgroup takes four rows. The codebook and a table of
// sign masks (bits 15 and 31 of each half pair, per sign byte) sit in threadgroup memory, so a vector
// decodes with two table reads and four XORs, once for every batch row.
template <typename T, uint BATCH>
VARIANTS(T, bfloat)
VARIANTS(BATCH, 1, 2)
PUBLIC KERNEL(SphericalVqGemv)(
    const device uint8_t* signs,
    const device uint8_t* directions,
    const device uint32_t* radius_codes,
    const device half* radius_grid,
    const device half* codebook,
    const device T* input,
    device T* output,
    const constant uint32_t& k,
    const constant uint32_t& n,
    threadgroup uint4 shared_codebook[SVQ_CODEBOOK_ENTRIES],
    threadgroup uint4 shared_sign_masks[256],
    const ThreadContext thread_context,
    const uint32_t row_block GROUPS(n.div_ceil(SVQ_SIMDGROUPS * SVQ_ROWS_PER_SIMDGROUP)),
    const uint32_t simd_lane THREADS(32),
    const uint32_t simd_group THREADS(SVQ_SIMDGROUPS)
) {
  const uint32_t thread_index = simd_group * 32 + simd_lane;
  shared_codebook[thread_index] = reinterpret_cast<const device uint4*>(codebook)[thread_index];
  uint4 masks;
  for (uint32_t pair = 0; pair < 4; ++pair) {
    masks[pair] = (((thread_index >> (2 * pair)) & 1u) << 15) | (((thread_index >> (2 * pair + 1)) & 1u) << 31);
  }
  shared_sign_masks[thread_index] = masks;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint32_t vectors_per_row = k / 8;
  const uint32_t first_row = (row_block * SVQ_SIMDGROUPS + simd_group) * SVQ_ROWS_PER_SIMDGROUP;
  float acc[BATCH][SVQ_ROWS_PER_SIMDGROUP];
  for (uint32_t b = 0; b < BATCH; ++b) {
    for (uint32_t r = 0; r < SVQ_ROWS_PER_SIMDGROUP; ++r) {
      acc[b][r] = 0.0f;
    }
  }

  for (uint32_t vector = simd_lane * 2; vector < vectors_per_row; vector += 64) {
    // Inputs stay packed as bf16 pairs (widened on use) to keep the batch-2 tile in registers.
    uint4 x[BATCH][2];
    for (uint32_t b = 0; b < BATCH; ++b) {
      const device uint4* source = reinterpret_cast<const device uint4*>(input + b * k + vector * 8);
      x[b][0] = source[0];
      x[b][1] = source[1];
    }
    for (uint32_t r = 0; r < SVQ_ROWS_PER_SIMDGROUP; ++r) {
      const uint32_t row = min(first_row + r, n - 1);
      const uint32_t index = row * vectors_per_row + vector;
      const uint32_t sign_pair = *reinterpret_cast<const device ushort*>(signs + index);
      const uint32_t direction_pair = *reinterpret_cast<const device ushort*>(directions + index);
      const uint32_t codes = radius_codes[index / 16] >> (2 * (index % 16));
      const float grid_min = static_cast<float>(radius_grid[2 * (index / 16)]);
      const float grid_step = static_cast<float>(radius_grid[2 * (index / 16) + 1]);
      for (uint32_t j = 0; j < 2; ++j) {
        const uint4 signed_bits =
            shared_codebook[(direction_pair >> (8 * j)) & 0xFFu] ^ shared_sign_masks[(sign_pair >> (8 * j)) & 0xFFu];
        float weights[8];
        for (uint32_t pair = 0; pair < 4; ++pair) {
          const float2 values = static_cast<float2>(as_type<half2>(signed_bits[pair]));
          weights[2 * pair] = values.x;
          weights[2 * pair + 1] = values.y;
        }
        const float radius = grid_min + static_cast<float>((codes >> (2 * j)) & 3u) * grid_step;
        for (uint32_t b = 0; b < BATCH; ++b) {
          float dot = 0.0f;
          for (uint32_t pair = 0; pair < 4; ++pair) {
            const uint32_t packed = x[b][j][pair];
            dot = fma(weights[2 * pair], as_type<float>(packed << 16), dot);
            dot = fma(weights[2 * pair + 1], as_type<float>(packed & 0xFFFF0000u), dot);
          }
          acc[b][r] = fma(radius, dot, acc[b][r]);
        }
      }
    }
  }

  for (uint32_t r = 0; r < SVQ_ROWS_PER_SIMDGROUP; ++r) {
    const uint32_t row = first_row + r;
    for (uint32_t b = 0; b < BATCH; ++b) {
      const float total = simd_sum(acc[b][r]);
      if (simd_lane == 0 && row < n) {
        output[b * n + row] = static_cast<T>(total);
      }
    }
  }
}
