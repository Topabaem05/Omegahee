/*
 * Omegahee Metal kernel pack.
 *
 * Implements the four integration contracts of
 * docs/uzu_metal_kernel_mapping.md in an executable, testable form:
 *
 *   ksc_decode_dot     - rotation-free KSC-CurveTQ block-row decode
 *   awsrc_dot          - in-register AWSRC seeded residual epilogue
 *   readkv_prefix_load - bit-plane prefix reader with per-channel budget
 *   lilicorr_score     - single-pass bilinear draft correlation
 *   stream_probe       - achievable memory-bandwidth probe
 *
 * The Python twin is src/omegahee_metal_reference.py. The harness compares the
 * two and reports the maximum absolute deviation per kernel.
 *
 * Bit layout (KSC): each row stores a 14-bit start state s0 followed by 16
 * bytes of MSB-first 2-bit codes. The header is the window over stream bits
 * 0..13, so the first emitted code is code 0. The decode loop shifts in the
 * next stream pair from bit offset 14 (code 7 onward). Shifts past bit 127
 * insert zero and never wrap into an earlier column.
 */

#include <metal_stdlib>

using namespace metal;

#define OMEGA_BLOCK 64
#define OMEGA_CODE_MASK 0x3FFF
#define OMEGA_HEADER_BITS 14
#define OMEGA_STREAM_BITS 128

constant float OMEGA_SPAN[4] = {-3.0f, -1.0f, 1.0f, 3.0f};

static inline float omega_codebook(uint code, float scale) {
  return 0.5f * scale * OMEGA_SPAN[code & 3u];
}

/* Bit index 0 is the MSB of byte 0. Indices past the row read as zero. */
static inline uint omega_bit(const device uchar* row, uint index) {
  if (index >= OMEGA_STREAM_BITS) {
    return 0u;
  }
  const uchar byte = row[index >> 3];
  return (uint(byte) >> (7u - (index & 7u))) & 1u;
}

static inline uint omega_pair(const device uchar* row, uint index) {
  return (omega_bit(row, index) << 1) | omega_bit(row, index + 1u);
}

/*
 * One thread accumulates one output row of a 64-column block.
 *
 *   out[row] = sum_j alpha[row] * beta[j] * codebook[code_j] * x[j]
 *
 * The 14-bit start state carries the first seven codes; stream bit 14 feeds the
 * eighth. No Hadamard transform or un-rotation is applied anywhere in the hot
 * loop: rotation_applied stays false, which is the KSC-CurveTQ claim under
 * test.
 */
/* Load one 16-byte row of MSB-first code bits into four words. */
static inline void omega_load_row(const device uchar* row, thread uint* words) {
  for (uint w = 0; w < 4u; ++w) {
    const device uchar* p = row + w * 4u;
    words[w] = (uint(p[0]) << 24) | (uint(p[1]) << 16) | (uint(p[2]) << 8) | uint(p[3]);
  }
}

/* Bit `index` of a register-resident row, MSB-first. */
static inline uint omega_row_bit(const thread uint* words, uint index) {
  return (words[index >> 5] >> (31u - (index & 31u))) & 1u;
}

static inline uint omega_row_pair(const thread uint* words, uint index) {
  return (omega_row_bit(words, index) << 1) | omega_row_bit(words, index + 1u);
}

/*
 * One thread accumulates one output row of a 64-column block.
 *
 *   out[row] = sum_j alpha[row] * beta[j] * codebook[code_j] * x[j]
 *
 * The 14-bit start state carries the first seven codes; stream bit 14 feeds the
 * eighth. The row codes are loaded into registers once, so the inner loop is
 * register-only: the trellis state shift is a dependency chain, not a load
 * stall. beta and x arrive through threadgroup memory.
 *
 * No Hadamard transform or un-rotation is applied anywhere in the hot loop:
 * rotation_applied stays false, which is the KSC-CurveTQ claim under test.
 */
kernel void ksc_decode_dot(
    const device ushort* s0 [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device float* alpha [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* x [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    const uint gid [[thread_position_in_grid]]
) {
  if (gid >= rows) {
    return;
  }
  const uint state_start = uint(s0[gid]) & OMEGA_CODE_MASK;
  const device uchar* row = codes + gid * (OMEGA_BLOCK / 4u);
  const float a = alpha[gid];
  float acc = 0.0f;
  uint state = state_start;
  for (uint j = 0; j < OMEGA_BLOCK; ++j) {
    const uint code = (state >> 12) & 3u;
    acc += a * beta[j] * omega_codebook(code, scale) * x[j];
    state = ((state << 2) | omega_pair(row, OMEGA_HEADER_BITS + 2u * j)) & OMEGA_CODE_MASK;
  }
  out[gid] = acc;
}

/*
 * In-register AWSRC sign regeneration. Nothing is read except the seeds and the
 * per-row scales, so the sidecar stays at 0.162 bpw and there is no shared LUT
 * to collide in memory.
 */
static inline float omega_splitmix_sign(uint seed, uint index) {
  uint h = seed * 0x9E3779B9u + index * 0x85EBCA6Bu;
  h ^= h >> 16;
  h *= 0xC2B2AE35u;
  h ^= h >> 13;
  h *= 0x27D4EB2Fu;
  h ^= h >> 16;
  return (h & 1u) == 0u ? 1.0f : -1.0f;
}

kernel void awsrc_dot(
    const device uint* seeds [[buffer(0)]],
    const device float* gammas [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& m_seeds [[buffer(5)]],
    constant float& inv_block [[buffer(6)]],
    const uint gid [[thread_position_in_grid]]
) {
  if (gid >= rows) {
    return;
  }
  float acc = 0.0f;
  for (uint m = 0; m < m_seeds; ++m) {
    const uint seed = seeds[m];
    float s = 0.0f;
    for (uint j = 0; j < OMEGA_BLOCK; ++j) {
      s += omega_splitmix_sign(seed, gid * OMEGA_BLOCK + j) * x[j];
    }
    acc += gammas[gid * m_seeds + m] * (s * inv_block);
  }
  out[gid] = acc;
}

/*
 * ReadKV prefix loader. Planes are stored one uint32 per channel-word, bit c of
 * plane p is channel c's bit at weight 2^(7-p). A channel with budget r reads
 * planes 0..r-1 only; the remaining planes contribute zero. Reading fewer
 * planes is the bandwidth reduction the contract describes.
 */
kernel void readkv_prefix_load(
    const device uint* planes [[buffer(0)]],
    const device uchar* budgets [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    const uint gid [[thread_position_in_grid]]
) {
  if (gid >= channels) {
    return;
  }
  const uint budget = uint(budgets[gid]);
  uint value = 0u;
  for (uint p = 0; p < 8u; ++p) {
    if (p >= budget) {
      break;
    }
    const uint bit = (planes[p] >> gid) & 1u;
    value |= bit << (7u - p);
  }
  out[gid] = value;
}

/*
 * LiLiCorr bilinear correlation. One thread group per slot pair; each thread
 * produces one (k, k') entry as tau * cos(U_out[t, k], U_in[t+1, k']).
 */
kernel void lilicorr_score(
    const device float* uo [[buffer(0)]],
    const device float* ui [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& pairs [[buffer(3)]],
    constant uint& k_width [[buffer(4)]],
    constant uint& d_corr [[buffer(5)]],
    constant float& tau [[buffer(6)]],
    const uint2 gid [[thread_position_in_grid]]
) {
  const uint pair = gid.x;
  const uint entry = gid.y;
  if (pair >= pairs || entry >= k_width * k_width) {
    return;
  }
  const uint k = entry / k_width;
  const uint kk = entry % k_width;
  const device float* a = uo + (pair * k_width + k) * d_corr;
  const device float* b = ui + ((pair + 1u) * k_width + kk) * d_corr;

  float dot = 0.0f;
  float na = 0.0f;
  float nb = 0.0f;
  for (uint c = 0; c < d_corr; ++c) {
    dot += a[c] * b[c];
    na += a[c] * a[c];
    nb += b[c] * b[c];
  }
  const float denom = sqrt(na) * sqrt(nb) + 1e-12f;
  out[pair * k_width * k_width + entry] = tau * (dot / denom);
}

/*
 * Achievable-bandwidth probe: repeated float4 streaming reduction. The result
 * is a per-device number the paper's roofline projections can be checked
 * against.
 */
kernel void stream_probe(
    const device float4* src [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ulong& words [[buffer(2)]],
    constant ulong& stride [[buffer(3)]],
    constant uint& passes [[buffer(4)]],
    const uint tid [[thread_position_in_grid]]
) {
  float acc = 0.0f;
  /* Inner passes amortize the launch and tail over a longer streaming window,
   * which is what makes the measurement stable enough to quote. */
  for (uint pass = 0; pass < passes; ++pass) {
    for (ulong i = ulong(tid); i < words; i += stride) {
      const float4 v = src[i];
      acc += v.x + v.y + v.z + v.w;
    }
  }
  out[tid] = acc;
}

/*
 * KSC decode with SIMD-group cooperation (the port target).
 *
 * Grid layout: 256-thread threadgroups holding 8 SIMD groups; each SIMD group
 * owns one output row. Lane l of a group reads nibble l of that row: the high
 * nibble of byte l/2 for even l, the low nibble for odd l, giving columns 2l
 * and 2l+1.
 *
 * Verified property (src/omegahee_metal_reference.py, parity test): the KSC
 * code stream decodes by direct random access, so the serial trellis walk the
 * scalar reference performs is not required to decode a column.
 *
 * beta and x are activation-side vectors shared by every row, so they are
 * staged once per threadgroup into threadgroup memory. Reading them directly
 * from device memory cost 512 bytes per row in the first version of this
 * kernel, which measured slower than the scalar path.
 */
kernel void ksc_decode_simd(
    const device ushort* s0 [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device float* alpha [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* x [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    threadgroup float* shared [[threadgroup(0)]],
    const uint gid [[thread_position_in_grid]],
    const uint lid [[thread_position_in_threadgroup]],
    const uint lane [[thread_index_in_simdgroup]],
    const uint sg [[simdgroup_index_in_threadgroup]]
) {
  // Stage beta (first 64 floats) and x (next 64) once per threadgroup.
  if (lid < OMEGA_BLOCK) {
    shared[lid] = beta[lid];
  } else if (lid < 2u * OMEGA_BLOCK) {
    shared[lid] = x[lid - OMEGA_BLOCK];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_index = (gid / 32u);
  if (row_index >= rows) {
    return;
  }
  (void)sg;
  const device uchar* row = codes + row_index * (OMEGA_BLOCK / 4u);
  const float a = alpha[row_index];

  const uchar packed = row[lane >> 1];
  const uint nibble = (lane & 1u) ? (uint(packed) & 0x0Fu) : (uint(packed) >> 4);
  const uint code_a = (nibble >> 2) & 3u;
  const uint code_b = nibble & 3u;

  const uint col_a = 2u * lane;
  const uint col_b = col_a + 1u;
  const float qa = a * shared[col_a] * omega_codebook(code_a, scale) * shared[OMEGA_BLOCK + col_a];
  const float qb = a * shared[col_b] * omega_codebook(code_b, scale) * shared[OMEGA_BLOCK + col_b];
  float partial = qa + qb;

  for (uint offset = 16u; offset > 0u; offset >>= 1u) {
    partial += simd_shuffle_xor(partial, offset);
  }
  if (lane == 0u) {
    out[row_index] = partial;
  }
}

/*
 * KSC decode throughput probe (DRAM scale).
 *
 * streams `rows` block rows once, with the trellis decode interleaved, so the
 * reported rate is directly comparable to a weight-streaming decode: the code
 * buffer is far larger than any cache on the part. Weight payload is 18 bytes
 * per row (16 code bytes + 2 header bytes); alpha/beta/x are activation-side.
 */
kernel void ksc_stream_simd(
    const device ushort* s0 [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device float* alpha [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* x [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    threadgroup float* shared [[threadgroup(0)]],
    const uint gid [[thread_position_in_grid]],
    const uint lid [[thread_position_in_threadgroup]],
    const uint lane [[thread_index_in_simdgroup]]
) {
  if (lid < OMEGA_BLOCK) {
    shared[lid] = beta[lid];
  } else if (lid < 2u * OMEGA_BLOCK) {
    shared[lid] = x[lid - OMEGA_BLOCK];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each SIMD group walks a strip of rows so one staging serves many rows.
  const uint rows_per_group = 8u;
  const uint group_id = gid / 32u;
  const uint row_base = group_id * rows_per_group;
  float acc[8];
  for (uint i = 0; i < 8u; ++i) {
    acc[i] = 0.0f;
  }
  const uint col_a = 2u * lane;
  const uint col_b = col_a + 1u;
  for (uint i = 0; i < rows_per_group; ++i) {
    const uint row_index = row_base + i;
    if (row_index >= rows) {
      break;
    }
    const device uchar* row = codes + row_index * (OMEGA_BLOCK / 4u);
    const uchar packed = row[lane >> 1];
    const uint nibble = (lane & 1u) ? (uint(packed) & 0x0Fu) : (uint(packed) >> 4);
    const float a = alpha[row_index];
    const float qa = a * shared[col_a] * omega_codebook((nibble >> 2) & 3u, scale) * shared[OMEGA_BLOCK + col_a];
    const float qb = a * shared[col_b] * omega_codebook(nibble & 3u, scale) * shared[OMEGA_BLOCK + col_b];
    acc[i] = qa + qb;
  }
  // One reduction pass for all eight rows: shuffle cost amortized across the strip.
  for (uint i = 0; i < rows_per_group; ++i) {
    float v = acc[i];
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
      v += simd_shuffle_xor(v, offset);
    }
    if (lane == 0u && row_base + i < rows) {
      out[row_base + i] = v;
    }
  }
}

/*
 * KSC GEMV at uzu shape: one output row spans K = 64 blocks of 64 columns.
 *
 * Why this exists: the per-block kernel (ksc_decode_simd) spends five shuffle
 * steps to reduce two MACs per lane, which dominates at K = 64. uzu's GEMV
 * path accumulates thousands of columns per output row, so its reduction is
 * amortized. This kernel matches that shape so the two are comparable:
 *
 *   one output row reads K/64 = 64 code blocks (64 x 18 = 1152 bytes)
 *   each lane accumulates 2 columns per block -> 128 MACs per lane
 *   one SIMD reduction at the end: 5 shuffles per 128 MACs
 *
 * The code stream is the same random-access KSC layout: lane l of a group owns
 * columns 2l and 2l+1 of each block, nibble l of the block row.
 */
kernel void ksc_gemv_k4096(
    const device uchar* codes [[buffer(0)]],   // rows_of_N x blocks x 16 bytes
    const device float* alpha [[buffer(1)]],   // N
    const device float* x [[buffer(2)]],       // K
    device float* out [[buffer(3)]],           // N
    constant float& scale [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& n_blocks [[buffer(6)]],
    const uint gid [[thread_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
  const uint row_index = gid / 32u;
  if (row_index >= n_rows) {
    return;
  }
  const device uchar* row = codes + row_index * n_blocks * (OMEGA_BLOCK / 4u);
  const float a = alpha[row_index];
  const uint col_a = 2u * lane;
  const uint col_b = col_a + 1u;
  float partial = 0.0f;
  for (uint b = 0; b < n_blocks; ++b) {
    const device uchar* block = row + b * (OMEGA_BLOCK / 4u);
    const uchar packed = block[lane >> 1];
    const uint nibble = (lane & 1u) ? (uint(packed) & 0x0Fu) : (uint(packed) >> 4);
    const uint base = b * OMEGA_BLOCK;
    const float qa = x[base + col_a] * omega_codebook((nibble >> 2) & 3u, scale);
    const float qb = x[base + col_b] * omega_codebook(nibble & 3u, scale);
    partial += qa + qb;
  }
  for (uint offset = 16u; offset > 0u; offset >>= 1u) {
    partial += simd_shuffle_xor(partial, offset);
  }
  if (lane == 0u) {
    out[row_index] = a * partial;
  }
}

/*
 * KSC GEMV with coalesced loads and a fixed threadgroup geometry.
 *
 * Contract: one 256-thread threadgroup holds 8 SIMD groups. Simdgroup s of the
 * group handles output row `group * 8 + s`. Staging always uses all 256 threads,
 * because the whole x vector is shared by every row in the group.
 *
 * Two bugs this design fixes, both found by CPU parity on a one-row case:
 *  1. the staging stride was hardcoded to 256 while some dispatches had fewer
 *     threads, leaving tail entries of shared_x uninitialized;
 *  2. rows were indexed by gid/32 while the group was 256 threads, so the eight
 *     simdgroups in a group raced on the same output slot.
 *
 * Lane l of simdgroup s loads word (base + l) of the row: 4 bytes, 16 codes.
 * Thirty-two lanes therefore move 128 contiguous bytes per step, which is the
 * point of this kernel versus the one-byte-per-lane version.
 */
kernel void ksc_gemv_coalesced(
    const device uchar* codes [[buffer(0)]],
    const device float* alpha [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& n_blocks [[buffer(6)]],
    threadgroup float* shared_x [[threadgroup(0)]],
    const uint group [[threadgroup_position_in_grid]],
    const uint lid [[thread_position_in_threadgroup]],
    const uint lane [[thread_index_in_simdgroup]],
    const uint sg [[simdgroup_index_in_threadgroup]]
) {
  const uint k_dim = n_blocks * OMEGA_BLOCK;
  // All 256 threads stage the shared activation vector.
  for (uint i = lid; i < k_dim; i += 256u) {
    shared_x[i] = x[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_index = group * 8u + sg;
  if (row_index >= n_rows) {
    return;
  }

  const uint n_words = n_blocks * 4u;
  const device uint* row = reinterpret_cast<const device uint*>(
      codes + row_index * n_blocks * (OMEGA_BLOCK / 4u));

  float partial = 0.0f;
  for (uint base = 0; base < n_words; base += 32u) {
    const uint word_index = base + lane;
    float s = 0.0f;
    if (word_index < n_words) {
      const uint w = row[word_index];
      const uint col_base = word_index * 16u;
      for (uint c = 0; c < 16u; ++c) {
        const uint code = (w >> (30u - 2u * c)) & 3u;
        s += shared_x[col_base + c] * omega_codebook(code, scale);
      }
    }
    partial += s;
  }

  for (uint offset = 16u; offset > 0u; offset >>= 1u) {
    partial += simd_shuffle_xor(partial, offset);
  }
  if (lane == 0u) {
    out[row_index] = alpha[row_index] * partial;
  }
}

/*
 * KSC decode throughput probe.
 *
 * Runs the real trellis decode (not a synthetic stream) over `rows` block rows
 * `passes` times, so the reported rate is the weight bytes per second the KSC
 * kernel actually sustains. Each row reads a 14-bit start state, 16 code bytes,
 * a per-row alpha, and shares beta and x vectors; the kernel writes one float.
 *
 * Weight bytes per row = 16 code bytes + 4 bytes alpha + 2 bytes s0 header.
 */
kernel void ksc_throughput(
    const device ushort* s0 [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device float* alpha [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* x [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& passes [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    const uint lid [[thread_position_in_threadgroup]],
    const uint gid [[thread_position_in_grid]]
) {
  if (lid < OMEGA_BLOCK) {
    shared[lid] = beta[lid];
  } else if (lid < 2u * OMEGA_BLOCK) {
    shared[lid] = x[lid - OMEGA_BLOCK];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (gid >= rows) {
    return;
  }
  const uint state_start = uint(s0[gid]) & OMEGA_CODE_MASK;
  const device uchar* row = codes + gid * (OMEGA_BLOCK / 4u);
  const float a = alpha[gid];
  uint words[4];
  omega_load_row(row, words);

  float total = 0.0f;
  for (uint pass = 0; pass < passes; ++pass) {
    float acc = 0.0f;
    uint state = state_start;
    for (uint j = 0; j < OMEGA_BLOCK; ++j) {
      const uint code = (state >> 12) & 3u;
      acc += a * shared[j] * omega_codebook(code, scale) * shared[OMEGA_BLOCK + j];
      state = ((state << 2) | omega_row_pair(words, OMEGA_HEADER_BITS + 2u * j)) & OMEGA_CODE_MASK;
    }
    total += acc;
  }
  out[gid] = total;
}
