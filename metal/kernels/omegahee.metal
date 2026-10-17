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
