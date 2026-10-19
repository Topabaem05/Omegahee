#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/gated_act_mul.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;
using namespace uzu::activation_type;

#define BLOCK_SIZE 1024

// Fused nGPT MLP hidden activation per row of the fused [up | gate] projection:
//   up, gate = up * up_scale, gate * gate_scale   (rounded to T, optional)
//   hidden = up * silu(gate)                      (as GatedActMul, rounded to T)
//   output = hidden * rsqrt(sum(hidden^2) + epsilon)
// This replaces ScaleColumns + GatedActMul + a weightless sum-reduction Normalization.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(NgptSwiGLUNorm)(
    const device T* fused_up,
    const device float* channel_scales OPTIONAL(has_scales),
    device T* output,
    constant uint& batch_size,
    constant uint& hidden_dim,
    constant float& epsilon,
    const bool has_scales SPECIALIZE,
    threadgroup float shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(BLOCK_SIZE)
) {
  const device T* up_row = fused_up + (ulong)batch_idx * (ulong)(2 * hidden_dim);
  const device T* gate_row = up_row + hidden_dim;
  device T* out_row = output + (ulong)batch_idx * (ulong)hidden_dim;

  float sum_of_squares = 0.0f;
  for (uint i = thread_in_row; i < hidden_dim; i += BLOCK_SIZE) {
    T up = up_row[i];
    T gate = gate_row[i];
    if (has_scales) {
      up = static_cast<T>(static_cast<float>(up) * channel_scales[i]);
      gate = static_cast<T>(static_cast<float>(gate) * channel_scales[hidden_dim + i]);
    }
    const T hidden = static_cast<T>(gated_act_mul(up, gate, ActivationType::SILU));
    out_row[i] = hidden;
    const float value = static_cast<float>(hidden);
    sum_of_squares += value * value;
  }

  const float total = threadgroup_cooperative_reduce<SimdReduceSum<float>, BLOCK_SIZE>(
      sum_of_squares,
      shared_sum,
      thread_context
  );
  const float inverse_norm = rsqrt(total + epsilon);

  // Each thread rescales only the elements it wrote, so no barrier is needed.
  for (uint i = thread_in_row; i < hidden_dim; i += BLOCK_SIZE) {
    out_row[i] = static_cast<T>(static_cast<float>(out_row[i]) * inverse_norm);
  }
}
