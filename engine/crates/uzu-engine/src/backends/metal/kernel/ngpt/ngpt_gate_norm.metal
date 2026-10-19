#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../attention/attention_gate.h"

using namespace metal;

#define BLOCK_SIZE 1024

// Fused nGPT attention output per row:
//   gate = gate * gate_scale            (rounded to T, as ScaleColumns)
//   gated = attention * sigmoid(gate)   (as SigmoidGate)
//   output = gated * rsqrt(sum(gated^2) + epsilon)
// This replaces ScaleColumns + SigmoidGate + a weightless sum-reduction Normalization.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(NgptGateNorm)(
    const device T* attention,
    const device T* gate,
    const device float* gate_scales,
    device T* output,
    constant uint& batch_size,
    constant uint& dim,
    constant uint& gate_row_stride,
    constant float& epsilon,
    threadgroup float shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(BLOCK_SIZE)
) {
  const device T* attention_row = attention + (ulong)batch_idx * (ulong)dim;
  const device T* gate_row = gate + (ulong)batch_idx * (ulong)gate_row_stride;
  device T* out_row = output + (ulong)batch_idx * (ulong)dim;

  float sum_of_squares = 0.0f;
  for (uint i = thread_in_row; i < dim; i += BLOCK_SIZE) {
    const T scaled_gate = static_cast<T>(static_cast<float>(gate_row[i]) * gate_scales[i]);
    const T gated = attention_apply_gate(attention_row[i], scaled_gate);
    out_row[i] = gated;
    const float value = static_cast<float>(gated);
    sum_of_squares += value * value;
  }
  const float total = threadgroup_cooperative_reduce<SimdReduceSum<float>, BLOCK_SIZE>(
      sum_of_squares,
      shared_sum,
      thread_context
  );
  const float inverse_norm = rsqrt(total + epsilon);
  for (uint i = thread_in_row; i < dim; i += BLOCK_SIZE) {
    out_row[i] = static_cast<T>(static_cast<float>(out_row[i]) * inverse_norm);
  }
}
