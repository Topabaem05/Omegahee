#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define BLOCK_SIZE 1024

// Normalized-GPT residual update per row:
// output = norm(norm(residual) + weights * (norm(update) - norm(residual)))
// with norm(x) = x * rsqrt(sum(x^2) + epsilon). The two inner norms are rounded to T.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(NgptResidualMix)(
    const device T* residual,
    const device T* update,
    const device float* weights,
    device T* output,
    device T* output_copy OPTIONAL(write_copy),
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    const bool write_copy SPECIALIZE,
    threadgroup float shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(BLOCK_SIZE)
) {
  const ulong row_offset = (ulong)batch_idx * (ulong)element_count;
  residual += row_offset;
  update += row_offset;
  output += row_offset;
  if (write_copy) {
    output_copy += row_offset;
  }

  float residual_sum_of_squares = 0.0f;
  float update_sum_of_squares = 0.0f;
  for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
    const float residual_value = static_cast<float>(residual[i]);
    const float update_value = static_cast<float>(update[i]);
    residual_sum_of_squares += residual_value * residual_value;
    update_sum_of_squares += update_value * update_value;
  }
  const float residual_inv = rsqrt(
      threadgroup_cooperative_reduce<SimdReduceSum<float>, BLOCK_SIZE>(
          residual_sum_of_squares,
          shared_sum,
          thread_context
      ) +
      epsilon
  );
  const float update_inv = rsqrt(
      threadgroup_cooperative_reduce<SimdReduceSum<float>, BLOCK_SIZE>(update_sum_of_squares, shared_sum, thread_context) +
      epsilon
  );

  float mixed_sum_of_squares = 0.0f;
  for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
    const float base = static_cast<float>(static_cast<T>(static_cast<float>(residual[i]) * residual_inv));
    const float target = static_cast<float>(static_cast<T>(static_cast<float>(update[i]) * update_inv));
    const float mixed = base + weights[i] * (target - base);
    mixed_sum_of_squares += mixed * mixed;
  }
  const float mixed_inv = rsqrt(
      threadgroup_cooperative_reduce<SimdReduceSum<float>, BLOCK_SIZE>(mixed_sum_of_squares, shared_sum, thread_context) +
      epsilon
  );

  for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
    const float base = static_cast<float>(static_cast<T>(static_cast<float>(residual[i]) * residual_inv));
    const float target = static_cast<float>(static_cast<T>(static_cast<float>(update[i]) * update_inv));
    const float mixed = base + weights[i] * (target - base);
    const T result = static_cast<T>(mixed * mixed_inv);
    output[i] = result;
    if (write_copy) {
      output_copy[i] = result;
    }
  }
}
