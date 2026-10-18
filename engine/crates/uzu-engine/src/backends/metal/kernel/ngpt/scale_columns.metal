#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

// In place data[row * row_stride + column_offset + column] *= scales[column] for column < num_cols.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(ScaleColumns)(
    device T* data,
    const device float* scales,
    constant uint& num_rows,
    constant uint& num_cols,
    constant uint& row_stride,
    constant uint& column_offset,
    const uint column AXIS(num_cols, 32),
    const uint row AXIS(num_rows, 1)
) {
  device T* element = data + (ulong)row * (ulong)row_stride + column_offset + column;
  *element = static_cast<T>(static_cast<float>(*element) * scales[column]);
}
