#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/fp4.h"

using namespace metal;

// Expands FP4 e2m1 codes (two per byte, low nibble first) and per-group scales into dense rows:
// output[row * columns + column] = scale(row, column / group_size) * e2m1(code).
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(Fp4Dequantize)(
    const device uint8_t* codes,
    const device T* scales,
    device T* output,
    constant uint& rows,
    constant uint& packed_columns,
    constant uint& group_size,
    constant uint& scale_output_stride,
    constant uint& scale_group_stride,
    const uint pair AXIS(packed_columns, 64),
    const uint row AXIS(rows, 1)
) {
  const uint8_t byte = codes[(ulong)row * packed_columns + pair];
  const uint column = 2u * pair;
  const float scale = float(scales[row * scale_output_stride + (column / group_size) * scale_group_stride]);
  device T* destination = output + (ulong)row * (2u * packed_columns) + column;
  destination[0] = static_cast<T>(scale * fp4_e2m1_value(byte & 0xFu));
  destination[1] = static_cast<T>(scale * fp4_e2m1_value(byte >> 4));
}
