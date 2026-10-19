#pragma once

#include "arguments.h"
#include "quant_chunk.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, typename AT, typename BT, typename DT, GemmBPrologueKind B_PROLOGUE, uint BITS>
struct QuantMetadata {
private:
  float scale[Tile::ROWS_PER_LANE];
  float origin[Tile::ROWS_PER_LANE];
  float bias[Tile::ROWS_PER_LANE];
  bool signed_codes;

public:
  static METAL_FUNC uint group_count(const thread GemvParams& params, uint group_size) {
    return (params.in_vec_size + group_size - 1) / group_size;
  }

  METAL_FUNC void load(
      uint group_index,
      const thread uint (&weight_rows)[Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params
  ) thread {
    signed_codes = params.signed_codes;
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint row = weight_rows[R];
      const uint scale_index = row * params.scale_output_stride + group_index * params.scale_group_stride;
      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleFp4E4M3Dequant) {
        // decode_fp4 yields e2m1 * 2^-14. The row scale is constant across groups, so QuantBSource applies it
        // once after the reduction.
        const uint8_t block_scale = reinterpret_cast<const device uint8_t*>(ops.scales)[scale_index];
        scale[R] = fp8_e4m3_scaled_down(block_scale) * (FP8_E4M3_HALF_SCALE * FP4_E2M1_HALF_SCALE);
        origin[R] = 0.0f;
        bias[R] = 0.0f;
      } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleFp4Dequant) {
        // decode_fp4 yields e2m1 * 2^-14.
        scale[R] = float(ops.scales[scale_index]) * FP4_E2M1_HALF_SCALE;
        origin[R] = 0.0f;
        bias[R] = 0.0f;
      } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
        scale[R] = float(ops.scales[scale_index]);
        constexpr uint ZERO_POINTS_PER_BYTE = QuantChunk<BITS>::BITS_PER_BYTE / BITS;
        const uint zero_point_index =
            row * params.zero_point_output_stride + group_index * params.zero_point_group_stride;
        const uint zero_point_row = zero_point_index / ZERO_POINTS_PER_BYTE;
        const uint packed_index = zero_point_index % ZERO_POINTS_PER_BYTE;
        origin[R] = FLOAT_OFFSET + float(decode_zero_point<BITS>(ops.zero_points + zero_point_row, packed_index));
        bias[R] = 0.0f;
      } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant) {
        scale[R] = float(ops.scales[scale_index]);
        origin[R] = FLOAT_OFFSET + float(symmetric_zero_point<BITS>());
        bias[R] = 0.0f;
      } else {
        static_assert(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant);
        scale[R] = float(ops.scales[scale_index]);
        origin[R] = FLOAT_OFFSET;
        bias[R] = float(ops.biases[scale_index]);
      }
    });
  }

  template <uint WORDS>
  METAL_FUNC void decode(
      const thread uint4 (&words)[WORDS],
      uint chunk,
      uint row,
      thread float (&values)[QuantChunk<BITS>::VALUES]
  ) const thread {
    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleFp4Dequant ||
                  B_PROLOGUE == GemmBPrologueKind::ScaleFp4E4M3Dequant) {
      QuantChunk<BITS>::decode_fp4(words, chunk, values);
    } else {
      QuantChunk<BITS>::decode(words, chunk, values, origin[row], signed_codes);
    }
  }

  METAL_FUNC void fold(
      thread float (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread float (&partial)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread float (&input_sum)[Tile::INPUT_ROWS]
  ) const thread {
    Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
      constexpr uint I = decltype(input_index)::value;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        result[I][R] = finish(result[I][R], partial[I][R], input_sum[I], R);
      });
    });
  }

private:
  METAL_FUNC float finish(float prior, float partial, float input_sum, uint row) const thread {
    float result = fma(scale[row], partial, prior);
    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
      result = fma(bias[row], input_sum, result);
    }
    return result;
  }
};

} // namespace gemm
} // namespace uzu
