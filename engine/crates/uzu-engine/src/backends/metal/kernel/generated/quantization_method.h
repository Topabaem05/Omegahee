// Auto-generated from gpu_types/quantization_method - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::quantization_method {
static constant constexpr float FP4_E2M1_VALUES[16] =
    {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

enum class QuantizationMethod : uint32_t {
  ScaleBias = 0,
  ScaleZeroPoint = 1,
  ScaleSymmetric = 2,
  ScaleFp4 = 3,
};
} // namespace uzu::quantization_method
