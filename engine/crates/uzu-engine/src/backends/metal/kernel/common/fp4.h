#pragma once

#include <metal_stdlib>

// FP4 e2m1 code: bit 3 sign, bits 2..1 exponent, bit 0 mantissa.
// Magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
// Reading (code & 7) << 9 | (code & 8) << 12 as FP16 gives e2m1(code) * 2^-14 exactly (e2m1 exponent
// bias 1 against FP16 bias 15, and the e = 0 code becomes the FP16 subnormal 2^-15). Callers that fold
// FP4_E2M1_HALF_SCALE into their scale decode with three integer operations and one conversion.
constant constexpr float FP4_E2M1_HALF_SCALE = 16384.0f;

METAL_FUNC float fp4_e2m1_scaled_down(uint code) {
  return float(as_type<half>(ushort(((code & 7u) << 9) | ((code & 8u) << 12))));
}

METAL_FUNC float fp4_e2m1_value(uint code) {
  const uint exponent = (code >> 1) & 3u;
  const uint mantissa = code & 1u;
  const uint normal_bits = ((exponent + 126u) << 23) | (mantissa << 22);
  const float magnitude = exponent == 0u ? 0.5f * float(mantissa) : as_type<float>(normal_bits);
  return (code & 8u) != 0u ? -magnitude : magnitude;
}
