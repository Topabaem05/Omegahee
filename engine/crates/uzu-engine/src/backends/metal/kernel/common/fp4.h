#pragma once

#include <metal_stdlib>

// FP4 e2m1 code: bit 3 sign, bits 2..1 exponent, bit 0 mantissa.
// Magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
// Reading (code & 7) << 9 | (code & 8) << 12 as FP16 gives e2m1(code) * 2^-14 exactly (e2m1 exponent
// bias 1 against FP16 bias 15, and the e = 0 code becomes the FP16 subnormal 2^-15). Callers that fold
// FP4_E2M1_HALF_SCALE into their scale decode with three integer operations and one conversion.
constant constexpr float FP4_E2M1_HALF_SCALE = 16384.0f;

// FP8 E4M3 (e4m3fn) byte as float: bias 7, subnormals below exponent 1.
METAL_FUNC float fp8_e4m3_value(uint byte) {
  const uint exponent = (byte >> 3) & 15u;
  const uint mantissa = byte & 7u;
  const float magnitude = exponent == 0u ? float(mantissa) * (1.0f / 512.0f)
                                         : as_type<float>(((exponent + 120u) << 23) | (mantissa << 20));
  return (byte & 0x80u) != 0u ? -magnitude : magnitude;
}

// Reading (byte & 0x7F) << 7 | (byte & 0x80) << 8 as FP16 gives e4m3(byte) * 2^-8 exactly, subnormals included
// (E4M3 bias 7 against FP16 bias 15, 3 mantissa bits in the top of the FP16 mantissa). The NaN byte 0x7F reads
// as 480, which block scales never hold.
constant constexpr float FP8_E4M3_HALF_SCALE = 256.0f;

METAL_FUNC float fp8_e4m3_scaled_down(uint byte) {
  return float(as_type<half>(ushort(((byte & 0x7Fu) << 7) | ((byte & 0x80u) << 8))));
}

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
