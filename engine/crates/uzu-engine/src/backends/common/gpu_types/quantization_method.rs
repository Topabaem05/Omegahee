use std::fmt;

/// FP8 E4M3 (`e4m3fn`: no infinities, 0x7F/0xFF are NaN) as f32.
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 {
        -1.0
    } else {
        1.0
    };
    let exponent = i32::from((byte >> 3) & 0xF);
    let mantissa = f32::from(byte & 0x7);
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else if exponent == 0xF && mantissa == 7.0 {
        f32::NAN
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    sign * magnitude
}

/// FP4 e2m1 values by code (bit 3 sign, bits 2..1 exponent, bit 0 mantissa).
pub const FP4_E2M1_VALUES: [f32; 16] =
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum QuantizationMethod {
    ScaleBias,
    ScaleZeroPoint,
    ScaleSymmetric,
    /// FP4 (e2m1) codes times a per-group scale (NVFP4 values with the tensor scale folded in).
    ScaleFp4,
    /// FP4 (e2m1) codes times an FP8 E4M3 group scale times an FP32 per-row scale (NVFP4 as stored).
    ScaleFp4E4M3,
}

impl fmt::Display for QuantizationMethod {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
