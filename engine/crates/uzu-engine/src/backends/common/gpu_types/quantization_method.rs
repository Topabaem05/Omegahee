use std::fmt;

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
}

impl fmt::Display for QuantizationMethod {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
