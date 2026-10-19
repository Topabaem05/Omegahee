use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::gpu_types::quantization_method::FP4_E2M1_VALUES, data_type::DataTypeRepr};

/// Expands FP4 e2m1 codes (two per byte, low nibble first) and per-group scales into dense rows.
#[kernel(Fp4Dequantize)]
#[variants(T, f32, bf16)]
pub fn fp4_dequantize<T: DataTypeRepr + Float>(
    codes: *const u8,
    scales: *const T,
    output: *mut T,
    rows: u32,
    packed_columns: u32,
    group_size: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
) {
    let packed_columns = packed_columns as usize;
    for row in 0..rows as usize {
        for pair in 0..packed_columns {
            let byte = unsafe { *codes.add(row * packed_columns + pair) };
            let column = 2 * pair;
            let scale_index =
                row * scale_output_stride as usize + (column / group_size as usize) * scale_group_stride as usize;
            let scale = unsafe { *scales.add(scale_index) }.to_f32().unwrap();
            for (offset, code) in [byte & 0xF, byte >> 4].into_iter().enumerate() {
                let value = scale * FP4_E2M1_VALUES[usize::from(code)];
                unsafe { *output.add(row * 2 * packed_columns + column + offset) = T::from(value).unwrap() };
            }
        }
    }
}
