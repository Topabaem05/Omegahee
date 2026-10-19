use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

/// Experimental spherical vector-quantized GEMV (see the Metal kernel for the layout): each 8-weight vector
/// is `signs * (min + code * step) * codebook[direction]`.
#[kernel(SphericalVqGemv)]
#[variants(T, bf16)]
#[variants(BATCH, 1, 2)]
pub fn spherical_vq_gemv<T: DataTypeRepr + Float, const BATCH: u32>(
    signs: *const u8,
    directions: *const u8,
    radius_codes: *const u32,
    radius_grid: *const f16,
    codebook: *const f16,
    input: *const T,
    output: *mut T,
    k: u32,
    n: u32,
) {
    let (k, n) = (k as usize, n as usize);
    let vectors_per_row = k / 8;
    for row in 0..n {
        for b in 0..BATCH as usize {
            let mut total = 0.0f32;
            for vector in 0..vectors_per_row {
                let index = row * vectors_per_row + vector;
                unsafe {
                    let sign_bits = *signs.add(index);
                    let direction = *directions.add(index) as usize;
                    let code = (*radius_codes.add(index / 16) >> (2 * (index % 16))) & 3;
                    let radius = (*radius_grid.add(2 * (index / 16))).to_f32()
                        + code as f32 * (*radius_grid.add(2 * (index / 16) + 1)).to_f32();
                    let mut dot = 0.0f32;
                    for i in 0..8 {
                        let value = (*codebook.add(direction * 8 + i)).to_f32();
                        let value = if (sign_bits >> i) & 1 == 1 {
                            -value
                        } else {
                            value
                        };
                        dot += value * (*input.add(b * k + vector * 8 + i)).to_f32().unwrap();
                    }
                    total += radius * dot;
                }
            }
            unsafe { *output.add(b * n + row) = T::from(total).unwrap() };
        }
    }
}
