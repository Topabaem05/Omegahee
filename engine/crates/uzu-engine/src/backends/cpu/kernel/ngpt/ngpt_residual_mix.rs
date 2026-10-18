use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

fn inverse_l2_norm(
    values: impl Iterator<Item = f32>,
    epsilon: f32,
) -> f32 {
    (values.map(|value| value * value).sum::<f32>() + epsilon).sqrt().recip()
}

/// Normalized-GPT residual update per row:
/// `output = norm(norm(residual) + weights * (norm(update) - norm(residual)))`
/// with `norm(x) = x * rsqrt(sum(x^2) + epsilon)`. The two inner norms are rounded to `T`.
#[kernel(NgptResidualMix)]
#[variants(T, f32, f16, bf16)]
pub fn ngpt_residual_mix<T: DataTypeRepr + Float>(
    residual: *const T,
    update: *const T,
    weights: *const f32,
    output: *mut T,
    #[optional(write_copy)] output_copy: Option<*mut T>,
    batch_size: u32,
    element_count: u32,
    epsilon: f32,
    #[specialize] write_copy: bool,
) {
    assert_eq!(output_copy.is_some(), write_copy);
    let element_count = element_count as usize;
    let mut mixed = vec![0.0f32; element_count];
    for row in 0..batch_size as usize {
        let residual_row = unsafe { std::slice::from_raw_parts(residual.add(row * element_count), element_count) };
        let update_row = unsafe { std::slice::from_raw_parts(update.add(row * element_count), element_count) };
        let residual_inv = inverse_l2_norm(residual_row.iter().map(|value| value.to_f32().unwrap()), epsilon);
        let update_inv = inverse_l2_norm(update_row.iter().map(|value| value.to_f32().unwrap()), epsilon);
        for (index, value) in mixed.iter_mut().enumerate() {
            let normalized_residual = T::from(residual_row[index].to_f32().unwrap() * residual_inv).unwrap();
            let normalized_update = T::from(update_row[index].to_f32().unwrap() * update_inv).unwrap();
            let base = normalized_residual.to_f32().unwrap();
            let weight = unsafe { *weights.add(index) };
            *value = base + weight * (normalized_update.to_f32().unwrap() - base);
        }
        let mixed_inv = inverse_l2_norm(mixed.iter().copied(), epsilon);
        for (index, value) in mixed.iter().enumerate() {
            let result = T::from(value * mixed_inv).unwrap();
            unsafe { *output.add(row * element_count + index) = result };
            if let Some(output_copy) = output_copy {
                unsafe { *output_copy.add(row * element_count + index) = result };
            }
        }
    }
}
