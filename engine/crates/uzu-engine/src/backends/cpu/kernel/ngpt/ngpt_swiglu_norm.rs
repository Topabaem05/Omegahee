use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::gpu_types::activation_type::ActivationType, data_type::DataTypeRepr};

/// Fused nGPT MLP hidden activation per row of the fused `[up | gate]` projection: optional channel
/// scales, `up * silu(gate)` as GatedActMul computes it, then `x * rsqrt(sum(x^2) + epsilon)`.
#[kernel(NgptSwiGLUNorm)]
#[variants(T, f32, bf16)]
pub fn ngpt_swiglu_norm<T: DataTypeRepr + Float>(
    fused_up: *const T,
    #[optional(has_scales)] channel_scales: Option<*const f32>,
    output: *mut T,
    batch_size: u32,
    hidden_dim: u32,
    epsilon: f32,
    #[specialize] has_scales: bool,
) {
    assert_eq!(channel_scales.is_some(), has_scales);
    let hidden_dim = hidden_dim as usize;
    let scale = |value: T, index: usize| match channel_scales {
        Some(scales) => T::from(value.to_f32().unwrap() * unsafe { *scales.add(index) }).unwrap(),
        None => value,
    };
    for row in 0..batch_size as usize {
        let up_row = unsafe { fused_up.add(row * 2 * hidden_dim) };
        let out_row = unsafe { output.add(row * hidden_dim) };
        let mut sum_of_squares = 0.0f32;
        for index in 0..hidden_dim {
            let up = scale(unsafe { *up_row.add(index) }, index);
            let gate = scale(unsafe { *up_row.add(hidden_dim + index) }, hidden_dim + index);
            let hidden = up * ActivationType::SILU.activate(gate);
            let value = hidden.to_f32().unwrap();
            sum_of_squares += value * value;
            unsafe { *out_row.add(index) = hidden };
        }
        let inverse_norm = (sum_of_squares + epsilon).sqrt().recip();
        for index in 0..hidden_dim {
            unsafe {
                let value = (*out_row.add(index)).to_f32().unwrap() * inverse_norm;
                *out_row.add(index) = T::from(value).unwrap();
            }
        }
    }
}
