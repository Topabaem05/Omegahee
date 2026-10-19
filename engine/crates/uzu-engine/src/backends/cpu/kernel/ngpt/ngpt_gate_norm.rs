use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

/// Fused nGPT attention output per row: scaled gate (rounded to `T`), `attention * sigmoid(gate)`
/// (as SigmoidGate), then `x * rsqrt(sum(x^2) + epsilon)`.
#[kernel(NgptGateNorm)]
#[variants(T, f32, bf16)]
pub fn ngpt_gate_norm<T: DataTypeRepr + Float>(
    attention: *const T,
    gate: *const T,
    gate_scales: *const f32,
    output: *mut T,
    batch_size: u32,
    dim: u32,
    gate_row_stride: u32,
    epsilon: f32,
) {
    let dim = dim as usize;
    for row in 0..batch_size as usize {
        let attention_row = unsafe { attention.add(row * dim) };
        let gate_row = unsafe { gate.add(row * gate_row_stride as usize) };
        let out_row = unsafe { output.add(row * dim) };
        let mut sum_of_squares = 0.0f32;
        for index in 0..dim {
            let scaled_gate =
                T::from(unsafe { *gate_row.add(index) }.to_f32().unwrap() * unsafe { *gate_scales.add(index) })
                    .unwrap();
            let sigmoid = 1.0 / (1.0 + (-scaled_gate.to_f32().unwrap()).exp());
            let gated = T::from(unsafe { *attention_row.add(index) }.to_f32().unwrap() * sigmoid).unwrap();
            let value = gated.to_f32().unwrap();
            sum_of_squares += value * value;
            unsafe { *out_row.add(index) = gated };
        }
        let inverse_norm = (sum_of_squares + epsilon).sqrt().recip();
        for index in 0..dim {
            unsafe {
                let value = (*out_row.add(index)).to_f32().unwrap() * inverse_norm;
                *out_row.add(index) = T::from(value).unwrap();
            }
        }
    }
}
