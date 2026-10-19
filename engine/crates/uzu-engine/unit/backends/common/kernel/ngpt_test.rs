use std::fmt::{Debug, Display};

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        kernel::{
            NgptGateNormKernel, NgptResidualMixKernel, NgptSwiGLUNormKernel, NormalizationKernel, ScaleColumnsKernel,
        },
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
    },
};

const EPSILON: f32 = 1e-6;

fn values(
    count: usize,
    seed: f32,
) -> Vec<f32> {
    (0..count).map(|i| (i as f32 * 0.37 + seed).sin() * 1.7 + (i as f32 * 0.011).cos() * 0.3).collect()
}

fn round<T: DataTypeRepr + Float>(value: f32) -> f32 {
    T::from(value).unwrap().to_f32().unwrap()
}

fn unit_l2(row: &[f32]) -> Vec<f32> {
    let inverse = (row.iter().map(|x| x * x).sum::<f32>() + EPSILON).sqrt().recip();
    row.iter().map(|x| x * inverse).collect()
}

fn tolerance<T: DataTypeRepr>() -> f32 {
    if T::data_type() == DataType::BF16 {
        1e-2
    } else {
        1e-5
    }
}

fn residual_mix_reference<T: DataTypeRepr + Float>(
    residual: &[T],
    update: &[T],
    weights: &[f32],
    element_count: usize,
) -> Vec<T> {
    residual
        .chunks(element_count)
        .zip(update.chunks(element_count))
        .flat_map(|(residual_row, update_row)| {
            let as_f32 = |row: &[T]| row.iter().map(|x| x.to_f32().unwrap()).collect::<Vec<_>>();
            let base = unit_l2(&as_f32(residual_row)).into_iter().map(round::<T>).collect::<Vec<_>>();
            let target = unit_l2(&as_f32(update_row)).into_iter().map(round::<T>).collect::<Vec<_>>();
            let mixed = (0..element_count).map(|i| base[i] + weights[i] * (target[i] - base[i])).collect::<Vec<_>>();
            unit_l2(&mixed).into_iter().map(|x| T::from(x).unwrap()).collect::<Vec<_>>()
        })
        .collect()
}

fn run_residual_mix<T: DataTypeRepr + Float, B: Backend>(
    residual: &[T],
    update: &[T],
    weights: &[f32],
    batch_size: u32,
    element_count: u32,
) -> Vec<T> {
    let context = block_on(B::Context::new()).expect("Failed to create Context");
    let kernel = <B::Kernels as Kernels>::NgptResidualMixKernel::new(&context, T::data_type(), true)
        .expect("Failed to create NgptResidualMixKernel");
    let residual = create_buffer_with_data::<B, T>(&context, residual);
    let update = create_buffer_with_data::<B, T>(&context, update);
    let weights = create_buffer_with_data::<B, f32>(&context, weights);
    let mut output = create_buffer::<B, T>(&context, (batch_size * element_count) as usize);
    let mut output_copy = create_buffer::<B, T>(&context, (batch_size * element_count) as usize);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &residual,
        &update,
        &weights,
        &mut output,
        Some(&mut output_copy),
        batch_size,
        element_count,
        EPSILON,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");
    let output: Vec<T> = buffer_to_vec(&output);
    let output_copy: Vec<T> = buffer_to_vec(&output_copy);
    assert!(
        output.iter().zip(&output_copy).all(|(a, b)| a.to_f32().unwrap().to_bits() == b.to_f32().unwrap().to_bits()),
        "both outputs must be identical"
    );
    output
}

fn test_residual_mix<T: DataTypeRepr + Float + Debug + Display>() {
    for (batch_size, element_count) in [(3u32, 16u32), (2, 2560)] {
        let length = (batch_size * element_count) as usize;
        let residual = values(length, 0.1).into_iter().map(|x| T::from(x).unwrap()).collect::<Vec<_>>();
        let update = values(length, 2.3).into_iter().map(|x| T::from(x * 40.0).unwrap()).collect::<Vec<_>>();
        let weights = values(element_count as usize, 5.0).into_iter().map(|x| x.abs() * 0.1).collect::<Vec<_>>();
        let expected = residual_mix_reference(&residual, &update, &weights, element_count as usize);
        for_each_backend!(|B| {
            let output = run_residual_mix::<T, B>(&residual, &update, &weights, batch_size, element_count);
            assert_eq_float::<T>(
                &expected,
                &output,
                tolerance::<T>(),
                &format!("NgptResidualMix {batch_size}x{element_count} on {}", std::any::type_name::<B>()),
            );
        });
    }
}

#[uzu_test]
fn test_residual_mix_f32() {
    test_residual_mix::<f32>();
}

#[uzu_test]
fn test_residual_mix_bf16() {
    test_residual_mix::<bf16>();
}

fn test_scale_columns<T: DataTypeRepr + Float + Debug + Display>() {
    let (num_rows, row_stride, column_offset, num_cols) = (3u32, 12u32, 4u32, 5u32);
    let data =
        values((num_rows * row_stride) as usize, 0.7).into_iter().map(|x| T::from(x).unwrap()).collect::<Vec<_>>();
    let scales = values(num_cols as usize, 1.9);
    let expected = data
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let column = index as u32 % row_stride;
            if (column_offset..column_offset + num_cols).contains(&column) {
                T::from(value.to_f32().unwrap() * scales[(column - column_offset) as usize]).unwrap()
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel = <<B as Backend>::Kernels as Kernels>::ScaleColumnsKernel::new(&context, T::data_type())
            .expect("Failed to create ScaleColumnsKernel");
        let mut buffer = create_buffer_with_data::<B, T>(&context, &data);
        let scales_buffer = create_buffer_with_data::<B, f32>(&context, &scales);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
        kernel.encode(&mut buffer, &scales_buffer, num_rows, num_cols, row_stride, column_offset, &mut command_buffer);
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");
        assert_eq_float::<T>(
            &expected,
            &buffer_to_vec::<B, T>(&buffer),
            tolerance::<T>(),
            &format!("ScaleColumns on {}", std::any::type_name::<B>()),
        );
    });
}

#[uzu_test]
fn test_scale_columns_f32() {
    test_scale_columns::<f32>();
}

#[uzu_test]
fn test_scale_columns_bf16() {
    test_scale_columns::<bf16>();
}

#[uzu_test]
fn test_normalization_sum_reduction_is_unit_l2() {
    let (batch_size, element_count) = (2u32, 2560u32);
    let input = values((batch_size * element_count) as usize, 0.3);
    let expected = input.chunks(element_count as usize).flat_map(unit_l2).collect::<Vec<_>>();
    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel = <<B as Backend>::Kernels as Kernels>::NormalizationKernel::new(
            &context,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            false,
            false,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
        )
        .expect("Failed to create NormalizationKernel");
        let input_buffer = create_buffer_with_data::<B, f32>(&context, &input);
        let mut output = create_buffer::<B, f32>(&context, input.len());
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
        kernel.encode(
            Some(&input_buffer),
            None::<&<B as Backend>::GlobalBuffer>,
            None::<&<B as Backend>::GlobalBuffer>,
            &mut output,
            None::<&mut <B as Backend>::GlobalBuffer>,
            None::<&<B as Backend>::GlobalBuffer>,
            batch_size,
            element_count,
            EPSILON,
            0.0,
            1.0,
            &mut command_buffer,
        );
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");
        assert_eq_float::<f32>(
            &expected,
            &buffer_to_vec::<B, f32>(&output),
            1e-6,
            &format!("L2 normalization on {}", std::any::type_name::<B>()),
        );
    });
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn test_swiglu_norm<T: DataTypeRepr + Float + Debug + Display>(has_scales: bool) {
    let (batch_size, hidden_dim) = (2u32, 2200u32);
    let hidden = hidden_dim as usize;
    let fused_up =
        values(batch_size as usize * 2 * hidden, 0.9).into_iter().map(|x| T::from(x).unwrap()).collect::<Vec<_>>();
    let scales = values(2 * hidden, 3.1).into_iter().map(|x| x.abs() + 0.5).collect::<Vec<_>>();
    let expected = fused_up
        .chunks(2 * hidden)
        .flat_map(|row| {
            let scaled = |index: usize| {
                let value = row[index].to_f32().unwrap();
                if has_scales {
                    round::<T>(value * scales[index])
                } else {
                    value
                }
            };
            let activated =
                (0..hidden).map(|i| round::<T>(scaled(i) * scaled(hidden + i) * sigmoid(scaled(hidden + i))));
            unit_l2(&activated.collect::<Vec<_>>()).into_iter().map(|x| T::from(x).unwrap())
        })
        .collect::<Vec<_>>();
    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel =
            <<B as Backend>::Kernels as Kernels>::NgptSwiGLUNormKernel::new(&context, T::data_type(), has_scales)
                .expect("Failed to create NgptSwiGLUNormKernel");
        let input = create_buffer_with_data::<B, T>(&context, &fused_up);
        let scales_buffer = create_buffer_with_data::<B, f32>(&context, &scales);
        let mut output = create_buffer::<B, T>(&context, batch_size as usize * hidden);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
        kernel.encode(
            &input,
            has_scales.then_some(&scales_buffer),
            &mut output,
            batch_size,
            hidden_dim,
            EPSILON,
            &mut command_buffer,
        );
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");
        assert_eq_float::<T>(
            &expected,
            &buffer_to_vec::<B, T>(&output),
            tolerance::<T>(),
            &format!("NgptSwiGLUNorm (scales {has_scales}) on {}", std::any::type_name::<B>()),
        );
    });
}

#[uzu_test]
fn test_swiglu_norm_f32() {
    test_swiglu_norm::<f32>(true);
    test_swiglu_norm::<f32>(false);
}

#[uzu_test]
fn test_swiglu_norm_bf16() {
    test_swiglu_norm::<bf16>(true);
    test_swiglu_norm::<bf16>(false);
}

fn test_gate_norm<T: DataTypeRepr + Float + Debug + Display>() {
    let (batch_size, dim, gate_offset, gate_row_stride) = (3u32, 1280u32, 7u32, 1300u32);
    let dim_usize = dim as usize;
    let attention =
        values(batch_size as usize * dim_usize, 0.4).into_iter().map(|x| T::from(x).unwrap()).collect::<Vec<_>>();
    // The gate lives at an offset inside a wider row, as in the fused qkvg projection.
    let gate_rows = values(batch_size as usize * gate_row_stride as usize + gate_offset as usize, 1.3)
        .into_iter()
        .map(|x| T::from(x * 3.0).unwrap())
        .collect::<Vec<_>>();
    let gate_scales = values(dim_usize, 2.2).into_iter().map(|x| x.abs() + 0.2).collect::<Vec<_>>();
    let expected = (0..batch_size as usize)
        .flat_map(|row| {
            let gated = (0..dim_usize)
                .map(|i| {
                    let gate = gate_rows[gate_offset as usize + row * gate_row_stride as usize + i].to_f32().unwrap();
                    let scaled_gate = round::<T>(gate * gate_scales[i]);
                    round::<T>(attention[row * dim_usize + i].to_f32().unwrap() * sigmoid(scaled_gate))
                })
                .collect::<Vec<_>>();
            unit_l2(&gated).into_iter().map(|x| T::from(x).unwrap())
        })
        .collect::<Vec<_>>();
    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel = <<B as Backend>::Kernels as Kernels>::NgptGateNormKernel::new(&context, T::data_type())
            .expect("Failed to create NgptGateNormKernel");
        let attention_buffer = create_buffer_with_data::<B, T>(&context, &attention);
        let gate_buffer = create_buffer_with_data::<B, T>(&context, &gate_rows);
        let gate = gate_buffer.subrange(gate_offset as usize * T::data_type().size_in_bytes()..);
        let scales_buffer = create_buffer_with_data::<B, f32>(&context, &gate_scales);
        let mut output = create_buffer::<B, T>(&context, attention.len());
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
        kernel.encode(
            &attention_buffer,
            gate,
            &scales_buffer,
            &mut output,
            batch_size,
            dim,
            gate_row_stride,
            EPSILON,
            &mut command_buffer,
        );
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");
        assert_eq_float::<T>(
            &expected,
            &buffer_to_vec::<B, T>(&output),
            tolerance::<T>(),
            &format!("NgptGateNorm on {}", std::any::type_name::<B>()),
        );
    });
}

#[uzu_test]
fn test_gate_norm_f32() {
    test_gate_norm::<f32>();
}

#[uzu_test]
fn test_gate_norm_bf16() {
    test_gate_norm::<bf16>();
}
