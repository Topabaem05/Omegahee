//! Decode-shaped quantized GEMVs of Dory 4B (hidden 2560, 16 query heads x 256 with a gate, 4 KV heads,
//! MLP hidden 7168, vocabulary 131072). Batch 1 is plain decode; batches 2 and 4 show what a token
//! batch (for example loop wavefront verification) would cost on the same weights.

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Backend,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::Metal,
    },
    tests::{
        cold_pool::ColdPool,
        matmul::{QuantBuffers, QuantInput, Shape, iter_encode_loop, quant_arguments},
        util::type_short_name,
    },
};

const DORY_LINEARS: &[(&str, u32, u32)] =
    &[("qkvg", 2560, 10240), ("out", 4096, 2560), ("up", 2560, 14336), ("down", 7168, 2560)];
const DORY_READOUT: (&str, u32, u32) = ("readout", 2560, 131072);

fn bench_dory_shapes(
    c: &mut Criterion,
    label: &str,
    shapes: &[(&str, u32, u32)],
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let context = crate::tests::util::shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/DoryLayers/{}", type_short_name::<Metal>(), label));
    for &(layer, k, n) in shapes {
        for m in [1u32, 2, 4] {
            let shape = Shape::new(m, k, n);
            let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, quant_method, 42);
            let mut buffers =
                ColdPool::new(input.weight_buffer_bytes(), || QuantBuffers::<Metal, bf16>::allocate(&context, &input));
            let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
            )
            .unwrap();
            // Throughput in weight bytes, so criterion reports the achieved weight bandwidth.
            group.throughput(Throughput::Bytes(input.weight_buffer_bytes() as u64));
            group.bench_function(BenchmarkId::from_parameter(format!("{layer}_{shape}")), |b| {
                iter_encode_loop::<Metal, _>(&context, b, |command_buffer| {
                    matmul
                        .encode(quant_arguments(buffers.next_mut(), &input), command_buffer)
                        .expect("encode dory layer");
                });
            });
        }
    }
}

#[uzu_bench]
fn bench_dory_layers(c: &mut Criterion) {
    bench_dory_shapes(c, "ScaleBias_BF16_gs64_4b", DORY_LINEARS, 64, 4, QuantizationMethod::ScaleBias);
    bench_dory_shapes(c, "ScaleBias_BF16_gs32_4b", DORY_LINEARS, 32, 4, QuantizationMethod::ScaleBias);
    bench_dory_shapes(c, "Sym_BF16_gs64_8b", &[DORY_READOUT], 64, 8, QuantizationMethod::ScaleSymmetric);
    bench_dory_shapes(c, "ScaleBias_BF16_gs64_4b_readout", &[DORY_READOUT], 64, 4, QuantizationMethod::ScaleBias);
}

/// Candidate quantized GEMV tiles for small batches: (simdgroups, results per simdgroup, input rows
/// per threadgroup, reduction lanes, lanes per quantization group).
const SWEEP_TILES: &[(u32, u32, u32, u32, u32)] = &[(8, 4, 1, 32, 4), (2, 8, 2, 8, 1), (2, 4, 2, 8, 1)];

fn encode_once(
    context: &crate::backends::metal::context::MetalContext,
    matmul: &mut <<Metal as Backend>::Kernels as Kernels>::MatmulKernel,
    buffers: &mut QuantBuffers<Metal, bf16>,
    input: &QuantInput<bf16>,
) -> Option<Vec<bf16>> {
    use crate::backends::common::{CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context};
    let mut command_buffer = context.create_command_buffer(None, None).ok()?;
    matmul.encode(quant_arguments(buffers, input), &mut command_buffer).ok()?;
    futures::executor::block_on(command_buffer.end_encoding().submit().wait_until_completed()).ok()?;
    Some(crate::tests::helpers::buffer_to_vec::<Metal, bf16>(&buffers.y))
}

/// Times each candidate tile on the Dory shapes after checking its output against the default tile.
#[uzu_bench]
fn bench_dory_gemv_tiles(c: &mut Criterion) {
    use crate::backends::metal::kernel::matmul::gemv::{GemvTile, TILE_OVERRIDE};
    let context = crate::tests::util::shared_metal_context();
    let shapes = DORY_LINEARS.iter().chain(std::iter::once(&DORY_READOUT));
    for &(layer, k, n) in shapes {
        for m in [2u32] {
            let input = QuantInput::<bf16>::new(m, k, n, 64, 4, QuantizationMethod::ScaleBias, 7);
            let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
            )
            .unwrap();
            let mut check_buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
            TILE_OVERRIDE.with(|tile| tile.set(None));
            let expected = encode_once(&context, &mut matmul, &mut check_buffers, &input).expect("default tile");
            let mut group = c.benchmark_group(format!("Metal/Kernel/DoryTiles/{layer}_M{m}"));
            for &(simdgroups, results, input_rows, lanes, group_lanes) in SWEEP_TILES {
                let tile = GemvTile::quantized(simdgroups, results, input_rows, lanes, group_lanes);
                TILE_OVERRIDE.with(|override_tile| override_tile.set(Some(tile)));
                let Some(output) = encode_once(&context, &mut matmul, &mut check_buffers, &input) else {
                    eprintln!(
                        "{layer} m{m} tile {:?}: encode failed",
                        (simdgroups, results, input_rows, lanes, group_lanes)
                    );
                    continue;
                };
                let max_error = expected
                    .iter()
                    .zip(&output)
                    .map(|(a, b)| (f32::from(*a) - f32::from(*b)).abs() / f32::from(*a).abs().max(1.0))
                    .fold(0.0f32, f32::max);
                if max_error > 2e-2 {
                    eprintln!(
                        "{layer} m{m} tile {:?}: wrong output (max relative error {max_error})",
                        (simdgroups, results, input_rows, lanes, group_lanes)
                    );
                    continue;
                }
                let mut buffers = ColdPool::new(input.weight_buffer_bytes(), || {
                    QuantBuffers::<Metal, bf16>::allocate(&context, &input)
                });
                group.throughput(Throughput::Bytes(input.weight_buffer_bytes() as u64));
                group.bench_function(
                    BenchmarkId::from_parameter(format!(
                        "sg{simdgroups}_r{results}_in{input_rows}_l{lanes}_g{group_lanes}"
                    )),
                    |b| {
                        iter_encode_loop::<Metal, _>(&context, b, |command_buffer| {
                            matmul.encode(quant_arguments(buffers.next_mut(), &input), command_buffer).expect("encode");
                        });
                    },
                );
            }
            TILE_OVERRIDE.with(|tile| tile.set(None));
            group.finish();
        }
    }
}

/// Error of INT4 GEMV tiles against the CPU reference (f32 accumulation) on the Dory shapes: the
/// batch-1 default tile and, at batch 2, the default and two-row tiles.
#[uzu_engine_macros::uzu_test]
fn dory_gemv_tile_accuracy() {
    use crate::backends::metal::kernel::matmul::gemv::{GemvTile, TILE_OVERRIDE};
    let context = crate::tests::util::shared_metal_context();
    for &(layer, k, n) in DORY_LINEARS.iter().chain(std::iter::once(&DORY_READOUT)) {
        for (m, tile) in [
            (1u32, None),
            (2, Some(GemvTile::quantized(8, 4, 1, 32, 4))),
            (2, Some(GemvTile::quantized(2, 8, 2, 8, 1))),
        ] {
            let input = QuantInput::<bf16>::new(m, k, n, 64, 4, QuantizationMethod::ScaleBias, 11);
            let reference = crate::tests::matmul::quant::run_quant_cpu::<bf16>(&input);
            let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
                <bf16 as crate::data_type::DataTypeRepr>::data_type(),
            )
            .unwrap();
            let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
            TILE_OVERRIDE.with(|override_tile| override_tile.set(tile));
            let output = encode_once(&context, &mut matmul, &mut buffers, &input).expect("encode");
            TILE_OVERRIDE.with(|override_tile| override_tile.set(None));
            let scale = reference.iter().map(|value| f32::from(*value).abs()).fold(0.0f32, f32::max);
            let errors = reference.iter().zip(&output).map(|(a, b)| (f32::from(*a) - f32::from(*b)).abs());
            let (max_error, sum_error) = errors.fold((0.0f32, 0.0f32), |(max, sum), e| (max.max(e), sum + e));
            eprintln!(
                "{layer} m{m} tile {:?}: max error {:.3e}, mean error {:.3e} (output max {:.3e})",
                tile,
                max_error,
                sum_error / reference.len() as f32,
                scale
            );
        }
    }
}
