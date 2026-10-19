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
