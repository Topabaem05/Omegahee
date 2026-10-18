from collections.abc import Mapping

from jaxtyping import Array

from lalamo.modules.decoder import Decoder
from lalamo.modules.embedding import UntiedEmbedding
from lalamo.modules.learned_scale import LearnedScale
from lalamo.modules.mlp import DenseMLP
from lalamo.modules.ngpt_layer import NGPTAttentionLayer, NGPTMLPLayer
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.transformer_layer import TransformerLayerBase
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .huggingface import load_linear, load_untied_embedding

__all__ = ["load_dory_decoder"]


def _load_learned_scale(
    module: LearnedScale | None,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> LearnedScale:
    assert module is not None, f"Dory weights at {path} require a learned scale in the module."
    values = weights_dict[path].astype(module.values.dtype)
    return load_as_at(lambda m: (m.values,), module, (values,))


def _load_attention_layer(
    layer: NGPTAttentionLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> NGPTAttentionLayer:
    attention = layer.mixer
    assert isinstance(attention, Attention)
    mixer_path = path / "mixer"
    loaded_attention = load_as_at(
        lambda m: (m.qkvg_projection, m.out_projection, m.query_scale, m.gate_scale),
        attention,
        (
            load_linear(
                attention.qkvg_projection,
                weights_dict,
                mixer_path,
                sublayers_to_fuse=["q_proj", "k_proj", "v_proj", "g_proj"],
                implementation=implementation,
            ),
            load_linear(attention.out_projection, weights_dict, mixer_path / "o_proj", implementation=implementation),
            _load_learned_scale(attention.query_scale, weights_dict, mixer_path / "sqk"),
            _load_learned_scale(attention.gate_scale, weights_dict, mixer_path / "s_gate"),
        ),
    )
    return load_as_at(
        lambda m: (m.mixer, m.residual_scale),
        layer,
        (loaded_attention, _load_learned_scale(layer.residual_scale, weights_dict, path / "attn_alpha")),
    )


def _load_mlp_layer(
    layer: NGPTMLPLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> NGPTMLPLayer:
    mlp = layer.mlp
    assert isinstance(mlp, DenseMLP)
    mixer_path = path / "mixer"
    loaded_mlp = load_as_at(
        lambda m: (m.up_projection, m.down_projection, m.up_scale, m.gate_scale),
        mlp,
        (
            load_linear(
                mlp.up_projection,
                weights_dict,
                mixer_path,
                sublayers_to_fuse=["up_proj", "gate_proj"],
                implementation=implementation,
            ),
            load_linear(mlp.down_projection, weights_dict, mixer_path / "down_proj", implementation=implementation),
            _load_learned_scale(mlp.up_scale, weights_dict, mixer_path / "suv_up"),
            _load_learned_scale(mlp.gate_scale, weights_dict, mixer_path / "suv_gate"),
        ),
    )
    return load_as_at(
        lambda m: (m.mlp, m.residual_scale),
        layer,
        (loaded_mlp, _load_learned_scale(layer.residual_scale, weights_dict, path / "mlp_alpha")),
    )


def _load_layer(
    layer: TransformerLayerBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    implementation: CompressionImplementation,
) -> TransformerLayerBase:
    match layer:
        case NGPTAttentionLayer():
            return _load_attention_layer(layer, weights_dict, path, implementation)
        case NGPTMLPLayer():
            return _load_mlp_layer(layer, weights_dict, path, implementation)
        case _:
            raise TypeError(f"Unsupported Dory layer type: {type(layer)}")


def load_dory_decoder(
    module: Decoder,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Decoder:
    embedding = module.embedding
    assert isinstance(embedding, UntiedEmbedding)
    backbone_path = ParameterPath("backbone")
    lm_head_path = ParameterPath("lm_head")
    untied_embedding = load_untied_embedding(
        embedding,
        weights_dict,
        backbone_path / "embeddings",
        lm_head_path,
        implementation=implementation,
    )
    loaded_embedding = load_as_at(
        lambda m: (m.logit_scales,),
        untied_embedding,
        (_load_learned_scale(embedding.logit_scales, weights_dict, lm_head_path / "logit_scale"),),
    )
    layers = tuple(
        _load_layer(layer, weights_dict, backbone_path / "layers" / layer_index, implementation)
        for layer_index, layer in enumerate(module.transformer.layers)
    )
    return load_as_at(
        lambda m: (m.embedding, m.transformer.layers),
        module,
        (loaded_embedding, layers),
    )
