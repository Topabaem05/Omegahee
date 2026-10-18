from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.dory_loader import load_dory_decoder
from lalamo.models import LanguageModel
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.modules.learned_scale import LearnedScaleConfig, ScaleTransform
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.ngpt_layer import NGPTAttentionLayerConfig, NGPTMLPLayerConfig
from lalamo.modules.normalization import NormalizationConfig, NormReduction, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import RecurrenceConfig, RecurrentKVCacheMode, TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfigBase
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig

__all__ = ["HFDoryConfig"]


class DoryLayerType(StrEnum):
    SLIDING_ATTENTION = "sliding_attention"
    FULL_ATTENTION = "full_attention"
    MLP = "mlp"


class DoryRoPEType(StrEnum):
    DEFAULT = "default"


@dataclass(frozen=True)
class DoryRoPEParameters:
    rope_type: DoryRoPEType
    rope_theta: float
    partial_rotary_factor: float = 1.0


@dataclass(frozen=True)
class DoryRoPEParametersByLayerType:
    sliding_attention: DoryRoPEParameters
    full_attention: DoryRoPEParameters


@dataclass(frozen=True)
class HFDoryConfig(HuggingFaceLMConfig):
    """NVIDIA Dory: a looped normalized-GPT transformer (vLLM PR #60149, `DoryForCausalLM`)."""

    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    hidden_act: str
    initializer_range: float
    sliding_window: int
    rope_parameters: DoryRoPEParametersByLayerType
    layer_types: tuple[DoryLayerType, ...]
    ngpt_alpha_init: float
    attention_output_gate_logit_scale_init: float | None
    mlp_bias: bool
    attention_bias: bool
    n_input_layers: int
    n_recurrent_layers: int
    n_output_layers: int
    n_recurrent_loops: int
    recurrent_kv_cache_mode: RecurrentKVCacheMode = RecurrentKVCacheMode.PER_LOOP

    def __post_init__(self) -> None:
        if self.model_type != "dory":
            raise ValueError(f"Expected model_type 'dory', got {self.model_type!r}.")
        if self.hidden_act != "silu" or self.tie_word_embeddings or self.mlp_bias or self.attention_bias:
            raise ValueError("Dory requires SiLU, untied embeddings, and bias-free projections.")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries.")
        if self.n_input_layers + self.n_recurrent_layers + self.n_output_layers != self.num_hidden_layers:
            raise ValueError("Dory input, recurrent, and output layers must sum to num_hidden_layers.")

    @property
    def eos_token_ids(self) -> list[int]:
        # The Dory checkpoint config has no eos_token_id; the stop tokens come from generation_config.json.
        return []

    def _rope_config(self, layer_type: DoryLayerType) -> UnscaledRoPEConfig:
        match layer_type:
            case DoryLayerType.SLIDING_ATTENTION:
                parameters = self.rope_parameters.sliding_attention
            case DoryLayerType.FULL_ATTENTION:
                parameters = self.rope_parameters.full_attention
            case DoryLayerType.MLP:
                raise ValueError("MLP layers have no positional embeddings.")
        return UnscaledRoPEConfig(
            base=parameters.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=int(self.head_dim * parameters.partial_rotary_factor),
        )

    def to_decoder_config(
        self,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        unit_norm_config = NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
            has_scale=False,
            reduction=NormReduction.SUM,
        )
        linear_config = LinearConfig()
        gate_logit_scale_init = self.attention_output_gate_logit_scale_init
        if gate_logit_scale_init is None:
            gate_logit_scale_init = self.hidden_size**0.5
        residual_scale_config = LearnedScaleConfig(
            multiplier=self.ngpt_alpha_init / self.initializer_range,
            transform=ScaleTransform.ABSOLUTE,
        )
        mlp_channel_scale_config = LearnedScaleConfig(
            multiplier=self.hidden_size**0.5,
            transform=ScaleTransform.IDENTITY,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            gate_clipping=None,
            up_clipping=None,
            up_scale_config=mlp_channel_scale_config,
            gate_scale_config=mlp_channel_scale_config,
            hidden_norm_config=unit_norm_config,
        )

        def attention_config(layer_type: DoryLayerType) -> AttentionConfig:
            return AttentionConfig(
                qkvg_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=unit_norm_config,
                key_norm_config=unit_norm_config,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=self.head_dim**-0.5,
                sliding_window_size=self.sliding_window if layer_type == DoryLayerType.SLIDING_ATTENTION else None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkvg_biases=False,
                has_out_biases=False,
                has_gate=True,
                query_scale_config=LearnedScaleConfig(
                    multiplier=self.head_dim**0.5 / self.initializer_range,
                    transform=ScaleTransform.SQUARE,
                ),
                gate_scale_config=LearnedScaleConfig(
                    multiplier=gate_logit_scale_init**0.5 / self.initializer_range,
                    transform=ScaleTransform.SQUARE,
                ),
                output_norm_config=unit_norm_config,
            )

        def layer_config(layer_type: DoryLayerType) -> TransformerLayerConfigBase:
            if layer_type == DoryLayerType.MLP:
                return NGPTMLPLayerConfig(
                    mlp_config=mlp_config,
                    residual_norm_config=unit_norm_config,
                    residual_scale_config=residual_scale_config,
                )
            return NGPTAttentionLayerConfig(
                mixer_config=attention_config(layer_type),
                residual_norm_config=unit_norm_config,
                residual_scale_config=residual_scale_config,
                rope_config=self._rope_config(layer_type),
            )

        transformer_config = TransformerConfig(
            layer_configs=tuple(layer_config(layer_type) for layer_type in self.layer_types),
            output_norm_config=None,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            recurrence=RecurrenceConfig(
                start_layer_index=self.n_input_layers,
                end_layer_index=self.n_input_layers + self.n_recurrent_layers,
                num_loops=self.n_recurrent_loops,
                kv_cache_mode=self.recurrent_kv_cache_mode,
                reinjection_norm_config=unit_norm_config,
            ),
        )
        embedding_config = UntiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            logit_scales_config=LearnedScaleConfig(
                multiplier=1 / self.initializer_range,
                transform=ScaleTransform.IDENTITY,
            ),
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, LanguageModel)
        decoder = load_dory_decoder(model.decoder, weights_dict, implementation=implementation)
        return eqx.tree_at(lambda m: (m.decoder,), model, (decoder,))
