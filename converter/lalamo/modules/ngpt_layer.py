from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LogicalAxis

from .learned_scale import LearnedScale, LearnedScaleConfig
from .mlp import MLPBase, MLPConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .token_mixer import StateLayerBase, TokenMixerBase, TokenMixerConfig
from .transformer_layer import (
    NGPTLayerActivationTrace,
    TransformerForwardPassConfig,
    TransformerLayerBase,
    TransformerLayerConfigBase,
    TransformerLayerResult,
)
from .utils import call_vmapped, call_vmapped_twice, gather_suffix_tokens

__all__ = [
    "NGPTAttentionLayer",
    "NGPTAttentionLayerConfig",
    "NGPTLayerConfigBase",
    "NGPTMLPLayer",
    "NGPTMLPLayerConfig",
]


@dataclass(frozen=True)
class NGPTLayerConfigBase(TransformerLayerConfigBase):
    """A single sublayer followed by the normalized-GPT residual update.

    outputs = norm(norm(inputs) + scale * (norm(sublayer(inputs)) - norm(inputs)))
    """

    residual_norm_config: NormalizationConfig
    residual_scale_config: LearnedScaleConfig

    def __post_init__(self) -> None:
        if self.residual_norm_config.has_scale or self.residual_norm_config.has_biases:
            raise ValueError("The nGPT residual normalization must be weightless.")
        if self.kv_source_layer_index is not None:
            raise ValueError("nGPT layers do not support KV sharing.")


class NGPTLayerBase[ConfigT: NGPTLayerConfigBase](TransformerLayerBase[ConfigT]):
    residual_norm: Normalization
    residual_scale: LearnedScale

    def _interpolate(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        sublayer_outputs: Float[Array, "batch suffix_tokens channels"],
    ) -> Float[Array, "batch suffix_tokens channels"]:
        normalized_inputs = call_vmapped_twice(self.residual_norm, inputs).astype(jnp.float32)
        normalized_outputs = call_vmapped_twice(self.residual_norm, sublayer_outputs).astype(jnp.float32)
        weights = self.residual_scale.effective_scale()
        mixed = normalized_inputs + weights * (normalized_outputs - normalized_inputs)
        return call_vmapped_twice(self.residual_norm, mixed).astype(inputs.dtype)

    def _result(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        sublayer_outputs: Float[Array, "batch suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: StateLayerBase | None,
        updated_state: StateLayerBase | None,
        return_activation_trace: bool,
    ) -> TransformerLayerResult:
        outputs = self._interpolate(inputs, sublayer_outputs)
        if return_activation_trace:
            activation_trace = NGPTLayerActivationTrace(
                inputs=inputs,
                positional_embeddings=positional_embeddings,
                state=state,
                sublayer=sublayer_outputs,
            )
        else:
            activation_trace = None
        assert outputs.dtype == inputs.dtype
        return TransformerLayerResult(
            outputs=outputs,
            updated_state=updated_state,
            activation_trace=activation_trace,
        )


def _init_residual(
    config: NGPTLayerConfigBase,
    initializer: Initializer,
    model_dim: int,
) -> tuple[Normalization, LearnedScale]:
    return (
        config.residual_norm_config.init(initializer, model_dim),
        config.residual_scale_config.init(initializer, model_dim),
    )


@dataclass(frozen=True)
class NGPTAttentionLayerConfig(NGPTLayerConfigBase):
    mixer_config: TokenMixerConfig

    @property
    def owns_state(self) -> bool:
        return True

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        hidden_dim: int,  # noqa: ARG002
    ) -> "NGPTAttentionLayer":
        residual_norm, residual_scale = _init_residual(self, initializer, model_dim)
        return NGPTAttentionLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            residual_norm=residual_norm,
            residual_scale=residual_scale,
            mixer=self.mixer_config.init(initializer, model_dim=model_dim),
        )


class NGPTAttentionLayer(NGPTLayerBase[NGPTAttentionLayerConfig]):
    mixer: TokenMixerBase

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: StateLayerBase | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_suffix_tokens: int | None = None,
        *,
        keychain: Keychain,
    ) -> TransformerLayerResult:
        if per_layer_input is not None:
            raise ValueError("nGPT layers do not support per-layer inputs.")
        if return_suffix_tokens is not None and return_activation_trace:
            raise ValueError("return_suffix_tokens cannot be combined with return_activation_trace.")

        def call_mixer(
            mixer_inputs: tuple[
                Float[Array, "suffix_tokens channels"],
                PositionalEmbeddings | None,
                StateLayerBase | None,
                Int[Array, ""] | None,
                Int[Array, " suffix_tokens"] | None,
            ],
            *,
            keychain: Keychain,
        ) -> tuple[Float[Array, "suffix_tokens channels"], StateLayerBase | None]:
            mixer_input, positional_embedding, mixer_state, length_without_padding, parent_indices = mixer_inputs
            return self.mixer(
                mixer_input,
                positional_embedding,
                mixer_state,
                return_updated_state=return_updated_state or return_activation_trace,
                length_without_padding=length_without_padding,
                forward_pass_config=forward_pass_config.mixer_forward_pass_config,
                attention_parent_indices=parent_indices,
                keychain=keychain,
            )

        mixer_outputs, updated_state = call_vmapped(
            call_mixer,
            (inputs, positional_embeddings, state, lengths_without_padding, attention_parent_indices),
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        if return_suffix_tokens is not None:
            inputs, mixer_outputs = (
                gather_suffix_tokens(array, lengths_without_padding, return_suffix_tokens, self.sharding_config)
                for array in (inputs, mixer_outputs)
            )
        return self._result(
            inputs,
            mixer_outputs,
            positional_embeddings,
            state,
            updated_state,
            return_activation_trace,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> StateLayerBase:
        return jax.tree.map(
            lambda array: jax.device_put(
                jnp.repeat(array[None, ...], batch_size, axis=0),
                self.sharding_config.resolve_sharding((LogicalAxis.BATCH, *((None,) * array.ndim))),
            ),
            self.mixer.init_static_state(capacity, dtype),
        )


@dataclass(frozen=True)
class NGPTMLPLayerConfig(NGPTLayerConfigBase):
    mlp_config: MLPConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rope_config is not None:
            raise ValueError("nGPT MLP layers do not use positional embeddings.")

    @property
    def owns_state(self) -> bool:
        return False

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        hidden_dim: int,
    ) -> "NGPTMLPLayer":
        residual_norm, residual_scale = _init_residual(self, initializer, model_dim)
        return NGPTMLPLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            residual_norm=residual_norm,
            residual_scale=residual_scale,
            mlp=self.mlp_config.init(initializer, model_dim, hidden_dim),
        )


class NGPTMLPLayer(NGPTLayerBase[NGPTMLPLayerConfig]):
    mlp: MLPBase

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: StateLayerBase | None = None,
        return_updated_state: bool = False,  # noqa: ARG002
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,  # noqa: ARG002
        return_suffix_tokens: int | None = None,
        *,
        keychain: Keychain,
    ) -> TransformerLayerResult:
        if per_layer_input is not None:
            raise ValueError("nGPT layers do not support per-layer inputs.")
        if state is not None:
            raise ValueError("nGPT MLP layers do not own a state.")
        if return_suffix_tokens is not None and return_activation_trace:
            raise ValueError("return_suffix_tokens cannot be combined with return_activation_trace.")
        if return_suffix_tokens is not None:
            inputs = gather_suffix_tokens(inputs, lengths_without_padding, return_suffix_tokens, self.sharding_config)
            # The gathered window is tail-aligned, while MLP padding masks assume the valid tokens
            # form a prefix, so the window is treated as fully valid instead.
            lengths_without_padding = None
        mlp_outputs = self.mlp(
            inputs,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=forward_pass_config.mlp_forward_pass_config,
            keychain=keychain,
        )
        return self._result(inputs, mlp_outputs, positional_embeddings, None, None, return_activation_trace)

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> StateLayerBase:  # noqa: ARG002
        raise ValueError("nGPT MLP layers do not own a state.")
