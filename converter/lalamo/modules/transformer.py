from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, field

from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixer import State, StateLayerBase
from .transformer_layer import (
    TransformerForwardPassConfig,
    TransformerLayer,
    TransformerLayerBase,
    TransformerLayerConfigBase,
    TransformerLayerResult,
)
from .utils import call_vmapped, call_vmapped_twice, gather_suffix_tokens

__all__ = [
    "ExecutionStep",
    "RecurrenceConfig",
    "RecurrentKVCacheMode",
    "Transformer",
    "TransformerConfig",
    "TransformerForwardPassConfig",
    "TransformerResult",
]


class TransformerResult(Exportable, eqx.Module):
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    layer_results: tuple[TransformerLayerResult, ...] | None = None
    rope_embeddings: tuple[PositionalEmbeddings, ...] | None = None
    pre_norm_outputs: Float[Array, "batch suffix_tokens channels"] | None = None


class RecurrentKVCacheMode(StrEnum):
    # Every loop owns a separate cache for each recurrent layer.
    PER_LOOP = "per_loop"
    # All loops share one cache per recurrent layer: every loop reads the cache as it was before the
    # current forward pass, and only the last loop's keys and values are kept for the current tokens.
    LAST_LOOP = "last_loop"


@dataclass(frozen=True)
class RecurrenceConfig(LalamoConfig):
    """Runs layers [start_layer_index, end_layer_index) `num_loops` times with shared weights.

    With `h` the hidden state before the block, each loop starts from `norm(state + h)`
    (`state = h` for the first loop) and ends with `state = norm(outputs)`; the block returns `state`.
    """

    start_layer_index: int
    end_layer_index: int
    num_loops: int
    kv_cache_mode: RecurrentKVCacheMode
    reinjection_norm_config: NormalizationConfig

    def __post_init__(self) -> None:
        if not 0 <= self.start_layer_index < self.end_layer_index:
            raise ValueError(
                f"Invalid recurrent block [{self.start_layer_index}, {self.end_layer_index}).",
            )
        if self.num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {self.num_loops}.")

    def contains(self, layer_index: int) -> bool:
        return self.start_layer_index <= layer_index < self.end_layer_index


class ExecutionStep(NamedTuple):
    layer_index: int
    loop_index: int


class StateSlot(NamedTuple):
    layer_index: int
    loop_index: int


@dataclass(frozen=True)
class TransformerConfig(LalamoConfig):
    layer_configs: tuple[TransformerLayerConfigBase, ...]
    output_norm_config: NormalizationConfig | None
    model_dim: int
    hidden_dim: int
    recurrence: RecurrenceConfig | None = None

    def __post_init__(self) -> None:
        if self.recurrence is None:
            return
        if self.recurrence.end_layer_index > len(self.layer_configs):
            raise ValueError(
                f"The recurrent block ends at layer {self.recurrence.end_layer_index},"
                f" but the transformer has {len(self.layer_configs)} layers.",
            )
        if any(layer_config.kv_source_layer_index is not None for layer_config in self.layer_configs):
            raise ValueError("KV sharing is not supported together with recurrence.")

    @property
    def execution_steps(self) -> tuple[ExecutionStep, ...]:
        if self.recurrence is None:
            return tuple(ExecutionStep(layer_index, 0) for layer_index in range(len(self.layer_configs)))
        start, end = self.recurrence.start_layer_index, self.recurrence.end_layer_index
        return (
            *(ExecutionStep(layer_index, 0) for layer_index in range(start)),
            *(
                ExecutionStep(layer_index, loop_index)
                for loop_index in range(self.recurrence.num_loops)
                for layer_index in range(start, end)
            ),
            *(ExecutionStep(layer_index, 0) for layer_index in range(end, len(self.layer_configs))),
        )

    def state_slot(self, step: ExecutionStep) -> StateSlot:
        recurrence = self.recurrence
        has_own_loop_state = (
            recurrence is not None
            and recurrence.kv_cache_mode == RecurrentKVCacheMode.PER_LOOP
            and recurrence.contains(step.layer_index)
        )
        if has_own_loop_state:
            return StateSlot(step.layer_index, step.loop_index)
        return StateSlot(step.layer_index, 0)

    @property
    def state_slots(self) -> tuple[StateSlot, ...]:
        slots = (
            self.state_slot(step) for step in self.execution_steps if self.layer_configs[step.layer_index].owns_state
        )
        return tuple(dict.fromkeys(slots))

    def _init_ropes(self, initializer: Initializer) -> tuple[tuple[RoPE, ...], tuple[int, ...]]:
        rope_cache: dict[RoPEConfig, int] = {}
        ropes: list[RoPE] = []
        rope_indices: list[int] = []
        for layer_config in self.layer_configs:
            rope_config = layer_config.rope_config
            if rope_config is None:
                rope_indices.append(-1)
                continue

            if rope_config not in rope_cache:
                rope_cache[rope_config] = len(ropes)
                ropes.append(rope_config.init(initializer))
            rope_indices.append(rope_cache[rope_config])
        return tuple(ropes), tuple(rope_indices)

    def init(self, initializer: Initializer) -> "Transformer":
        ropes, rope_indices = self._init_ropes(initializer)

        layers = tuple(
            layer_config.init(
                initializer,
                model_dim=self.model_dim,
                hidden_dim=layer_config.hidden_dim if layer_config.hidden_dim is not None else self.hidden_dim,
            )
            for layer_config in self.layer_configs
        )
        if self.output_norm_config is not None:
            output_norm = self.output_norm_config.init(initializer, self.model_dim)
        else:
            output_norm = None
        if self.recurrence is not None:
            reinjection_norm = self.recurrence.reinjection_norm_config.init(initializer, self.model_dim)
        else:
            reinjection_norm = None

        return Transformer(
            config=self,
            sharding_config=initializer.sharding_config,
            ropes=ropes,
            rope_indices=rope_indices,
            layers=layers,
            output_norm=output_norm,
            reinjection_norm=reinjection_norm,
        )


class Transformer(LalamoModule[TransformerConfig]):
    ropes: tuple[RoPE, ...]
    rope_indices: tuple[int, ...] = field(static=True)
    layers: tuple[TransformerLayerBase, ...]
    output_norm: Normalization | None
    reinjection_norm: Normalization | None = None

    @property
    def state_slots(self) -> tuple[StateSlot, ...]:
        return self.config.state_slots

    @property
    def standard_layers(self) -> tuple[TransformerLayer, ...]:
        result = tuple(layer for layer in self.layers if isinstance(layer, TransformerLayer))
        if len(result) != len(self.layers):
            raise TypeError("The transformer contains layers other than TransformerLayer.")
        return result

    def _state_owner_step_indices(self, steps: tuple[ExecutionStep, ...]) -> tuple[int, ...]:
        return tuple(
            step_index
            for step_index, step in enumerate(steps)
            if self.config.layer_configs[step.layer_index].owns_state
        )

    @eqx.filter_jit
    def __call__(
        self,
        inner_features: Float[Array, "batch suffix_tokens channels"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None,
        return_updated_state: bool,
        return_layer_results: bool,
        return_positional_embeddings: bool,
        lengths_without_padding: Int[Array, " batch"] | None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        per_layer_inputs: tuple[Float[Array, "batch suffix_tokens ple_dim"], ...] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_suffix_tokens: int | None = None,
        *,
        keychain: Keychain,
    ) -> TransformerResult:
        if inner_features.ndim != 3:
            raise ValueError(
                "inner_features must be a 3D array of size (batch_size, sequence_length, hidden_dim),"
                f" got {inner_features.shape}",
            )
        if token_positions.ndim != 2:
            raise ValueError(
                "token_positions must be a 2D array of size (batch_size, sequence_length),"
                f" got {token_positions.shape}",
            )
        steps = self.config.execution_steps
        state_owner_step_indices = self._state_owner_step_indices(steps)
        if return_suffix_tokens is not None:
            if return_layer_results or return_positional_embeddings:
                raise ValueError(
                    "return_suffix_tokens cannot be combined with return_layer_results"
                    " or return_positional_embeddings.",
                )
            if attention_parent_indices is not None:
                raise ValueError("return_suffix_tokens cannot be combined with attention_parent_indices.")
            if not state_owner_step_indices:
                raise ValueError("return_suffix_tokens requires at least one layer that owns its state.")
            recurrence = self.config.recurrence
            if recurrence is not None and steps[state_owner_step_indices[-1]].layer_index < recurrence.end_layer_index:
                raise ValueError("return_suffix_tokens requires a state-owning layer after the recurrent block.")
            has_trailing_shared_layers = any(
                self.config.layer_configs[step.layer_index].kv_source_layer_index is not None
                for step in steps[state_owner_step_indices[-1] + 1 :]
            )
            if (
                return_suffix_tokens > 1
                and has_trailing_shared_layers
                and state is None
                and lengths_without_padding is not None
            ):
                # Dynamic KV caches anchor padded-row masks at the physical tail of the cache,
                # which only matches the gathered suffix positions for a single-token suffix.
                raise ValueError(
                    "return_suffix_tokens > 1 on padded batches requires a preallocated state"
                    " when the model has trailing KV-sharing layers.",
                )
        state_by_slot = dict(zip(self.state_slots, state, strict=True)) if state is not None else {}
        mixer_forward_pass_config = forward_pass_config.mixer_forward_pass_config
        rope_embeddings = tuple(
            call_vmapped(
                rope,
                token_positions,
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
            ).astype(
                mixer_forward_pass_config.rope_dtype,
            )
            for rope in self.ropes
        )
        has_kv_sharing = any(
            layer_config.kv_source_layer_index is not None for layer_config in self.config.layer_configs
        )
        must_return_state = return_updated_state or has_kv_sharing

        # In suffix mode, every step past the last state-owning one only affects the returned
        # suffix outputs, so those steps run on the gathered suffix instead of the full sequence.
        if return_suffix_tokens is not None:
            last_state_owner_step_index = state_owner_step_indices[-1]
            suffix_token_positions = gather_suffix_tokens(
                token_positions,
                lengths_without_padding,
                return_suffix_tokens,
                self.sharding_config,
            )
            suffix_rope_embeddings = tuple(
                call_vmapped(
                    rope,
                    suffix_token_positions,
                    added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
                ).astype(
                    mixer_forward_pass_config.rope_dtype,
                )
                for rope in self.ropes
            )
        else:
            last_state_owner_step_index = None
            suffix_rope_embeddings = ()

        recurrence = self.config.recurrence
        residual_dtype = inner_features.dtype
        step_keychains = keychain.split(len(steps))
        updated_states: dict[StateSlot, StateLayerBase | None] = {}
        layer_results = []
        recurrent_inputs = None
        loop_state = None

        for step_index, (step, step_keychain) in enumerate(zip(steps, step_keychains, strict=True)):
            assert inner_features.dtype == residual_dtype
            layer_index, loop_index = step
            layer = self.layers[layer_index]
            layer_config = self.config.layer_configs[layer_index]

            if recurrence is not None and layer_index == recurrence.start_layer_index:
                assert self.reinjection_norm is not None
                if loop_index == 0:
                    recurrent_inputs = inner_features
                    loop_state = inner_features
                assert recurrent_inputs is not None and loop_state is not None
                inner_features = call_vmapped_twice(self.reinjection_norm, loop_state + recurrent_inputs)

            runs_on_suffix_only = last_state_owner_step_index is not None and step_index > last_state_owner_step_index
            rope_index = self.rope_indices[layer_index]
            active_rope_embeddings = suffix_rope_embeddings if runs_on_suffix_only else rope_embeddings
            positional_embeddings = active_rope_embeddings[rope_index] if rope_index >= 0 else None

            per_layer_input = per_layer_inputs[layer_index] if per_layer_inputs is not None else None
            if runs_on_suffix_only and per_layer_input is not None:
                assert return_suffix_tokens is not None
                per_layer_input = gather_suffix_tokens(
                    per_layer_input,
                    lengths_without_padding,
                    return_suffix_tokens,
                    self.sharding_config,
                )

            slot = self.config.state_slot(step)
            kv_source_layer_index = layer_config.kv_source_layer_index
            if not layer_config.owns_state and kv_source_layer_index is None:
                effective_state = None
            elif kv_source_layer_index is None:
                # Under last-loop caching every loop reads the cache from before this forward pass.
                effective_state = state_by_slot.get(slot)
            else:
                source_slot = StateSlot(kv_source_layer_index, 0)
                effective_state = updated_states.get(source_slot, state_by_slot.get(source_slot))

            layer_result = layer(
                inner_features,
                positional_embeddings,
                state=effective_state,
                return_updated_state=must_return_state,
                return_activation_trace=return_layer_results,
                lengths_without_padding=None if runs_on_suffix_only else lengths_without_padding,
                forward_pass_config=forward_pass_config,
                per_layer_input=per_layer_input,
                attention_parent_indices=attention_parent_indices,
                return_suffix_tokens=return_suffix_tokens if step_index == last_state_owner_step_index else None,
                keychain=step_keychain,
            )

            inner_features = layer_result.outputs
            layer_results.append(layer_result)

            if layer_config.owns_state:
                updated_states[slot] = layer_result.updated_state

            if recurrence is not None and layer_index == recurrence.end_layer_index - 1:
                assert self.reinjection_norm is not None
                loop_state = call_vmapped_twice(self.reinjection_norm, inner_features)
                if loop_index == recurrence.num_loops - 1:
                    inner_features = loop_state

        assert inner_features.dtype == residual_dtype
        pre_norm_outputs = inner_features if return_suffix_tokens is not None else None
        if self.output_norm is not None:
            normalized_outputs = call_vmapped_twice(self.output_norm, inner_features)
        else:
            normalized_outputs = inner_features

        if return_updated_state:
            compact_state_layers = []
            for slot in self.state_slots:
                slot_state = updated_states[slot]
                if slot_state is None:
                    raise ValueError(f"State slot {slot} did not return an updated state.")
                compact_state_layers.append(slot_state)
            compact_state = State(tuple(compact_state_layers))
        else:
            compact_state = None
        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=compact_state,
            layer_results=tuple(layer_results) if return_layer_results else None,
            rope_embeddings=rope_embeddings if return_positional_embeddings else None,
            pre_norm_outputs=pre_norm_outputs,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        return State(
            self.layers[slot.layer_index].init_static_state(batch_size, capacity, dtype) for slot in self.state_slots
        )
