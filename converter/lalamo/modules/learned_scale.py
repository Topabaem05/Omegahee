from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule

__all__ = [
    "LearnedScale",
    "LearnedScaleConfig",
    "ScaleTransform",
]


class ScaleTransform(StrEnum):
    IDENTITY = "identity"
    SQUARE = "square"
    ABSOLUTE = "absolute"


@dataclass(frozen=True)
class LearnedScaleConfig(LalamoConfig):
    """A per-channel scale computed as `transform(values * multiplier)` in float32."""

    multiplier: float
    transform: ScaleTransform

    def init(self, initializer: Initializer, num_channels: int) -> "LearnedScale":
        return LearnedScale(
            config=self,
            sharding_config=initializer.sharding_config,
            values=initializer.ones((num_channels,), dtype=jnp.float32),
        )


class LearnedScale(LalamoModule[LearnedScaleConfig]):
    values: Float[Array, " channels"]

    @property
    def num_channels(self) -> int:
        (result,) = self.values.shape
        return result

    @eqx.filter_jit
    def effective_scale(self) -> Float[Array, " channels"]:
        scaled_values = self.values.astype(jnp.float32) * self.config.multiplier
        match self.config.transform:
            case ScaleTransform.IDENTITY:
                return scaled_values
            case ScaleTransform.SQUARE:
                return jnp.square(scaled_values)
            case ScaleTransform.ABSOLUTE:
                return jnp.abs(scaled_values)

    def __call__(self, inputs: Float[Array, "*tokens channels"]) -> Float[Array, "*tokens channels"]:
        return (inputs.astype(jnp.float32) * self.effective_scale()).astype(inputs.dtype)
