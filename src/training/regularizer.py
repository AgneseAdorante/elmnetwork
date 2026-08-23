# regularizer.py

from abc import ABC, abstractmethod
from typing import Any

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp

MONITOR_REQUIREMENTS = {
    "neuron_mlp_magnitude": (
        [],
        [
            "mlp",
        ],
    ),
    "population_activity": (
        [
            "activity",
        ],
        [],
    ),
    # Weight-based targets, they require no monitoring 
    "connectivity_weights": ([], []),
    "ff_weights": ([], []),
}
REQUIRED_REGULARIZER_KEYS = {"layers", "strength", "proportional"}

# Targets computed from model weights
WEIGHT_REGULARIZERS = frozenset({"connectivity_weights", "ff_weights"})


class Regularizer(eqx.Module, ABC):
    @property
    @abstractmethod
    def monitor(self) -> Any: ...

    @abstractmethod
    def __call__(self, recording, model=None): ...

    @property
    def requires_model(self) -> bool:
        return False


class ELMNetworkRegularizer(Regularizer):
    config: dict = eqx.field(static=True)

    def __init__(self, config: dict):
        self.config = config

        # simple sanity checks
        for reg_type, cfg in self.config.items():
            assert reg_type in MONITOR_REQUIREMENTS
            assert set(cfg.keys()) == REQUIRED_REGULARIZER_KEYS
            assert isinstance(cfg["layers"], (list, tuple))

    @property
    def requires_model(self) -> bool:
        return any(reg_type in WEIGHT_REGULARIZERS for reg_type in self.config)

    @property
    def monitor(self) -> tuple[list[str], list[str]]:
        layer_monitor, ensemble_monitor = [], []

        for reg_type in self.config:
            layer_vars, ensemble_vars = MONITOR_REQUIREMENTS[reg_type]

            # add recording requirements without duplication
            layer_monitor = list(dict.fromkeys(layer_monitor + layer_vars))
            ensemble_monitor = list(dict.fromkeys(ensemble_monitor + ensemble_vars))

        return layer_monitor, ensemble_monitor

    def __call__(self, recording, model=None):

        if self.requires_model and model is None:
            raise ValueError(
                "This regularizer config contains a weight-based target "
                f"({sorted(set(self.config) & WEIGHT_REGULARIZERS)}) and so "
                "requires `model=` to be passed to __call__."
            )

        loss = 0.0

        for reg_type, cfg in self.config.items():
            strength = cfg["strength"]
            proportional = cfg["proportional"]

            for layer in cfg["layers"]:
                
                if reg_type in WEIGHT_REGULARIZERS:
                    # weight reg needs no recording
                    num_neuron = model.layers[layer].num_neuron
                else:
                    # var = [time, neurons, ...]
                    layer_recording = recording[layer]
                    num_neuron = layer_recording[0]["activity"].shape[1]

                if reg_type == "neuron_mlp_magnitude":
                    # encourage small per-neuron MLP outputs
                    var = layer_recording[1]["mlp"]
                    axes = tuple(i for i in range(var.ndim) if i != 1)
                    layer_loss = jnp.mean(jnp.mean(jnp.abs(var), axis=axes) ** 2)

                elif reg_type == "population_activity":
                    # encourage sparse neuron activity
                    var = layer_recording[0]["activity"]
                    layer_loss = jnp.mean(jnn.relu(var))

                elif reg_type == "connectivity_weights":
                    # encourage sparse synapse weights

                    var = model.layers[layer].effective_weights()
                    layer_loss = jnp.mean(jnp.abs(var))

                elif reg_type == "ff_weights":
                    # encourgae space ff synapse weights
                    var = model.layers[layer].feedforward_weights()
                    layer_loss = jnp.mean(jnp.abs(var))

                else:
                    raise NotImplementedError

                if proportional:
                    layer_loss = layer_loss * num_neuron

                loss = loss + strength * layer_loss

        return loss
