# SPDX-License-Identifier: Apache-2.0

"""Training-side expert selection for ESFT.

The inference pipeline writes expert IDs in a global, per-transformer-layer ID
space.  This module validates that artifact and applies it to the canonical
global parameter names produced by AReaL's Megatron conversion helpers.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn


def configure_esft_megatron(config: Any, *, use_esft: bool) -> Any:
    """Use the legacy ESFT optimizer mode without mutating shared FFT config."""
    if not use_esft:
        return config
    config = deepcopy(config)
    config.ddp.use_distributed_optimizer = False
    config.ddp.overlap_grad_reduce = False
    config.ddp.overlap_param_gather = False
    config.ddp.align_param_gather = False
    config.overlap_param_gather_with_optimizer_step = False
    config.use_precision_aware_optimizer = False
    return config

_EXPERT_PARAMETER_RE = re.compile(
    r"^module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\."
    r"mlp\.experts\.(.+)\.weight(\d+)$"
)


@dataclass(frozen=True)
class ESFTSelection:
    """Validated global expert selection loaded from ``esft_config.json``."""

    experts: dict[int, frozenset[int]]
    source: str

    @property
    def expert_keys(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            (layer_id, expert_id)
            for layer_id, expert_ids in self.experts.items()
            for expert_id in expert_ids
        )


@dataclass(frozen=True)
class ESFTApplyResult:
    """Local result from applying an ESFT selection to model parameters."""

    matched_experts: frozenset[tuple[int, int]]
    matched_components: frozenset[tuple[int, int, str]]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def load_esft_config(
    path: str | Path,
    *,
    num_transformer_layers: int,
    num_experts: int,
    model_architectures: Iterable[str] | None = None,
) -> ESFTSelection:
    """Load and strictly validate an inference-generated ESFT configuration."""

    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"ESFT config does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"ESFT config is not valid JSON: {config_path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError("ESFT config must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError("ESFT config schema_version must be 1")
    if num_transformer_layers <= 0:
        raise ValueError("num_transformer_layers must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    model = data.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("ESFT model metadata must be an object")
    expected_architecture = model.get("architecture")
    if expected_architecture and model_architectures is not None:
        actual_architectures = set(model_architectures)
        if expected_architecture not in actual_architectures:
            raise ValueError(
                "ESFT config architecture does not match the training model: "
                f"{expected_architecture!r} not in {sorted(actual_architectures)!r}"
            )

    raw_experts = data.get("experts")
    if not isinstance(raw_experts, dict) or not raw_experts:
        raise ValueError("ESFT config must contain a non-empty 'experts' object")

    routing_metadata = data.get("routing_metadata", {})
    if not isinstance(routing_metadata, dict):
        raise ValueError("ESFT routing_metadata must be an object")
    if routing_metadata.get("expert_id_space", "global") != "global":
        raise ValueError("ESFT expert_id_space must be 'global'")
    if routing_metadata.get("layer_id_base", 0) != 0:
        raise ValueError("ESFT layer_id_base must be 0")
    router_top_k = routing_metadata.get("router_top_k")
    if router_top_k is not None and (
        not isinstance(router_top_k, int)
        or isinstance(router_top_k, bool)
        or not 0 < router_top_k <= num_experts
    ):
        raise ValueError("routing_metadata.router_top_k must be in [1, num_experts]")
    moe_layer_ids = routing_metadata.get("moe_layer_ids")
    if moe_layer_ids is not None:
        if not isinstance(moe_layer_ids, list) or not all(
            isinstance(layer_id, int) and not isinstance(layer_id, bool)
            for layer_id in moe_layer_ids
        ):
            raise ValueError(
                "routing_metadata.moe_layer_ids must be a list of integers"
            )
        if len(set(moe_layer_ids)) != len(moe_layer_ids):
            raise ValueError(
                "routing_metadata.moe_layer_ids must not contain duplicates"
            )
        if len(moe_layer_ids) != len(raw_experts):
            raise ValueError(
                "routing_metadata.moe_layer_ids must have one entry per "
                "expert-selection layer"
            )

    expected_moe_layers = routing_metadata.get("num_moe_layers")
    if expected_moe_layers is not None and (
        not isinstance(expected_moe_layers, int)
        or isinstance(expected_moe_layers, bool)
        or expected_moe_layers != len(raw_experts)
    ):
        raise ValueError(
            "routing_metadata.num_moe_layers must equal the number of expert-selection layers"
        )
    if moe_layer_ids is None and len(raw_experts) != num_transformer_layers:
        raise ValueError(
            "routing_metadata.moe_layer_ids is required when MoE routing rows do "
            "not cover every transformer layer"
        )
    expected_num_experts = routing_metadata.get("num_experts")
    if expected_num_experts is not None:
        if not isinstance(expected_num_experts, int) or isinstance(
            expected_num_experts, bool
        ):
            raise ValueError("routing_metadata.num_experts must be an integer")
        if expected_num_experts != num_experts:
            raise ValueError(
                "ESFT config num_experts does not match the training model: "
                f"{expected_num_experts} != {num_experts}"
            )

    selection_metadata = data.get("selection", {})
    if not isinstance(selection_metadata, dict):
        raise ValueError("ESFT selection metadata must be an object")
    method = selection_metadata.get("method")
    if method is not None and method not in {
        "esft_gate",
        "esft_token",
        "reward_gate",
        "reward_token",
    }:
        raise ValueError(f"Unsupported ESFT selection method: {method!r}")
    routing_alignment = selection_metadata.get("routing_alignment")
    if routing_alignment is not None and routing_alignment != "causal_generation":
        raise ValueError("ESFT routing_alignment must be 'causal_generation'")
    expected_experts_per_layer = selection_metadata.get("experts_per_layer")
    if expected_experts_per_layer is not None and (
        not isinstance(expected_experts_per_layer, int)
        or isinstance(expected_experts_per_layer, bool)
        or expected_experts_per_layer <= 0
    ):
        raise ValueError("selection.experts_per_layer must be a positive integer")

    experts: dict[int, frozenset[int]] = {}
    for selection_layer, raw_ids in raw_experts.items():
        try:
            selection_layer_id = int(selection_layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid ESFT layer ID: {selection_layer!r}") from error
        if str(selection_layer_id) != str(selection_layer):
            raise ValueError(
                f"ESFT layer ID must be a canonical integer: {selection_layer!r}"
            )
        if selection_layer_id < 0:
            raise ValueError(f"ESFT layer ID must be nonnegative: {selection_layer_id}")
        if moe_layer_ids is not None:
            if selection_layer_id >= len(moe_layer_ids):
                raise ValueError(
                    f"ESFT selection layer {selection_layer_id} has no moe_layer_ids entry"
                )
            transformer_layer_id = moe_layer_ids[selection_layer_id]
        else:
            transformer_layer_id = selection_layer_id
        if not 0 <= transformer_layer_id < num_transformer_layers:
            raise ValueError(
                f"ESFT transformer layer {transformer_layer_id} is outside "
                f"[0, {num_transformer_layers})"
            )
        if transformer_layer_id in experts:
            raise ValueError(
                f"Multiple ESFT entries resolve to transformer layer {transformer_layer_id}"
            )
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(
                f"ESFT layer {selection_layer_id} must select a non-empty expert list"
            )
        if not all(
            isinstance(expert_id, int) and not isinstance(expert_id, bool)
            for expert_id in raw_ids
        ):
            raise ValueError(
                f"ESFT layer {selection_layer_id} expert IDs must be integers"
            )
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError(
                f"ESFT layer {selection_layer_id} contains duplicate experts"
            )
        if (
            expected_experts_per_layer is not None
            and len(raw_ids) != expected_experts_per_layer
        ):
            raise ValueError(
                f"ESFT layer {selection_layer_id} selects {len(raw_ids)} experts; "
                f"expected {expected_experts_per_layer}"
            )
        invalid = [
            expert_id for expert_id in raw_ids if not 0 <= expert_id < num_experts
        ]
        if invalid:
            raise ValueError(
                f"ESFT layer {selection_layer_id} contains out-of-range experts: {invalid}"
            )
        experts[transformer_layer_id] = frozenset(raw_ids)

    return ESFTSelection(experts=experts, source=str(config_path))


def apply_esft(
    model_parameters: Iterable[nn.Parameter],
    canonical_named_parameters: Iterable[tuple[str, Any]],
    selection: ESFTSelection,
) -> ESFTApplyResult:
    """Freeze a model and unfreeze parameters belonging to selected experts."""

    parameters = list(
        {id(parameter): parameter for parameter in model_parameters}.values()
    )
    for parameter in parameters:
        parameter.requires_grad_(False)

    matched_experts: set[tuple[int, int]] = set()
    matched_components: set[tuple[int, int, str]] = set()
    trainable_names: list[str] = []
    seen_parameter_ids: set[int] = set()
    for name, value in canonical_named_parameters:
        match = _EXPERT_PARAMETER_RE.match(name)
        if match is None or not isinstance(value, nn.Parameter):
            continue
        layer_id = int(match.group(1))
        component = match.group(2)
        expert_id = int(match.group(3))
        if expert_id not in selection.experts.get(layer_id, frozenset()):
            continue
        value.requires_grad_(True)
        matched_experts.add((layer_id, expert_id))
        matched_components.add((layer_id, expert_id, component))
        if id(value) not in seen_parameter_ids:
            trainable_names.append(name)
            seen_parameter_ids.add(id(value))

    return ESFTApplyResult(
        matched_experts=frozenset(matched_experts),
        matched_components=frozenset(matched_components),
        trainable_parameter_names=tuple(trainable_names),
        trainable_parameters=sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        total_parameters=sum(parameter.numel() for parameter in parameters),
    )
