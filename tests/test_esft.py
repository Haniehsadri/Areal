# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import torch

from areal.api.cli_args import TrainEngineConfig
from areal.utils.esft import apply_esft, load_esft_config
from areal.utils.reward_expert_selection import (
    ExpertScoreResult,
    RoutingMetadata,
    write_selection_artifacts,
)


def _write_config(tmp_path, *, experts, routing_metadata=None):
    path = tmp_path / "esft_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "routing_metadata": routing_metadata or {},
                "experts": experts,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_apply_esft_freezes_model_and_unfreezes_selected_experts(tmp_path):
    """Only parameters belonging to selected global experts remain trainable."""

    selected_fc1 = torch.nn.Parameter(torch.ones(2, 2))
    selected_fc2 = torch.nn.Parameter(torch.ones(2, 2))
    unselected_expert = torch.nn.Parameter(torch.ones(2, 2))
    attention = torch.nn.Parameter(torch.ones(2, 2))
    parameters = [selected_fc1, selected_fc2, unselected_expert, attention]
    named_parameters = [
        (
            "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight3",
            selected_fc1,
        ),
        (
            "module.module.decoder.layers.1.mlp.experts.linear_fc2.weight3",
            selected_fc2,
        ),
        (
            "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight4",
            unselected_expert,
        ),
        ("module.module.decoder.layers.1.self_attention.weight", attention),
    ]
    config_path = _write_config(
        tmp_path,
        experts={"0": [3]},
        routing_metadata={"num_moe_layers": 1, "moe_layer_ids": [1]},
    )
    selection = load_esft_config(config_path, num_transformer_layers=4, num_experts=8)

    result = apply_esft(parameters, named_parameters, selection)

    assert selected_fc1.requires_grad
    assert selected_fc2.requires_grad
    assert not unselected_expert.requires_grad
    assert not attention.requires_grad
    assert result.matched_experts == frozenset({(1, 3)})
    assert result.matched_components == frozenset(
        {(1, 3, "linear_fc1"), (1, 3, "linear_fc2")}
    )
    assert result.trainable_parameters == selected_fc1.numel() + selected_fc2.numel()


def test_load_esft_config_maps_moe_ordinals_to_transformer_layers(tmp_path):
    """moe_layer_ids maps routing rows to their real transformer-layer IDs."""

    config_path = _write_config(
        tmp_path,
        experts={"0": [1], "1": [2]},
        routing_metadata={
            "num_moe_layers": 2,
            "num_experts": 4,
            "moe_layer_ids": [2, 5],
        },
    )

    selection = load_esft_config(config_path, num_transformer_layers=6, num_experts=4)

    assert selection.experts == {2: frozenset({1}), 5: frozenset({2})}


@pytest.mark.parametrize(
    ("experts", "routing_metadata", "message"),
    [
        ({}, {}, "non-empty"),
        ({"0": [4]}, {"num_experts": 4}, "out-of-range"),
        ({"0": [1, 1]}, {}, "duplicate"),
        ({"2": [1]}, {"moe_layer_ids": [0]}, "no moe_layer_ids entry"),
        ({"0": [1]}, {"num_moe_layers": 1}, "moe_layer_ids is required"),
    ],
)
def test_load_esft_config_rejects_invalid_selection(
    tmp_path, experts, routing_metadata, message
):
    """Malformed or model-incompatible expert selections fail before training."""

    config_path = _write_config(
        tmp_path, experts=experts, routing_metadata=routing_metadata
    )

    with pytest.raises(ValueError, match=message):
        load_esft_config(config_path, num_transformer_layers=4, num_experts=4)


def test_apply_esft_supports_language_model_prefix(tmp_path):
    """Canonical VLM language-model expert names are selected correctly."""

    parameter = torch.nn.Parameter(torch.ones(1))
    config_path = _write_config(tmp_path, experts={"0": [2]})
    selection = load_esft_config(config_path, num_transformer_layers=1, num_experts=4)

    result = apply_esft(
        [parameter],
        [
            (
                "module.module.language_model.decoder.layers.0.mlp.experts."
                "linear_fc1.weight2",
                parameter,
            )
        ],
        selection,
    )

    assert parameter.requires_grad
    assert result.matched_experts == frozenset({(0, 2)})


def test_generated_selection_artifact_loads_as_training_selection(tmp_path):
    """The inference writer and training-side loader share one schema contract."""

    metadata = RoutingMetadata(
        num_moe_layers=2,
        num_experts=4,
        router_top_k=2,
        moe_layer_ids=(2, 5),
    )
    scores = np.asarray(
        [
            [0.1, 0.8, 0.2, 0.7],
            [0.9, 0.1, 0.6, 0.2],
        ],
        dtype=np.float64,
    )
    result = ExpertScoreResult(
        metadata=metadata,
        scores={
            "esft_gate": scores,
            "esft_token": scores,
            "reward_gate": scores,
            "reward_token": scores,
        },
        num_trajectories=1,
        num_valid_trajectories=1,
        num_zero_length_trajectories=0,
        num_scored_tokens=1,
        reward_mean=1.0,
        reward_std=0.0,
        total_selection_reward=1.0,
        num_positive_reward_trajectories=1,
        num_zero_reward_trajectories=0,
        num_negative_raw_reward_trajectories=0,
    )
    _, config_path = write_selection_artifacts(
        tmp_path,
        result,
        model={"architecture": "Qwen3MoeForCausalLM"},
        inference={"backend": "vllm", "backend_version": "0.23.0"},
        score_function="reward_gate",
        experts_per_layer=2,
    )

    selection = load_esft_config(
        config_path,
        num_transformer_layers=6,
        num_experts=4,
        model_architectures=["Qwen3MoeForCausalLM"],
    )

    assert selection.experts == {
        2: frozenset({1, 3}),
        5: frozenset({0, 2}),
    }


def test_train_engine_config_requires_esft_path():
    """Enabling ESFT without its generated selection file is rejected."""

    with pytest.raises(ValueError, match="esft_config is required"):
        TrainEngineConfig(use_esft=True)


def test_train_engine_config_rejects_esft_with_lora():
    """ESFT full-expert training and LoRA cannot be enabled together."""

    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainEngineConfig(
            use_esft=True,
            esft_config="selection/esft_config.json",
            use_lora=True,
        )
