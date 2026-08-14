# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from areal.utils.reward_expert_selection import (
    RoutingMetadata,
    TrajectoryRoutingRecord,
    accumulate_token_routing,
    aggregate_expert_scores,
    read_trajectory_jsonl,
    select_experts_per_layer,
    transform_selection_reward,
    write_selection_artifacts,
    write_trajectory_jsonl,
)


def _record(
    request_id: str,
    reward: float,
    gate_sum: list[list[float]],
    token_hit_count: list[list[int]],
    num_positions: int,
) -> TrajectoryRoutingRecord:
    return TrajectoryRoutingRecord(
        prompt_id="prompt-0",
        response_id=int(request_id[-1]),
        request_id=request_id,
        prompt_token_ids=[1, 2],
        response_token_ids=list(range(num_positions)),
        raw_reward=reward,
        selection_reward=reward,
        finish_reason="stop",
        num_scored_positions=num_positions,
        gate_sum=np.asarray(gate_sum, dtype=np.float64),
        token_hit_count=np.asarray(token_hit_count, dtype=np.int64),
    )


def test_accumulate_token_routing_synthetic_values_returns_exact_sums():
    """Gate weights and token hits are accumulated per layer and expert."""
    metadata = RoutingMetadata(num_moe_layers=2, num_experts=4, router_top_k=2)
    expert_ids = np.asarray(
        [
            [[0, 2], [1, 3]],
            [[0, 1], [1, 2]],
        ],
        dtype=np.int32,
    )
    expert_weights = np.asarray(
        [
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.2, 0.8], [0.9, 0.1]],
        ],
        dtype=np.float32,
    )

    gate_sum, token_hits = accumulate_token_routing(
        expert_ids, expert_weights, metadata
    )

    np.testing.assert_allclose(
        gate_sum,
        [[0.9, 0.8, 0.3, 0.0], [0.0, 1.3, 0.1, 0.6]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        token_hits,
        [[2, 1, 1, 0], [0, 2, 1, 1]],
    )


def test_aggregate_expert_scores_manual_rewards_matches_formulas():
    """Raw reward weighting matches the hand-calculated numerical example."""
    metadata = RoutingMetadata(num_moe_layers=1, num_experts=4, router_top_k=2)
    records = [
        _record("request-0", 1.0, [[0.6, 0.3, 0.1, 0.0]], [[1, 1, 0, 0]], 1),
        _record("request-1", 0.0, [[0.1, 0.1, 0.2, 0.6]], [[0, 0, 1, 1]], 1),
        _record("request-2", 0.5, [[0.2, 0.5, 0.3, 0.0]], [[1, 0, 1, 0]], 1),
    ]

    result = aggregate_expert_scores(records, metadata)

    np.testing.assert_allclose(
        result.scores["reward_gate"],
        [[0.4666666667, 0.3666666667, 0.1666666667, 0.0]],
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.scores["esft_gate"],
        [[0.3, 0.3, 0.2, 0.2]],
        rtol=1e-8,
        atol=1e-8,
    )


def test_aggregate_expert_scores_different_lengths_same_proportions_match():
    """Length normalization prevents long trajectories from dominating."""
    metadata = RoutingMetadata(num_moe_layers=1, num_experts=2, router_top_k=2)
    records = [
        _record("request-0", 1.0, [[1.5, 0.5]], [[3, 1]], 2),
        _record("request-1", 1.0, [[6.0, 2.0]], [[12, 4]], 8),
    ]

    result = aggregate_expert_scores(records, metadata)

    np.testing.assert_allclose(
        result.scores["reward_gate"], [[0.75, 0.25]], rtol=0, atol=0
    )


def test_aggregate_expert_scores_binary_rewards_adds_diagnostics():
    """Binary rewards produce success-versus-failure diagnostics."""
    metadata = RoutingMetadata(num_moe_layers=1, num_experts=2, router_top_k=1)
    records = [
        _record("request-0", 1.0, [[0.8, 0.2]], [[1, 0]], 1),
        _record("request-1", 0.0, [[0.1, 0.9]], [[0, 1]], 1),
    ]

    result = aggregate_expert_scores(records, metadata)

    np.testing.assert_allclose(
        result.scores["success_failure_gate_difference"],
        [[0.7, -0.7]],
        rtol=1e-8,
        atol=1e-8,
    )


def test_aggregate_expert_scores_zero_total_reward_raises_clear_error():
    """All-zero selection rewards fail rather than falling back to ESFT."""
    metadata = RoutingMetadata(num_moe_layers=1, num_experts=2, router_top_k=1)
    records = [_record("request-0", 0.0, [[0.5, 0.5]], [[1, 0]], 1)]

    with pytest.raises(ValueError, match="total selection reward is zero"):
        aggregate_expert_scores(records, metadata)


def test_trajectory_validate_scored_position_mismatch_raises():
    """Causal routing rows must match the number of saved response tokens."""
    metadata = RoutingMetadata(num_moe_layers=1, num_experts=2, router_top_k=1)
    record = _record("request-0", 1.0, [[0.5, 0.5]], [[1, 0]], 1)
    record.response_token_ids.append(99)

    with pytest.raises(ValueError, match="Causal routing mismatch"):
        record.validate(metadata)


def test_select_experts_per_layer_exact_tie_prefers_lower_id():
    """Expert selection is independent per layer and deterministic on ties."""
    scores = np.asarray([[0.5, 0.5, 0.7], [0.1, 0.3, 0.2]])

    selected = select_experts_per_layer(scores, experts_per_layer=2)

    assert selected == {"0": [2, 0], "1": [1, 2]}


def test_transform_selection_reward_negative_requires_explicit_transform():
    """Negative raw rewards are never silently transformed."""
    with pytest.raises(ValueError, match="Negative rewards require"):
        transform_selection_reward(-1.0)
    assert transform_selection_reward(-1.0, "clip_nonnegative") == 0.0


def test_write_artifacts_round_trip_preserves_selected_experts(tmp_path):
    """Both JSON artifacts reload with the same deterministic selection."""
    metadata = RoutingMetadata(num_moe_layers=1, num_experts=3, router_top_k=1)
    records = [_record("request-0", 1.0, [[0.1, 0.7, 0.2]], [[0, 1, 0]], 1)]
    result = aggregate_expert_scores(records, metadata)

    full_path, config_path = write_selection_artifacts(
        tmp_path,
        result,
        model={"name_or_path": "qwen-moe", "architecture": "QwenMoE"},
        inference={"backend": "vllm", "backend_version": "0.23.0"},
        score_function="reward_gate",
        experts_per_layer=2,
    )

    full = json.loads(full_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert full["selected_experts"] == {"0": [1, 2]}
    assert config["experts"] == full["selected_experts"]
    assert full["diagnostics"]["top_experts"]["reward_gate"] == {"0": [1, 2]}


def test_write_trajectory_jsonl_emits_sparse_routing(tmp_path):
    """Trajectory JSONL contains inspectable sparse routing dictionaries."""
    record = _record("request-0", 1.0, [[0.0, 0.7]], [[0, 1]], 1)
    output_path = tmp_path / "trajectories.jsonl"

    write_trajectory_jsonl(output_path, [record])

    decoded = json.loads(output_path.read_text(encoding="utf-8"))
    assert decoded["routing"]["gate_sum"] == {"0": {"1": 0.7}}
    assert decoded["routing"]["token_hit_count"] == {"0": {"1": 1}}

    metadata = RoutingMetadata(num_moe_layers=1, num_experts=2, router_top_k=1)
    loaded = read_trajectory_jsonl(output_path, metadata)
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0].gate_sum, record.gate_sum, rtol=0, atol=0)
    np.testing.assert_array_equal(loaded[0].token_hit_count, record.token_hit_count)
