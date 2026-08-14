# SPDX-License-Identifier: Apache-2.0

"""Reward-weighted selection of routed MoE experts.

This module is intentionally independent of inference backends. vLLM, vLLM Ascend,
or tests can provide compact trajectory routing summaries, which are validated and
aggregated here into ordinary and reward-weighted ESFT scores.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

ScoreFunction = Literal["esft_gate", "esft_token", "reward_gate", "reward_token"]
RewardTransform = Literal["identity", "clip_nonnegative"]


@dataclass(frozen=True)
class RoutingMetadata:
    """Shape and identity contract for trajectory routing summaries."""

    num_moe_layers: int
    num_experts: int
    router_top_k: int
    expert_id_space: Literal["global"] = "global"
    layer_id_base: int = 0
    moe_layer_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_moe_layers <= 0:
            raise ValueError("num_moe_layers must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 0 < self.router_top_k <= self.num_experts:
            raise ValueError("router_top_k must be in [1, num_experts]")
        if self.expert_id_space != "global":
            raise ValueError("Only global expert IDs are supported")
        if self.moe_layer_ids is not None:
            if len(self.moe_layer_ids) != self.num_moe_layers:
                raise ValueError("moe_layer_ids length must equal num_moe_layers")
            if len(set(self.moe_layer_ids)) != len(self.moe_layer_ids):
                raise ValueError("moe_layer_ids must not contain duplicates")


@dataclass
class TrajectoryRoutingRecord:
    """Compact routing and reward data for one generated trajectory."""

    prompt_id: str
    response_id: int
    request_id: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    raw_reward: float
    selection_reward: float
    finish_reason: str
    num_scored_positions: int
    gate_sum: np.ndarray
    token_hit_count: np.ndarray
    routing_alignment: Literal["causal_generation"] = "causal_generation"
    schema_version: int = 1

    def validate(self, metadata: RoutingMetadata) -> None:
        """Validate reward, alignment, shapes, and routing values."""

        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.routing_alignment != "causal_generation":
            raise ValueError("routing_alignment must be causal_generation")
        if not math.isfinite(self.raw_reward):
            raise ValueError(f"raw_reward must be finite for request {self.request_id}")
        if not math.isfinite(self.selection_reward):
            raise ValueError(
                f"selection_reward must be finite for request {self.request_id}"
            )
        if self.selection_reward < 0:
            raise ValueError(
                f"selection_reward must be nonnegative for request {self.request_id}"
            )
        if self.num_scored_positions < 0:
            raise ValueError("num_scored_positions must not be negative")
        if self.num_scored_positions != len(self.response_token_ids):
            raise ValueError(
                "Causal routing mismatch for request "
                f"{self.request_id}: num_scored_positions="
                f"{self.num_scored_positions}, response_tokens="
                f"{len(self.response_token_ids)}"
            )
        expected_shape = (metadata.num_moe_layers, metadata.num_experts)
        if self.gate_sum.shape != expected_shape:
            raise ValueError(
                f"gate_sum shape {self.gate_sum.shape} != {expected_shape}"
            )
        if self.token_hit_count.shape != expected_shape:
            raise ValueError(
                "token_hit_count shape "
                f"{self.token_hit_count.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(self.gate_sum)):
            raise ValueError("gate_sum must contain only finite values")
        if np.any(self.gate_sum < 0):
            raise ValueError("gate_sum must not contain negative values")
        if not np.issubdtype(self.token_hit_count.dtype, np.integer):
            raise ValueError("token_hit_count must use an integer dtype")
        if np.any(self.token_hit_count < 0):
            raise ValueError("token_hit_count must not contain negative values")
        expected_hits = self.num_scored_positions * metadata.router_top_k
        layer_hits = self.token_hit_count.sum(axis=1)
        if not np.all(layer_hits == expected_hits):
            raise ValueError(
                "Each MoE layer must contain num_scored_positions * router_top_k "
                f"hits; expected {expected_hits}, got {layer_hits.tolist()}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a sparse, human-inspectable trajectory representation."""

        def sparse_rows(values: np.ndarray) -> dict[str, dict[str, float | int]]:
            rows: dict[str, dict[str, float | int]] = {}
            for layer_id, row in enumerate(values):
                nonzero = np.flatnonzero(row)
                if nonzero.size:
                    rows[str(layer_id)] = {
                        str(int(expert_id)): (
                            int(row[expert_id])
                            if np.issubdtype(values.dtype, np.integer)
                            else float(row[expert_id])
                        )
                        for expert_id in nonzero
                    }
            return rows

        return {
            "schema_version": self.schema_version,
            "prompt_id": self.prompt_id,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "prompt_token_ids": self.prompt_token_ids,
            "response_token_ids": self.response_token_ids,
            "raw_reward": self.raw_reward,
            "selection_reward": self.selection_reward,
            "finish_reason": self.finish_reason,
            "routing_alignment": self.routing_alignment,
            "num_scored_positions": self.num_scored_positions,
            "routing": {
                "gate_sum": sparse_rows(self.gate_sum),
                "token_hit_count": sparse_rows(self.token_hit_count),
            },
        }

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, Any],
        metadata: RoutingMetadata,
    ) -> TrajectoryRoutingRecord:
        """Load a sparse trajectory JSON object and validate its schema."""

        routing = value.get("routing")
        if not isinstance(routing, Mapping):
            raise ValueError("Trajectory record is missing routing data")

        def dense(name: str, dtype: np.dtype) -> np.ndarray:
            output = np.zeros(
                (metadata.num_moe_layers, metadata.num_experts), dtype=dtype
            )
            rows = routing.get(name, {})
            if not isinstance(rows, Mapping):
                raise ValueError(f"routing.{name} must be an object")
            for layer_key, experts in rows.items():
                layer_id = int(layer_key)
                if not 0 <= layer_id < metadata.num_moe_layers:
                    raise ValueError(f"Invalid layer ID {layer_id} in {name}")
                if not isinstance(experts, Mapping):
                    raise ValueError(f"routing.{name}.{layer_key} must be an object")
                for expert_key, score in experts.items():
                    expert_id = int(expert_key)
                    if not 0 <= expert_id < metadata.num_experts:
                        raise ValueError(f"Invalid expert ID {expert_id} in {name}")
                    output[layer_id, expert_id] = score
            return output

        record = cls(
            schema_version=int(value.get("schema_version", 1)),
            prompt_id=str(value["prompt_id"]),
            response_id=int(value["response_id"]),
            request_id=str(value["request_id"]),
            prompt_token_ids=[int(token) for token in value["prompt_token_ids"]],
            response_token_ids=[int(token) for token in value["response_token_ids"]],
            raw_reward=float(value["raw_reward"]),
            selection_reward=float(value["selection_reward"]),
            finish_reason=str(value["finish_reason"]),
            routing_alignment=value.get("routing_alignment", "causal_generation"),
            num_scored_positions=int(value["num_scored_positions"]),
            gate_sum=dense("gate_sum", np.dtype(np.float64)),
            token_hit_count=dense("token_hit_count", np.dtype(np.int64)),
        )
        record.validate(metadata)
        return record


@dataclass(frozen=True)
class ExpertScoreResult:
    """All expert scores and aggregate calibration statistics."""

    metadata: RoutingMetadata
    scores: Mapping[str, np.ndarray]
    num_trajectories: int
    num_valid_trajectories: int
    num_zero_length_trajectories: int
    num_scored_tokens: int
    reward_mean: float
    reward_std: float
    total_selection_reward: float
    num_positive_reward_trajectories: int
    num_zero_reward_trajectories: int
    num_negative_raw_reward_trajectories: int


def transform_selection_reward(
    raw_reward: float,
    transform: RewardTransform | Callable[[float], float] = "identity",
) -> float:
    """Apply an explicit reward transform and validate its output."""

    if not math.isfinite(raw_reward):
        raise ValueError("raw_reward must be finite")
    if transform == "identity":
        selection_reward = raw_reward
    elif transform == "clip_nonnegative":
        selection_reward = max(raw_reward, 0.0)
    elif callable(transform):
        selection_reward = float(transform(raw_reward))
    else:
        raise ValueError(f"Unknown reward transform: {transform}")
    if not math.isfinite(selection_reward):
        raise ValueError("selection reward transform returned a non-finite value")
    if selection_reward < 0:
        raise ValueError(
            "Negative rewards require an explicit transform that produces "
            "nonnegative selection rewards"
        )
    return selection_reward


def accumulate_token_routing(
    expert_ids: np.ndarray,
    expert_weights: np.ndarray,
    metadata: RoutingMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate aligned token routing into per-layer expert statistics.

    Both inputs have shape ``[num_positions, num_moe_layers, router_top_k]``.
    """

    expected_tail = (metadata.num_moe_layers, metadata.router_top_k)
    if expert_ids.ndim != 3 or expert_ids.shape[1:] != expected_tail:
        raise ValueError(
            f"expert_ids shape must be [positions, {expected_tail[0]}, "
            f"{expected_tail[1]}], got {expert_ids.shape}"
        )
    if expert_weights.shape != expert_ids.shape:
        raise ValueError("expert_weights shape must match expert_ids")
    if not np.issubdtype(expert_ids.dtype, np.integer):
        raise ValueError("expert_ids must use an integer dtype")
    if expert_ids.size and (
        int(expert_ids.min()) < 0 or int(expert_ids.max()) >= metadata.num_experts
    ):
        raise ValueError("expert_ids contain values outside the global expert range")
    if metadata.router_top_k > 1 and np.any(
        np.diff(np.sort(expert_ids, axis=2), axis=2) == 0
    ):
        raise ValueError("A token cannot select the same expert more than once")
    if not np.all(np.isfinite(expert_weights)):
        raise ValueError("expert_weights must contain only finite values")
    if np.any(expert_weights < 0):
        raise ValueError("expert_weights must not contain negative values")

    gate_sum = np.zeros(
        (metadata.num_moe_layers, metadata.num_experts), dtype=np.float64
    )
    token_hit_count = np.zeros_like(gate_sum, dtype=np.int64)
    for layer_id in range(metadata.num_moe_layers):
        ids = expert_ids[:, layer_id, :].reshape(-1)
        weights = expert_weights[:, layer_id, :].reshape(-1)
        np.add.at(gate_sum[layer_id], ids, weights)
        np.add.at(token_hit_count[layer_id], ids, 1)
    return gate_sum, token_hit_count


def trajectory_record_from_routing(
    *,
    prompt_id: str,
    response_id: int,
    request_id: str,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    raw_reward: float,
    finish_reason: str,
    expert_ids: np.ndarray,
    expert_weights: np.ndarray,
    metadata: RoutingMetadata,
    reward_transform: RewardTransform | Callable[[float], float] = "identity",
) -> TrajectoryRoutingRecord:
    """Build a validated trajectory record from causal backend routing arrays."""

    if expert_ids.shape[0] != len(response_token_ids):
        raise ValueError(
            "Causal routing mismatch: backend returned "
            f"{expert_ids.shape[0]} positions for {len(response_token_ids)} "
            f"response tokens in request {request_id}"
        )
    if (
        metadata.moe_layer_ids is not None
        and expert_ids.shape[1] != metadata.num_moe_layers
    ):
        layer_indices = list(metadata.moe_layer_ids)
        if max(layer_indices) >= expert_ids.shape[1]:
            raise ValueError("moe_layer_ids reference unavailable routing layers")
        expert_ids = expert_ids[:, layer_indices, :]
        expert_weights = expert_weights[:, layer_indices, :]
    gate_sum, token_hit_count = accumulate_token_routing(
        expert_ids, expert_weights, metadata
    )
    record = TrajectoryRoutingRecord(
        prompt_id=prompt_id,
        response_id=response_id,
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        raw_reward=float(raw_reward),
        selection_reward=transform_selection_reward(raw_reward, reward_transform),
        finish_reason=finish_reason,
        num_scored_positions=expert_ids.shape[0],
        gate_sum=gate_sum,
        token_hit_count=token_hit_count,
    )
    record.validate(metadata)
    return record


def aggregate_expert_scores(
    records: Iterable[TrajectoryRoutingRecord],
    metadata: RoutingMetadata,
) -> ExpertScoreResult:
    """Calculate ordinary and raw-reward-weighted Gate and Token scores."""

    all_records = list(records)
    for record in all_records:
        record.validate(metadata)
    valid = [record for record in all_records if record.num_scored_positions > 0]
    if not valid:
        raise ValueError(
            f"No valid trajectories with scored positions (total={len(all_records)})"
        )

    gate = np.stack([record.gate_sum / record.num_scored_positions for record in valid])
    token = np.stack(
        [
            record.token_hit_count
            / (record.num_scored_positions * metadata.router_top_k)
            for record in valid
        ]
    )
    selection_rewards = np.asarray(
        [record.selection_reward for record in valid], dtype=np.float64
    )
    raw_rewards = np.asarray([record.raw_reward for record in valid], dtype=np.float64)
    total_selection_reward = float(selection_rewards.sum())
    if total_selection_reward <= 0:
        unique, counts = np.unique(selection_rewards, return_counts=True)
        distribution = {
            str(float(value)): int(count) for value, count in zip(unique, counts)
        }
        raise ValueError(
            "Reward-weighted selection is undefined because total selection reward "
            f"is zero: trajectories={len(all_records)}, valid={len(valid)}, "
            f"positive=0, distribution={distribution}"
        )

    scores: dict[str, np.ndarray] = {
        "esft_gate": gate.mean(axis=0),
        "esft_token": token.mean(axis=0),
        "reward_gate": np.tensordot(selection_rewards, gate, axes=(0, 0))
        / total_selection_reward,
        "reward_token": np.tensordot(selection_rewards, token, axes=(0, 0))
        / total_selection_reward,
    }

    binary = np.all(np.isin(selection_rewards, (0.0, 1.0)))
    if binary:
        success_mask = selection_rewards == 1.0
        failure_mask = selection_rewards == 0.0
        for name, values in (("gate", gate), ("token", token)):
            successful = (
                values[success_mask].mean(axis=0)
                if np.any(success_mask)
                else np.zeros(values.shape[1:], dtype=np.float64)
            )
            failed = (
                values[failure_mask].mean(axis=0)
                if np.any(failure_mask)
                else np.zeros(values.shape[1:], dtype=np.float64)
            )
            scores[f"successful_{name}"] = successful
            scores[f"failed_{name}"] = failed
            scores[f"success_failure_{name}_difference"] = successful - failed

    return ExpertScoreResult(
        metadata=metadata,
        scores=scores,
        num_trajectories=len(all_records),
        num_valid_trajectories=len(valid),
        num_zero_length_trajectories=len(all_records) - len(valid),
        num_scored_tokens=sum(record.num_scored_positions for record in valid),
        reward_mean=float(raw_rewards.mean()),
        reward_std=float(raw_rewards.std()),
        total_selection_reward=total_selection_reward,
        num_positive_reward_trajectories=int(np.count_nonzero(selection_rewards > 0)),
        num_zero_reward_trajectories=int(np.count_nonzero(selection_rewards == 0)),
        num_negative_raw_reward_trajectories=int(np.count_nonzero(raw_rewards < 0)),
    )


def select_experts_per_layer(
    scores: np.ndarray,
    experts_per_layer: int,
) -> dict[str, list[int]]:
    """Select a fixed number of experts per layer with deterministic ties."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [layers, experts]")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values")
    num_experts = scores.shape[1]
    if not 0 < experts_per_layer <= num_experts:
        raise ValueError("experts_per_layer must be in [1, num_experts]")
    expert_ids = np.arange(num_experts)
    return {
        str(layer_id): np.lexsort((expert_ids, -layer_scores))[
            :experts_per_layer
        ].tolist()
        for layer_id, layer_scores in enumerate(scores)
    }


def write_trajectory_jsonl(
    path: str | Path, records: Iterable[TrajectoryRoutingRecord]
) -> None:
    """Write inspectable trajectory records as JSON Lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")


def read_trajectory_jsonl(
    path: str | Path, metadata: RoutingMetadata
) -> list[TrajectoryRoutingRecord]:
    """Read and validate sparse trajectory records from JSON Lines."""

    records: list[TrajectoryRoutingRecord] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(TrajectoryRoutingRecord.from_json_dict(value, metadata))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid trajectory record at line {line_number}: {error}"
                ) from error
    return records


def write_selection_artifacts(
    output_dir: str | Path,
    result: ExpertScoreResult,
    *,
    model: Mapping[str, Any],
    inference: Mapping[str, Any],
    score_function: ScoreFunction,
    experts_per_layer: int,
    calibration: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the full research artifact and portable ESFT configuration."""

    if score_function not in result.scores:
        raise ValueError(f"Score function {score_function!r} is unavailable")
    selected = select_experts_per_layer(
        result.scores[score_function], experts_per_layer
    )

    def selected_for(name: str) -> dict[str, list[int]]:
        return select_experts_per_layer(result.scores[name], experts_per_layer)

    def overlap(left: str, right: str) -> dict[str, float]:
        left_selected = selected_for(left)
        right_selected = selected_for(right)
        return {
            layer_id: len(set(left_selected[layer_id]) & set(right_selected[layer_id]))
            / experts_per_layer
            for layer_id in left_selected
        }

    token_scores = result.scores["esft_token"]
    token_totals = token_scores.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        token_scores,
        token_totals,
        out=np.zeros_like(token_scores),
        where=token_totals > 0,
    )
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    diagnostics: dict[str, Any] = {
        "per_layer_routing_entropy": entropy.tolist(),
        "top_experts": {
            name: selected_for(name)
            for name in ("esft_gate", "esft_token", "reward_gate", "reward_token")
        },
        "ordinary_reward_topk_overlap": {
            "gate": overlap("esft_gate", "reward_gate"),
            "token": overlap("esft_token", "reward_token"),
        },
        "nearly_uniform_layers": {
            name: [
                layer_id
                for layer_id, values in enumerate(scores)
                if float(values.max() - values.min()) <= 1e-8
            ]
            for name, scores in result.scores.items()
            if name in ("esft_gate", "esft_token", "reward_gate", "reward_token")
        },
        "experts_receiving_no_traffic": {
            str(layer_id): np.flatnonzero(values == 0).tolist()
            for layer_id, values in enumerate(token_scores)
        },
        "response_length_scored_position_mismatches": 0,
    }
    if "successful_gate" in result.scores and "failed_gate" in result.scores:
        diagnostics["successful_failed_topk_overlap"] = {
            "gate": overlap("successful_gate", "failed_gate"),
            "token": overlap("successful_token", "failed_token"),
        }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = asdict(result.metadata)
    if metadata["moe_layer_ids"] is not None:
        metadata["moe_layer_ids"] = list(metadata["moe_layer_ids"])
    calibration_data = {
        "num_trajectories": result.num_trajectories,
        "num_valid_trajectories": result.num_valid_trajectories,
        "num_zero_length_trajectories": result.num_zero_length_trajectories,
        "num_scored_tokens": result.num_scored_tokens,
        "reward_mean": result.reward_mean,
        "reward_std": result.reward_std,
        "total_selection_reward": result.total_selection_reward,
        "num_positive_reward_trajectories": (result.num_positive_reward_trajectories),
        "num_zero_reward_trajectories": result.num_zero_reward_trajectories,
        "num_negative_raw_reward_trajectories": (
            result.num_negative_raw_reward_trajectories
        ),
        **(dict(calibration) if calibration is not None else {}),
    }
    full_artifact = {
        "schema_version": 1,
        "model": dict(model),
        "inference": dict(inference),
        "routing": {
            "alignment": "causal_generation",
            **metadata,
        },
        "calibration": calibration_data,
        "scores": {name: values.tolist() for name, values in result.scores.items()},
        "selected_experts": selected,
        "diagnostics": diagnostics,
        "selection": {
            "method": score_function,
            "experts_per_layer": experts_per_layer,
        },
    }
    full_path = output_path / "reward_expert_scores.json"
    full_path.write_text(
        json.dumps(full_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    esft_config = {
        "schema_version": 1,
        "model": dict(model),
        "routing_metadata": metadata,
        "selection": {
            "method": score_function,
            "experts_per_layer": experts_per_layer,
            "reward_type": (
                "transformed_nonnegative"
                if result.num_negative_raw_reward_trajectories
                else "raw_nonnegative"
            ),
            "routing_alignment": "causal_generation",
        },
        "experts": selected,
    }
    config_path = output_path / "esft_config.json"
    config_path.write_text(
        json.dumps(esft_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return full_path, config_path
