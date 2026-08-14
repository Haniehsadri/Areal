# SPDX-License-Identifier: Apache-2.0

"""RLVR calibration workflow that exports reward-aligned MoE routing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiofiles

from areal.api import InferenceEngine, ModelRequest, ModelResponse
from areal.api.cli_args import GenerationHyperparameters
from areal.utils.reward_expert_selection import (
    RewardTransform,
    RoutingMetadata,
    trajectory_record_from_routing,
)
from areal.workflow.rlvr import (
    RLVRWorkflow,
    default_data_extract_prompt_fn,
    default_get_input_ids_fn,
)


class RewardExpertSelectionWorkflow(RLVRWorkflow):
    """Run ordinary RLVR evaluation and append reward-aligned routing JSONL.

    Each workflow instance serializes its own asynchronous appends. When several
    processes run calibration, configure a distinct output shard per process and
    concatenate the JSONL shards before running the selector.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: Any,
        trajectory_output: str,
        routing_metadata: dict[str, Any],
        reward_transform: RewardTransform = "identity",
        prompt_id_field: str = "prompt_id",
        response_id_field: str = "response_id",
        enable_thinking: bool = False,
        get_input_ids_fn: Callable[..., list[int]] | str = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any]
        | str = default_data_extract_prompt_fn,
    ) -> None:
        super().__init__(
            reward_fn=reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            get_input_ids_fn=get_input_ids_fn,
            data_extract_prompt_fn=data_extract_prompt_fn,
        )
        self.trajectory_output = Path(trajectory_output)
        self.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        self.routing_metadata = RoutingMetadata(**routing_metadata)
        self.reward_transform = reward_transform
        self.prompt_id_field = prompt_id_field
        self.response_id_field = response_id_field
        self._write_lock = asyncio.Lock()

    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """Generate, reward, validate routing, and append one trajectory."""

        req.metadata["return_routed_experts"] = True
        resp, reward = await super()._collect_samples(
            engine, req, prompt_str, task_data
        )
        if resp.routed_experts is None or resp.routed_expert_weights is None:
            raise RuntimeError(
                "Reward expert selection requires routed expert IDs and weights"
            )
        prompt_id = str(task_data.get(self.prompt_id_field, req.rid))
        response_id = int(task_data.get(self.response_id_field, 0))
        record = trajectory_record_from_routing(
            prompt_id=prompt_id,
            response_id=response_id,
            request_id=req.rid,
            prompt_token_ids=resp.input_tokens,
            response_token_ids=resp.output_tokens,
            raw_reward=float(reward),
            finish_reason=resp.stop_reason,
            expert_ids=resp.routed_experts,
            expert_weights=resp.routed_expert_weights,
            metadata=self.routing_metadata,
            reward_transform=self.reward_transform,
        )
        line = record.to_json_dict()
        async with self._write_lock:
            async with aiofiles.open(
                self.trajectory_output, "a", encoding="utf-8"
            ) as output:
                await output.write(json.dumps(line, sort_keys=True) + "\n")
        return resp, reward
