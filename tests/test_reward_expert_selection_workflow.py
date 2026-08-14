# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from areal.api.io_struct import ModelRequest, ModelResponse
from areal.utils.reward_expert_selection import RoutingMetadata
from areal.workflow.reward_expert_selection import RewardExpertSelectionWorkflow
from areal.workflow.rlvr import RLVRWorkflow


@pytest.mark.asyncio
async def test_reward_expert_selection_workflow_writes_reward_aligned_record(
    tmp_path, monkeypatch
):
    """The calibration workflow joins one response's routing and reward."""
    response = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        routed_experts=np.asarray([[[0, 1]], [[1, 2]]], dtype=np.int32),
        routed_expert_weights=np.asarray(
            [[[0.7, 0.3]], [[0.6, 0.4]]], dtype=np.float32
        ),
    )
    base_collect = AsyncMock(return_value=(response, 1.0))
    monkeypatch.setattr(RLVRWorkflow, "_collect_samples", base_collect)

    workflow = RewardExpertSelectionWorkflow.__new__(RewardExpertSelectionWorkflow)
    workflow.trajectory_output = tmp_path / "trajectories.jsonl"
    workflow.routing_metadata = RoutingMetadata(
        num_moe_layers=1, num_experts=3, router_top_k=2
    )
    workflow.reward_transform = "identity"
    workflow.prompt_id_field = "prompt_id"
    workflow.response_id_field = "response_id"
    workflow._write_lock = asyncio.Lock()
    request = ModelRequest(rid="request-7", input_ids=[1, 2])

    returned_response, reward = await workflow._collect_samples(
        engine=object(),
        req=request,
        prompt_str="prompt",
        task_data={"prompt_id": "prompt-3", "response_id": 7},
    )

    assert returned_response is response
    assert reward == 1.0
    assert request.metadata["return_routed_experts"] is True
    record = json.loads(workflow.trajectory_output.read_text(encoding="utf-8"))
    assert record["request_id"] == "request-7"
    assert record["raw_reward"] == 1.0
    assert record["routing"]["gate_sum"] == {
        "0": {
            "0": pytest.approx(0.7),
            "1": pytest.approx(0.9),
            "2": pytest.approx(0.4),
        }
    }
