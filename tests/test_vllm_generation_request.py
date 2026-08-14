import base64
import io

import numpy as np

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.engine.vllm_remote import VLLMBackend


def test_vllm_forwards_frequency_penalty_and_stop():
    """The vLLM backend must forward frequency_penalty and stop like the SGLang
    backend does; both are GenerationHyperparameters and are accepted by vLLM's
    OpenAI-compatible /v1/completions endpoint."""
    gconfig = GenerationHyperparameters(
        max_new_tokens=8, frequency_penalty=0.5, stop=["STOP"]
    )
    req = ModelRequest(input_ids=[11, 12], gconfig=gconfig)

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["frequency_penalty"] == 0.5
    assert payload["stop"] == ["STOP"]


def test_vllm_routing_request_starts_at_final_prompt_position():
    """Causal collection includes only the position predicting token one."""
    req = ModelRequest(input_ids=[11, 12, 13])
    req.metadata["return_routed_experts"] = True

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["routed_experts_prompt_start"] == 2


def test_vllm_parse_generation_response_decodes_aligned_routing_arrays():
    """vLLM's base64 NumPy payload preserves expert IDs and final weights."""
    expert_ids = np.asarray([[[1, 3]], [[2, 0]]], dtype=np.uint8)
    expert_weights = np.asarray([[[0.7, 0.3]], [[0.6, 0.4]]], dtype=np.float32)

    def encode(values: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.save(buffer, values)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    result = VLLMBackend().parse_generation_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["token:4", "token:5"],
                        "token_logprobs": [-0.1, -0.2],
                    },
                    "routed_experts": encode(expert_ids),
                    "routed_expert_weights": encode(expert_weights),
                }
            ]
        }
    )

    np.testing.assert_array_equal(result.routed_experts, expert_ids)
    np.testing.assert_allclose(
        result.routed_expert_weights, expert_weights, rtol=0, atol=0
    )
