# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific inference bridge backend."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import numpy as np

from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    detect_image_mime,
    get_versioned_lora_name,
)

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


class VLLMBridgeBackend:
    """vLLM-specific backend for :class:`InfBridge`.

    Mirrors the relevant subset of
    :class:`areal.engine.vllm_remote.VLLMBackend`.
    """

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        """Build a ``/v1/completions`` or ``/v1/chat/completions`` request."""
        gconfig = req.gconfig

        # Compute effective max_new_tokens (cap by remaining context window)
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids),
            gconfig.max_new_tokens,
        )

        payload: dict[str, Any] = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
            "use_beam_search": gconfig.use_beam_search,
            "stream": False,
        }
        if req.metadata.get("return_routed_experts", False):
            payload["routed_experts_prompt_start"] = max(len(req.input_ids) - 1, 0)

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["model"] = get_versioned_lora_name(lora_name, version)

        if req.vision_msg_vllm:
            images = iter(req.image_data or [])
            parsed_input = req.vision_msg_vllm[0]
            for msg in parsed_input:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content.get("type") == "image_url":
                            try:
                                base64_img = next(images)
                            except StopIteration as exc:
                                raise ValueError(
                                    "Not enough images in req.image_data to match image_url entries."
                                ) from exc
                            mime = detect_image_mime(base64_img)
                            content["image_url"] = {
                                "url": f"data:{mime};base64,{base64_img}"
                            }
            payload["messages"] = parsed_input.copy()
            payload["logprobs"] = True
            return HttpRequest(endpoint="/v1/chat/completions", payload=payload)

        payload["prompt"] = list(req.input_ids)
        return HttpRequest(endpoint="/v1/completions", payload=payload)

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        """Parse vLLM JSON into :class:`HttpGenerationResult`."""
        meta_info = response["choices"][0]
        stop_reason = meta_info["finish_reason"]

        def decode_array(field: str) -> np.ndarray | None:
            encoded = meta_info.get(field)
            if encoded is None:
                return None
            return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)

        routed_experts = decode_array("routed_experts")
        routed_expert_weights = decode_array("routed_expert_weights")
        if (routed_experts is None) != (routed_expert_weights is None):
            raise ValueError("vLLM returned incomplete routed-expert data")

        if "tokens" in meta_info["logprobs"]:
            output_tokens = [
                int(token.split(":")[1]) for token in meta_info["logprobs"]["tokens"]
            ]
            output_logprobs = meta_info["logprobs"]["token_logprobs"]
        elif "content" in meta_info["logprobs"]:
            outputs = meta_info["logprobs"]["content"]
            output_tokens = [int(token["token"].split(":")[1]) for token in outputs]
            output_logprobs = [token["logprob"] for token in outputs]
        else:
            raise ValueError("Unexpected vLLM response format.")

        if stop_reason == "abort" and len(output_tokens) == 0:
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
                routed_experts=routed_experts,
                routed_expert_weights=routed_expert_weights,
            )

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            routed_experts=routed_experts,
            routed_expert_weights=routed_expert_weights,
        )

    def get_pause_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/areal_pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/areal_continue_generation", payload={})

    def get_offload_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/sleep", payload={}, method="POST")

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        if tags is not None:
            from urllib.parse import urlencode

            tags_query = urlencode({"tags": tags}, doseq=True)
            endpoint = f"/wake_up?{tags_query}"
        else:
            endpoint = "/wake_up"
        return HttpRequest(endpoint=endpoint, payload={}, method="POST")

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        return int(http_req.payload["max_tokens"])

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        http_req.payload["max_tokens"] = remaining_tokens
        if "prompt" in http_req.payload:
            http_req.payload["prompt"] = list(req.input_ids) + accumulated_tokens
