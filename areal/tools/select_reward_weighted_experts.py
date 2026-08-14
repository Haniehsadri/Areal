# SPDX-License-Identifier: Apache-2.0

"""Build reward-weighted expert-selection artifacts from trajectory JSONL."""

from __future__ import annotations

import argparse

from areal.utils.reward_expert_selection import (
    RoutingMetadata,
    aggregate_expert_scores,
    read_trajectory_jsonl,
    write_selection_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--checkpoint-hash", default="")
    parser.add_argument("--backend", choices=("vllm", "vllm_ascend"), required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--responses-per-prompt", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-moe-layers", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--router-top-k", type=int, required=True)
    parser.add_argument("--experts-per-layer", type=int, required=True)
    parser.add_argument(
        "--score-function",
        choices=("esft_gate", "esft_token", "reward_gate", "reward_token"),
        default="reward_gate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = RoutingMetadata(
        num_moe_layers=args.num_moe_layers,
        num_experts=args.num_experts,
        router_top_k=args.router_top_k,
    )
    records = read_trajectory_jsonl(args.trajectories, metadata)
    result = aggregate_expert_scores(records, metadata)
    write_selection_artifacts(
        args.output_dir,
        result,
        model={
            "name_or_path": args.model,
            "revision": args.revision,
            "architecture": args.architecture,
            "checkpoint_hash": args.checkpoint_hash,
        },
        inference={
            "backend": args.backend,
            "backend_version": args.backend_version,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "expert_parallel_size": args.expert_parallel_size,
        },
        score_function=args.score_function,
        experts_per_layer=args.experts_per_layer,
        calibration={
            key: value
            for key, value in {
                "dataset": args.dataset,
                "num_prompts": args.num_prompts,
                "responses_per_prompt": args.responses_per_prompt,
                "seed": args.seed,
            }.items()
            if value is not None and value != ""
        },
    )


if __name__ == "__main__":
    main()
