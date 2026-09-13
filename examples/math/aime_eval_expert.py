# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import sys

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.dataset import get_custom_dataset
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler
from areal.utils import logging, seeding
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats

logger = logging.getLogger("GSM8KEval")


def main(args):
    parser = argparse.ArgumentParser(description="Collect AIME routing for ESFT", add_help=False)
    parser.add_argument("--trajectory-output", required=True)
    parser.add_argument("--num-moe-layers", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--router-top-k", type=int, required=True)
    capture, config_args = parser.parse_known_args(args)
    from areal.utils.reward_expert_selection import RoutingMetadata

    metadata = RoutingMetadata(
        num_moe_layers=capture.num_moe_layers,
        num_experts=capture.num_experts,
        router_top_k=capture.router_top_k,
    )
    output = Path(capture.trajectory_output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh trajectory output path: {output}")
    config, _ = load_expr_config(config_args, GRPOConfig)
    logging.setup_file_logging(f"{config.cluster.fileroot}/eval.log")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key="eval")

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")

    if rollout_alloc.backend != "vllm":
        raise ValueError("AIME routing capture requires the patched vLLM backend")

    # Initialize scheduler
    cfg = config.scheduler
    if cfg.type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif cfg.type == "ray":
        scheduler = RayScheduler(exp_config=config)
    elif cfg.type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")

    # Load evaluation dataset
    valid_dataset = get_custom_dataset(
        dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )

    # Initialize RolloutController
    config.rollout.max_head_offpolicyness = int(1e12)

    if rollout_alloc.backend == "sglang":
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=rollout_alloc.parallel.tp_size,
            base_gpu_id=0,
        )
    elif rollout_alloc.backend == "vllm":
        engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=rollout_alloc.parallel.tp_size,
            pp_size=rollout_alloc.parallel.pp_size,
        )
    else:
        raise ValueError(f"Invalid backend: {rollout_alloc.backend}")

    eval_rollout = engine_cls.as_controller(config.rollout, scheduler)

    try:
        eval_rollout.initialize(
            role="eval-rollout",
            server_args=server_args,
        )

        # Create evaluation workflow
        workflow = "areal.workflow.reward_expert_selection.RewardExpertSelectionWorkflow"
        workflow_kwargs = dict(
            reward_fn="areal.reward.aime.aime_reward_fn",
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=True,
            trajectory_output=str(output),
            routing_metadata={
                "num_moe_layers": metadata.num_moe_layers,
                "num_experts": metadata.num_experts,
                "router_top_k": metadata.router_top_k,
            },
            reward_transform="identity",
        )

        # Submit all evaluation tasks
        cnt = 0
        for data in valid_dataloader:
            for item in data:
                eval_rollout.submit(
                    item,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    group_size=config.gconfig.n_samples,
                )
                cnt += 1

        eval_rollout.wait(cnt, timeout=None)
        if cnt == 0:
            raise ValueError("No calibration prompts were submitted; check split and drop_last")
        eval_stats = eval_rollout.export_stats()
        logger.info(f"Routing trajectories: {output}")

        # Print and log results
        logger.info(f"Evaluation Results: {tabulate_stats(eval_stats)}")
    finally:
        eval_rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
