# SPDX-License-Identifier: Apache-2.0

"""Run RLVR on NuminaMath prepared with areal/tools/prepare_numinamath.py."""

import sys
from pathlib import Path

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    for dataset_config in (config.train_dataset, config.valid_dataset):
        if not dataset_config.path or not (
            Path(dataset_config.path) / "dataset_dict.json"
        ).is_file():
            raise ValueError("Dataset path must point to a prepared local DatasetDict")
        if dataset_config.type != "rl":
            raise ValueError("NuminaMath RLVR requires dataset type: rl")
    if config.train_dataset.split != "train":
        raise ValueError("Use the train split for optimization")
    if config.valid_dataset.split not in ("validation", "test"):
        raise ValueError("Use validation or test for evaluation")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_kwargs = dict(kwargs)
    eval_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)
    workflow = "areal.workflow.rlvr.RLVRWorkflow"
    with PPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
