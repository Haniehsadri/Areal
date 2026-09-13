# SPDX-License-Identifier: Apache-2.0

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    if config.rollout.teacher:
        raise ValueError("This AIME entry point supports RLVR, not teacher distillation.")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    workflow = "areal.workflow.rlvr.RLVRWorkflow"
    workflow_kwargs = dict(
        reward_fn="areal.reward.aime.aime_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=True,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)
    with PPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
