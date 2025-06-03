import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from transformers import default_data_collator

from quelle.approx_unrolling.build_index import build_index
from quelle.approx_unrolling.EK_FAC import compute_EK_FAC_checkpoints
from quelle.approx_unrolling.language_task import LanguageModelingTask
from quelle.approx_unrolling.model_checkpoints import PythiaCheckpoints
from quelle.approx_unrolling.pile_data import get_pile_dataset
from quelle.hessians import Analyzer, prepare_model
from quelle.hessians.arguments import FactorArguments
from quelle.hessians.utils.common.factor_arguments import (
    all_low_precision_factor_arguments,
)
from quelle.hessians.utils.dataset import DataLoaderKwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis.")

    # parser.add_argument(
    #     "--checkpoint_dir",
    #     type=str,
    #     default="./checkpoints",
    #     help="A path that is storing models and (approximate) Hessians.",
    # )

    parser.add_argument(
        "--index_size",
        type=int,
        default=0,
        help="Maximum of examples to use for the index, or 0 to use the entire dataset",
    )
    parser.add_argument(
        "--precondition",
        action="store_true",
        help="Use the preconditioner to build the index",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=False,
        help="Whether to use torch compile for computing factors and scores.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Batch size for computing matrices.",
    )
    parser.add_argument(
        "--compute_per_token_scores",
        action="store_true",
        default=False,
        help="Boolean flag to compute per token scores.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )

    parser.add_argument(
        "--pythia_model_name",
        type=str,
        default="EleutherAI/pythia-14m",
        help="Pythia model name.",
    )

    # TODO: Fix this to take list of list
    # parser.add_argument(
    #     "--checkpoints",
    #     type=int,
    #     nargs="+",
    #     default=[2000, 3000, 4000],
    #     help="List of checkpoints to load.",
    # )

    args = parser.parse_args()

    return args


def run_hessians():
    args = parse_args()

    # all_checkpoints = [[2000, 3000, 4000], [6000]]
    all_checkpoints = [[1000]]
    model_name = args.pythia_model_name

    pythia_checkpoints_manager = PythiaCheckpoints(all_checkpoints, model_name)
    pythia_checkpoints_manager.save_models(overwrite=False)

    assert pythia_checkpoints_manager.module_keys is not None

    task = LanguageModelingTask(module_keys=pythia_checkpoints_manager.module_keys)
    pythia_checkpoints_manager.module_keys = task.get_influence_tracked_modules()

    model = pythia_checkpoints_manager.load_checkpoint(checkpoint=1000)
    EK_FAC_args = args
    train_dataset = get_pile_dataset(model_str=args.pythia_model_name, step=0, max_samples=4000)
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=EK_FAC_args.factor_strategy)  # type:ignore
    if EK_FAC_args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=EK_FAC_args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    if EK_FAC_args.use_compile:
        factors_name += "_compile"
    # Define task and prepare model.
    model = prepare_model(model, task)

    if args.use_compile:
        model = torch.compile(model)  # type: ignore

    analyzer = Analyzer(
        analysis_name="",
        model=model,
        task=task,
        profile=args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=EK_FAC_args.per_device_batch_size,
        factor_args=factor_args,
        initial_per_device_batch_size_attempt=32,
        dataloader_kwargs=dataloader_kwargs,
    )


def run():
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        # Single GPU/CPU mode
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        print("Running in single-GPU mode")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU
    np.random.seed(42)
    random.seed(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(rank)

    args = parse_args()

    # all_checkpoints = [[2000, 3000, 4000], [6000]]
    all_checkpoints = [[1000]]
    model_name = args.pythia_model_name

    pythia_checkpoints_manager = PythiaCheckpoints(all_checkpoints, model_name)
    pythia_checkpoints_manager.save_models(overwrite=False)

    assert pythia_checkpoints_manager.module_keys is not None

    task = LanguageModelingTask(module_keys=pythia_checkpoints_manager.module_keys)
    pythia_checkpoints_manager.module_keys = task.get_influence_tracked_modules()

    train_dataset = get_pile_dataset(model_str=args.pythia_model_name, step=0, max_samples=4000)

    compute_EK_FAC_checkpoints(
        checkpoint_manager=pythia_checkpoints_manager,
        task=task,
        train_dataset=train_dataset,
        EK_FAC_args=args,
    )
    if rank == 0:
        print("Building index...")

    # Build the index
    build_index(pythia_checkpoints_manager, train_dataset)

    # dist.destroy_process_group()


if __name__ == "__main__":
    # run()

    run()
# python unrolling.py --query_batch_size 32 --train_batch_size 64 --use_half_precision --per_device_batch_size 32 --pythia_model_name EleutherAI/pythia-14m
