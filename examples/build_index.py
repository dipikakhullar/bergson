import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import build_index, estimate_preconditioner, estimate_second_moments
from bergson.data import MemmapDataset
from bergson.utils import assert_type


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "run_name",
        type=str,
        help="Name of the run. Used to create a directory for the index.",
    )
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--dataset",
        type=str,
        default="EleutherAI/SmolLM2-135M-10B",
    )
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
    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group("nccl")

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": f"cuda:{rank}"}
    )

    if args.dataset.endswith(".bin"):
        # If the dataset is a memmap file, use MemmapDataset
        dataset = MemmapDataset(args.dataset, 2049)
        dataset = dataset.shard(world_size, rank)
    else:
        dataset = assert_type(Dataset, load_dataset(args.dataset, split="train")).shard(
            world_size, rank
        )
        # Tokenize
        dataset = (
            dataset.map(
                lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
                batched=True,
                remove_columns=["text"],
            )
            .map(
                lambda x: {"length": len(x["input_ids"])},
            )
            .sort("length")
        )

    # Create the run directory
    os.makedirs(args.run_name, exist_ok=True)

    if not os.path.exists(args.run_name + "/second_moments.pth"):
        if rank == 0:
            print("Estimating second moments...")

        moments = estimate_second_moments(model, dataset, num_examples=1000)
        torch.save(moments, args.run_name + "/second_moments.pth")
    else:
        if rank == 0:
            print("Loading second moments from disk.")

        moments = torch.load(
            args.run_name + "/second_moments.pth",
            map_location=f"cuda:{rank}",
            weights_only=True,
        )

    if args.precondition:
        # TODO: Actually use the preconditioner
        if not os.path.exists(args.run_name + "/preconditioner.pth"):
            if rank == 0:
                print("Estimating preconditioner...")

            # We need a lot of examples for the preconditioner
            preconditioner = estimate_preconditioner(
                model, dataset, moments, num_examples=10_000
            )
            torch.save(preconditioner, args.run_name + "/preconditioner.pth")
        else:
            if rank == 0:
                print("Loading preconditioner from disk.")

            preconditioner = torch.load(
                args.run_name + "/preconditioner.pth",
                map_location=f"cuda:{rank}",
                weights_only=True,
            )

    if rank == 0:
        print("Building index...")

    # Build the index
    build_index(model, dataset, moments, args.run_name)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
