import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import build_index, fit_normalizers
from bergson.data import MemmapDataset, compute_batches
from bergson.gradients import GradientProcessor
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
        "--projection-dim",
        type=int,
        default=16,
        help="Dimension of the random projection for the index, or 0 to disable it.",
    )
    parser.add_argument(
        "--token-batch-size",
        type=int,
        default=8192,
        help="Batch size in tokens for building the index.",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="text",
        help="Column in the dataset that contains the prompts.",
    )
    parser.add_argument(
        "--completion-column",
        type=str,
        default="",
        help="Optional column in the dataset that contains the completions.",
    )
    parser.add_argument(
        "--index-size",
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

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    def tokenize(batch):
        # We're dealing with a prompt-completion dataset
        if args.completion_column:
            return tokenizer.apply_chat_template(
                conversation=[
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                    ]
                    for prompt, resp in zip(
                        batch[args.prompt_column], batch[args.completion_column]
                    )
                ],
                return_dict=True,
                tokenizer_kwargs=dict(
                    return_attention_mask=False,
                    return_length=True,
                ),
                truncation=True,
            )
        # We're dealing with vanilla next-token prediction
        else:
            return tokenizer(
                batch[args.prompt_column],
                return_attention_mask=False,
                return_length=True,
                truncation=True,
            )

    if args.dataset.endswith(".bin"):
        # TODO: Make this configurable, right now this is just a hack to support
        # the Pythia preshuffled Pile dataset.
        MEMMAP_CTX_LEN = 2049

        # If the dataset is a memmap file, use MemmapDataset
        ds = MemmapDataset(args.dataset, MEMMAP_CTX_LEN)
        ds = ds.shard(world_size, rank)

        # Uniform batches
        batch_size = args.token_batch_size // MEMMAP_CTX_LEN
        batches = [
            slice(start, start + batch_size) for start in range(0, len(ds), batch_size)
        ]
    else:
        ds = assert_type(Dataset, load_dataset(args.dataset, split="train"))

        # Shuffle before sharding to make sure each rank gets a different subset
        ds = ds.shuffle(seed=42)
        ds = ds.shard(world_size, rank)

        # Tokenize
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        ds = ds.sort("length", reverse=True)
        batches = compute_batches(ds["length"], args.token_batch_size)

    if os.path.exists(args.run_name):
        processor = GradientProcessor.load(args.run_name, map_location=f"cuda:{rank}")
    else:
        if rank == 0:
            print("Estimating normalizers...")

        processor = GradientProcessor(
            normalizers=fit_normalizers(
                model,
                ds,
                batches=batches,
                max_documents=10_000,
            ),
            projection_dim=args.projection_dim or None,
        )

    # TODO: Actually use the preconditioner
    if args.precondition and not processor.preconditioners:
        if rank == 0:
            print("Estimating preconditioner...")

        # We need a lot of examples for the preconditioner
        processor.estimate_preconditioners(model, ds, num_examples=10_000)

    processor.save(args.run_name)
    if rank == 0:
        print("Building index...")

    # Build the index
    build_index(model, ds, processor, args.run_name, batches=batches)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
