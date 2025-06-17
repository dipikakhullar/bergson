import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data import IndexConfig, compute_batches, tokenize
from .gradients import GradientProcessor
from .processing import collect_gradients, fit_normalizers
from .utils import assert_type


def worker(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset):
    dist.init_process_group(
        "nccl",
        init_method="tcp://localhost:29500",
        device_id=torch.device(f"cuda:{rank}"),
        rank=rank,
        timeout=timedelta(hours=1),
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=True) if cfg.load_in_8bit else None
        ),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else "auto",
    )

    # Check for PEFT adapters
    try:
        adapters = model.active_adapters()
    except ValueError:
        target_modules = None
    else:
        if rank == 0:
            print("PEFT model detected.")

        target_modules = set()

        for adapter_name in adapters:
            state = model.get_adapter_state_dict(adapter_name)

            for name in state:
                prefix = name.removesuffix(".weight")
                name = prefix + "." + adapter_name

                try:
                    model.get_submodule(name)
                except AttributeError:
                    print(f"Adapter parameter '{name}' not found in the model.")

                target_modules.add(name.removeprefix("model."))

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    batches = compute_batches(ds["length"], cfg.token_batch_size)
    if os.path.exists(cfg.processor_path):
        if rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            cfg.processor_path,
            map_location=f"cuda:{rank}",
        )
    else:
        if cfg.normalizer != "none":
            normalizers = fit_normalizers(
                model,
                ds,
                batches=batches,
                kind=cfg.normalizer,
                max_documents=cfg.stats_sample_size or None,
                target_modules=target_modules,
            )
        else:
            normalizers = {}

        processor = GradientProcessor(
            normalizers,
            fisher_fourth_root=cfg.fisher_fourth_root,
            projection_dim=cfg.projection_dim or None,
        )
        processor.save(cfg.run_path)

    # Build the index
    collect_gradients(
        model,
        ds,
        processor,
        cfg.run_path,
        batches=batches,
        target_modules=target_modules,
    )
    dist.destroy_process_group()


def build_index(cfg: IndexConfig):
    # Do all the data loading and preprocessing on the main process
    try:
        ds = assert_type(Dataset, load_dataset(cfg.dataset, split="train"))
    except ValueError as e:
        # Automatically use load_from_disk if appropriate
        if "load_from_disk" in str(e):
            ds = Dataset.load_from_disk(cfg.dataset, keep_in_memory=False)
        else:
            raise e

    metadata = {"length"}
    if cfg.drop_columns:
        metadata |= set(ds.column_names)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    ds = ds.map(lambda _, idx: dict(_row=idx), with_indices=True).shuffle(seed=42)
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg, tokenizer=tokenizer),
    )
    ds = ds.sort("length", reverse=True)

    world_size = torch.cuda.device_count()
    ctx = start_processes(
        "build",
        worker,
        args={i: (i, world_size, cfg, ds) for i in range(world_size)},
        envs={i: {"LOCAL_RANK": str(i)} for i in range(world_size)},
        logs_specs=DefaultLogsSpecs(),
    )
    ctx.wait()
