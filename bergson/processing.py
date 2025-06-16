import json
import random
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Sequence, Value
from tqdm.auto import tqdm, trange
from transformers import PreTrainedModel

from .data import MemmapDataset, pad_and_tensor
from .gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
    Normalizer,
)
from .utils import assert_type


def build_index(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[slice] | None = None,
    target_modules: set[str] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    shapes = None
    grad_length = -1

    def generator():
        nonlocal shapes, grad_length
        for batch in process_batches(data, batches=batches, device=model.device, desc="Building index"):
            x, y, pbar, seq_lengths = batch["x"], batch["y"], batch["pbar"], batch["sequence_lengths"]
            with GradientCollector(
                model.base_model,
                processor,
                target_modules=target_modules,
                sequence_lengths=seq_lengths,
                move_to_cpu=True,
            ) as mgr:
                logits = model(x).logits
                losses = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    y[:, 1:].flatten(),
                    reduction="none",
                ).reshape_as(y[:, 1:])

                mask = y[:, 1:] != -100
                denoms = mask.sum(dim=1, dtype=logits.dtype).clamp(min=1)
                avg_loss = losses.sum(1).div(denoms).mean()
                avg_loss.backward()

                pbar.set_postfix(
                    loss=f"{avg_loss.item():.3f}",
                )
                model.zero_grad()

            gradient = mgr.flattened_grads().cpu().float().numpy()
            losses = losses.detach().cpu().float().numpy()

            # Define names, shapes, and lengths of the gradients for serialization
            if shapes is None:
                # Drop the batch dimension from the shape
                shapes = {n: g.shape[1:] for n, g in mgr.collected_grads.items()}
                grad_length = gradient.shape[-1]

            for i, (g, l, m) in enumerate(zip(gradient, losses, mask.cpu())):
                row = {k: v[i] for k, v in batch["original"].items()}
                row.update(gradient=g, loss=l[m])
                yield row

    if dist.is_initialized():
        # FSDP modules don't pickle. from_generator pickles to find a config name
        index = Dataset.from_list(list(generator()))
    else:
        index = Dataset.from_generator(generator)
    assert_type(Dataset, index)

    features = index.features.copy()
    features["gradient"] = Sequence(Value("float32"), length=grad_length)
    index = index.cast(features=features)

    idx_path = path + f"/rank_{rank}.idx"
    index.save_to_disk(idx_path)  # type: ignore

    # Save the shapes of the gradients for later use
    if rank == 0:
        shapes_path = path + "/shapes.json"

        with open(shapes_path, "w") as f:
            json.dump(shapes, f, indent=2)

    return index


def process_batches(data, batches: list[slice] | None = None,
                    max_documents: int | None = None, device: torch.device | None = None,
                    max_length: int | None = None,
                    *, desc):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # If max_tokens is specified, randomly select a subset of batches
    elif max_documents is not None:
        batches = batches.copy()

        rng = random.Random(rank)
        rng.shuffle(batches)

    max_batch_len = torch.tensor(len(batches))
    if dist.is_initialized():
        dist.all_reduce(max_batch_len, op=dist.ReduceOp.MAX)
    max_batch_len = int(max_batch_len.item())
    last_batch_extra = False
    if len(batches) < max_batch_len:
        # assert max_batch_len - len(batches) == 1, f"Rank {rank} has {len(batches)} batches, but {max_batch_len} are needed"
        batches.append(slice(0, 1))
        last_batch_extra = True

    N = 0
    total = (max_documents or len(batches)) // world_size
    pbar = trange(total, disable=rank != 0, desc=desc)
    
    for i, sl in enumerate(batches):
        batch = data[sl]
        # Update progress
        n = len(batch["input_ids"])
        pbar.update(n)
        
        last_fake = i == len(batches) - 1 and last_batch_extra
        if not last_fake:
            N += n
        done = torch.tensor(float(total and N >= total))
        if dist.is_initialized():
            dist.all_reduce(done, op=dist.ReduceOp.SUM)
        if done.item() > 0:
            break
        
        sequence_lengths = torch.tensor([len(ids) for ids in batch["input_ids"]], device=device)
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels", None),  # type: ignore
            device=device,
            max_len=max_length
        )
        if last_fake:
            y[:] = -100

        yield {"x": x, "y": y, "N": N, "pbar": pbar, "original": batch, "sequence_lengths": sequence_lengths}

def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    *,
    batches: list[slice] | None = None,
    kind: Literal["adafactor", "adam"] = "adafactor",
    max_documents: int | None = None,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    normalizers: dict[str, Normalizer] = {}
    @torch.no_grad
    def adafactor_update(name: str, g: torch.Tensor):
        # We follow the tensor2tensor implementation of Adafactor, which
        # takes the mean rather than summing over the rows and columns.
        # row: mean over columns, shape [O]
        sq = g.float().square_().sum(0)
        row_acc = sq.mean(dim=1)
        # col: mean over rows,    shape [I]
        col_acc = sq.mean(dim=0)

        if (normalizer := normalizers.get(name)) is None:
            # initialize accumulators at zero
            normalizers[name] = normalizer = AdafactorNormalizer(
                torch.zeros_like(row_acc),
                torch.zeros_like(col_acc),
            )
        else:
            assert isinstance(normalizer, AdafactorNormalizer)

        # in‐place accumulate
        normalizer.row.add_(row_acc)
        normalizer.col.add_(col_acc)

    @torch.no_grad
    def adam_update(name: str, g: torch.Tensor):
        sq = g.square_().float().sum(0)

        # initialize accumulators at zero
        if (normalizer := normalizers.get(name)) is None:
            normalizers[name] = normalizer = AdamNormalizer(torch.zeros_like(sq))
        else:
            assert isinstance(normalizer, AdamNormalizer)

        # in‐place accumulate
        normalizer.avg_sq.add_(sq)

    callback = adafactor_update if kind == "adafactor" else adam_update

    for batch in process_batches(data, batches=batches, max_documents=max_documents, device=model.device, desc="Estimating normalizers"):
        # torch.cuda.empty_cache()

        x, y, N, seq_lengths = batch["x"], batch["y"], batch["N"], batch["sequence_lengths"]
        
        model(x, labels=y).loss.backward()
        model.zero_grad()

        with GradientCollector(
            model.base_model,
            closure=callback,
            target_modules=target_modules,
            sequence_lengths=seq_lengths
        ):
            model(x, labels=y).loss.backward()
            model.zero_grad()

    # Divide by the number of documents processed and average across all ranks
    # for normalizer in normalizers.values():
    for i, normalizer in enumerate(normalizers.values()):
        if isinstance(normalizer, AdamNormalizer):
            normalizer.avg_sq.div_(N)

            if dist.is_initialized():
                dist.all_reduce(normalizer.avg_sq, op=dist.ReduceOp.AVG)

        elif isinstance(normalizer, AdafactorNormalizer):
            normalizer.row.div_(N)
            normalizer.col.div_(N)

            if dist.is_initialized():
                dist.all_reduce(normalizer.row, op=dist.ReduceOp.AVG)
                dist.all_reduce(normalizer.col, op=dist.ReduceOp.AVG)

    return normalizers
