import math
import random
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from tqdm.auto import tqdm, trange
from transformers import PreTrainedModel

from .data import create_index, pad_and_tensor
from .gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
    Normalizer,
)


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
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

    # Mutable state for the GradientCollector callback
    mod_grads = []
    preconditioners = {}

    def callback(name: str, g: torch.Tensor):
        # We aren't interested in the matrix-shape of the gradient
        g = g.flatten(1)

        # Asynchronously move the gradient to CPU and convert to fp16
        mod_grads.append(g.to(device="cpu", dtype=torch.float16, non_blocking=True))

        # Compute the outer product of the flattened gradient
        g = g.float()
        preconditioner = preconditioners.get(name, None)
        if preconditioner is None:
            preconditioners[name] = g.mT @ g
        else:
            preconditioner.addmm_(g.mT, g)

    collector = GradientCollector(
        model.base_model,
        callback,
        processor,
        target_modules=target_modules,
    )
    # Allocate space ahead of time for the gradients
    grad_size = sum(math.prod(s) for s in collector.shapes().values())
    grad_buffer = create_index(
        path + "/gradients.bin",
        dtype=np.float16,
        shape=(len(data), grad_size),
    )
    per_token_losses: list[np.ndarray] = []

    for sl in tqdm(batches, disable=rank != 0, desc="Building index"):
        batch = data[sl]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )

        with collector:
            logits = model(x).logits
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])

            masks = y[:, 1:] != -100
            denoms = masks.sum(dim=1, dtype=logits.dtype)
            avg_loss = losses.sum(1).div(denoms).mean()

            # Start sending losses to the CPU just before the backward. Don't force
            # a blocking host-device sync here.
            losses = losses.detach().to(
                device="cpu",
                dtype=torch.float16,
                non_blocking=True,
            )
            masks = masks.to(device="cpu", non_blocking=True)
            avg_loss.backward()

            model.zero_grad()

        # This forces a host-device sync, but hopefully the transfer to CPU is
        # already done since we called to("cpu", non_blocking=True) in the callback.
        # We could make this even better, potentially, by using a ring buffer to wait
        # longer before syncing.
        indices = batch.get("_row") or sl
        grad_buffer[indices, :] = torch.cat(mod_grads, dim=1).numpy()
        mod_grads.clear()

        for loss, mask in zip(losses, masks):
            # We only store the losses for the tokens that are not masked
            per_token_losses.append(loss[mask].numpy())

    processor.preconditioners = preconditioners
    processor.save(path)

    # Make sure the gradients are written to disk
    grad_buffer.flush()


def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset,
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

    def adam_update(name: str, g: torch.Tensor):
        sq = g.square_().float().sum(0)

        # initialize accumulators at zero
        if (normalizer := normalizers.get(name)) is None:
            normalizers[name] = normalizer = AdamNormalizer(torch.zeros_like(sq))
        else:
            assert isinstance(normalizer, AdamNormalizer)

        # in‐place accumulate
        normalizer.avg_sq.add_(sq)

    N = 0
    callback = adafactor_update if kind == "adafactor" else adam_update
    total = (max_documents or len(batches)) // world_size
    pbar = trange(total, disable=rank != 0, desc="Estimating normalizers")

    for sl in batches:
        batch = data[sl]

        # Update progress
        n = len(batch["input_ids"])
        pbar.update(n)

        N += n
        if total and N >= total:
            break

        with GradientCollector(
            model.base_model,
            closure=callback,
            target_modules=target_modules,
        ):
            x, y = pad_and_tensor(
                batch["input_ids"],  # type: ignore
                labels=batch.get("labels", None),  # type: ignore
                device=model.device,
            )
            model(x, labels=y).loss.backward()
            model.zero_grad()

    # Divide by the number of documents processed and average across all ranks
    for normalizer in normalizers.values():
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
