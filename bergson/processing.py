import random

import torch
import torch.distributed as dist
from datasets import Dataset, Sequence, Value
from tqdm.auto import tqdm, trange
from transformers import PreTrainedModel

from .data import MemmapDataset, pad_and_tensor
from .gradients import AdafactorNormalizer, GradientCollector, GradientProcessor


@torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported())
def build_index(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[slice] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    index = None
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # Store the grads here
    gradients = []

    for sl in tqdm(batches, position=rank):
        batch = data[sl]

        with GradientCollector(model, processor) as mgr:
            x, y = pad_and_tensor(
                batch["input_ids"],  # type: ignore
                labels=batch.get("labels", None),  # type: ignore
                device=model.device,
            )
            model(x, labels=y).loss.backward()
            model.zero_grad()

        grads = mgr.flattened_grads()
        if isinstance(data, MemmapDataset):
            if index is None:
                from faiss import IndexFlat

                index = IndexFlat(grads.shape[1])
            else:
                index.add(grads.cpu().float().numpy())  # type: ignore
        else:
            gradients.extend(grads.cpu().float().numpy())

    idx_path = path + f"/rank_{rank}.idx"
    if isinstance(data, Dataset):
        index = data.add_column(  # type: ignore
            name="gradient",
            column=gradients,
            feature=Sequence(Value("float32"), length=gradients[0].shape[-1]),
        )
        index.save_to_disk(idx_path)  # type: ignore

        assert isinstance(index, Dataset)
    else:
        # Save the index to disk
        import faiss

        faiss.write_index(index, idx_path)
        assert isinstance(index, faiss.IndexFlat)

    print(f"Saving index to {idx_path}")
    return index


def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    *,
    batches: list[slice] | None = None,
    max_documents: int | None = None,
) -> dict[str, AdafactorNormalizer]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    moments: dict[str, AdafactorNormalizer] = {}
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # If max_tokens is specified, randomly select a subset of batches
    elif max_documents is not None:
        batches = batches.copy()

        rng = random.Random(rank)
        rng.shuffle(batches)

    N = 0
    pbar = trange(max_documents or len(batches), position=rank)
    should_break = False

    def callback(name: str, g: torch.Tensor):
        nonlocal N, should_break

        n = len(g)
        N += n
        if max_documents:
            pbar.update(n)

            if N > max_documents:
                pbar.close()
                should_break = True
                return
        else:
            pbar.update(1)

        # We follow the tensor2tensor implementation of Adafactor, which
        # takes the mean rather than summing over the rows and columns.
        # row: mean over columns, shape [O]
        sq = g.square().sum(0)
        row_acc = sq.mean(dim=1)
        # col: mean over rows,    shape [I]
        col_acc = sq.mean(dim=0)

        if name not in moments:
            # initialize accumulators at zero
            moments[name] = AdafactorNormalizer(
                torch.zeros_like(row_acc),
                torch.zeros_like(col_acc),
            )

        # in‐place accumulate
        moments[name].row.add_(row_acc)
        moments[name].col.add_(col_acc)

    for sl in batches:
        batch = data[sl]

        with GradientCollector(model, closure=callback):
            x, y = pad_and_tensor(
                batch["input_ids"],  # type: ignore
                labels=batch.get("labels", None),  # type: ignore
                device=model.device,
            )
            model(x, labels=y).loss.backward()
            model.zero_grad()

        if should_break:
            break

    for name, moment in moments.items():
        # normalize by the number of examples
        moment.row.div_(N)
        moment.col.div_(N)

        if dist.is_initialized():
            dist.all_reduce(moment.row, op=dist.ReduceOp.AVG)
            dist.all_reduce(moment.col, op=dist.ReduceOp.AVG)

    return moments
