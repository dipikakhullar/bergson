from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from dataclasses import dataclass, field

import faiss
import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from tqdm.auto import trange

from bergson.approx_unrolling.model_checkpoints import ModelCheckpointManager
from bergson.data import MemmapDataset


class Normalizer(ABC):
    """
    Base class for normalizers that can be used to scale gradients.
    """

    @abstractmethod
    def normalize_(self, grad: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Normalize gradients in-place, adding a small epsilon to avoid division by zero.
        """


@dataclass
class AdafactorNormalizer(Normalizer):
    """
    Row and column sums of second moments of gradients for a matrix-valued parameter.
    """

    row: Tensor  # shape [O]
    col: Tensor  # shape [I]

    def __post_init__(self):
        assert self.row.ndim == 1, f"Expected 1D tensor for row, got {self.row.ndim}D"
        assert self.col.ndim == 1, f"Expected 1D tensor for col, got {self.col.ndim}D"

    @torch.compile
    def normalize_(self, grad: Tensor, eps: float = 1e-30) -> Tensor:
        """
        Normalize the row and column sums by adding a small epsilon.

        Note: Our `eps` corresponds to epsilon_1 in the original Adafactor paper. They
        recommend 1e-30, but we use 1e-16 for extra numerical stability.
        """
        # We follow the Adafactor implementation in the tensor2tensor repo, which is
        # different from the paper and from the PyTorch implementation. First add eps
        # to ensure these second moments are sufficiently far from zero. Then we don't
        # need to worry about numerical stability anywhere else, and we don't need to
        # materialize the outer product at any point.
        r, c = self.row.add(eps), self.col.add(eps)

        # This is the denominator for V, the rank-one matrix of second moment estimates:
        # V = torch.outer(r, c) / denom
        # V_ij = r_i * c_j / denom
        # But we want to (implicitly) take the Hadamard product with the elementwise
        # reciprocal square root of V:
        # (V_ij)^{-1/2} = denom.sqrt() * r_i.rsqrt() * c_j.rsqrt()
        denom = r.mean()

        # Hadamard product with a rank-one matrix ab^T is the same as left-multiplying
        # by diag(a) and right-multiplying by diag(b). In this case we can represent
        # the elementwise reciprocal square root of V as ab^T where:
        # a = denom.sqrt() * r.rsqrt() and b = c.rsqrt()
        a = denom.sqrt() * r.rsqrt_()  # shape [O]
        b = c.rsqrt_()

        # Implicitly do the Hadamard product
        grad *= a[:, None]  # [N, O] * [O] → [N, O]
        grad *= b[None, :]
        return grad

    def to_adam(self) -> "AdamNormalizer":
        """
        Convert this Adafactor normalizer to an Adam normalizer by materializing the
        rank-one second moment matrix.
        """
        # Compute the second moment matrix as a square matrix of shape [O, I]
        # NOTE: We don't add the epsilon here, since the AdamNormalizer is going to
        # add it outside the square root. This could cause infs though if there are
        # any exactly zero rows or columns, so we should be careful.
        avg_sq = torch.outer(self.row, self.col) / self.row.mean()
        return AdamNormalizer(avg_sq=avg_sq)


@dataclass
class AdamNormalizer(Normalizer):
    """
    Contains the second moments of the gradients.
    """

    avg_sq: Tensor

    @torch.compile
    def normalize_(self, grad: Tensor, eps: float = 1e-8) -> Tensor:
        """Normalize the gradients by the square root of the second moments."""
        # Adam-style epsilon is added outside the square root
        return grad.div_(self.avg_sq.sqrt().add_(eps))

    def to_adafactor(self) -> AdafactorNormalizer:
        """
        Convert this Adam normalizer to an Adafactor normalizer, minimizing the
        I-divergence (generalized Kullback-Leibler divergence) between the original
        and the factored second moments.
        """
        # We assume avg_sq is a square matrix of shape [O, I]
        assert self.avg_sq.ndim == 2, f"Expected 2D tensor for avg_sq, got {self.avg_sq.ndim}D"

        # Compute row and column means
        return AdafactorNormalizer(
            row=self.avg_sq.mean(dim=1),  # shape [O]
            col=self.avg_sq.mean(dim=0),  # shape [I]
        )


@torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported())
def build_index(
    checkpoint_manager: ModelCheckpointManager,
    data: Dataset | MemmapDataset,
    *,
    batch_size: int = 32,
    num_examples: int = 0,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    from faiss import IndexFlat

    index = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Use the entire dataset
    if num_examples <= 0:
        num_examples = len(data)
    else:
        num_examples = min(num_examples, len(data))

    chunk_size = num_examples // world_size

    # Shuffle the order in which we visit the batches just so that the progress bar
    # shows a valid estimate for the total time
    num_batches = num_examples // (batch_size * world_size)
    indices = np.random.permutation(num_batches)

    for i, segment in enumerate(checkpoint_manager.all_checkpoints):
        cache_dict = checkpoint_manager.load_cache(segment=i)
        lambda_eigenvalues = cache_dict["lambda_matrix"]
        activation_covariance = cache_dict["activation_eigenvectors"]
        gradient_covariance = cache_dict["gradient_eigenvectors"]

        for checkpoint in segment:
            model = checkpoint_manager.load_checkpoint(checkpoint, device="cuda")

            for i in trange(num_batches, position=rank):
                j = indices[i]
                batch = data[j * batch_size : (j + 1) * batch_size]

                with GradientCollector(
                    model=model,
                    checkpoint_manager=checkpoint_manager,
                    lambda_eigenvalues=lambda_eigenvalues,
                    activation_covariance=activation_covariance,
                    gradient_covariance=gradient_covariance,
                ) as mgr:
                    x = torch.tensor(batch["input_ids"], dtype=torch.long, device="cuda")
                    y = torch.tensor(batch["labels"], dtype=torch.long, device="cuda")
                    model(x, labels=y).loss.backward()
                    model.zero_grad()

                grads = mgr.flattened_grads()
                assert grads is not None, "No matrix-valued gradients found"

                if index is None:
                    index = IndexFlat(grads.shape[1])

                    # Figure out how much RAM we have
                    ram = psutil.virtual_memory().available
                    grad_size = grads[0].element_size() * grads[0].numel()

                    # Check if we can fit the index in RAM
                    chunk_size = min(chunk_size, ram // (grad_size * 2))

                    if rank == 0:
                        print(f"Total dataset size: {len(data):_}")
                        if num_examples < len(data):
                            print(f"Using only {num_examples:_} examples for the index")

                        print(f"RAM available: {ram / 2**30:.2f} GB")
                        print(f"Grad dimension: {grads.shape[1]}")
                        if chunk_size < num_examples:
                            print(f"Conservatively using chunk size of {chunk_size:_} examples")

                index.add(grads.cpu().float().numpy())  # type: ignore

    # Save the index to disk

    path = checkpoint_manager.model_dir / "index"
    path.mkdir(parents=True, exist_ok=True)
    idx_path = path / f"rank_{rank}.faiss"
    print(f"Saving index to {idx_path}")
    faiss.write_index(index, str(idx_path))


class ProjectionGenerator:
    """Wrapper around a torch.Generator that generates random projection matrices."""

    def __init__(self, device: torch.device, seed: int = 42):
        self.prng = torch.Generator(device).manual_seed(seed)

    def next_projection(self, p: int, q: int, o: int, i: int) -> tuple[Tensor, Tensor]:
        """
        Return the left and right random projection matrices of shape [p, o] and [q, i]
        """
        A = torch.randn(p, o, device=self.prng.device, generator=self.prng)
        B = torch.randn(q, i, device=self.prng.device, generator=self.prng)
        A /= A.norm(dim=1, keepdim=True)
        B /= B.norm(dim=1, keepdim=True)
        return A, B


@dataclass
class GradientCollector(ContextDecorator):
    """
    Adds forward and backward hooks to `model` that efficiently collect per-sequence
    gradients for all the matrix-valued parameters, randomly projecting them using a
    fixed seed to compress them into lower-dimensional blocks of shape [p×q]. We use
    a dictionary of `AdafactorNormalizer` to scale the gradients by the second moments
    of the parameters, which are expected to be precomputed and passed in.

    The collected gradients are flattened into a single tensor after the backward pass.
    You can access the flattened gradients via the `flat_grads` attribute after exiting
    the context manager.

    We assume that the input to `model` is of shape `[N, S, I]`, where `N` is the
    batch size, `S` is the sequence length, and `I` is the input dimension. We take the
    mean over the sequence length to obtain a single gradient per sequence.
    """

    model: nn.Module
    checkpoint_manager: ModelCheckpointManager
    lambda_eigenvalues: dict[str, Tensor] = field(default_factory=dict)
    activation_covariance: dict[str, Tensor] = field(default_factory=dict)
    gradient_covariance: dict[str, Tensor] = field(default_factory=dict)

    normalizers: dict[str, Normalizer] = field(default_factory=dict)
    """
    Dictionary of normalizers for each matrix-valued parameter in the model. The keys
    should match the names of the parameters in the model. If a parameter does not have
    a normalizer, it will be skipped.
    """

    p: int = 16
    """Number of rows in the projection matrix."""

    q: int = 16
    """Number of columns in the projection matrix."""

    seed: int = 42
    """Random seed used for generating the projection matrices."""

    eps: float = 1e-8
    """Epsilon value used for numerical stability in normalization."""

    def __post_init__(self):
        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        # We actually take advantage of the fact that modern Python dicts are ordered
        # so that we can both keep track of the order in which the hooks are called
        # and also use the names of the layers as keys for the normalizers.
        self._buffers: dict[str, Tensor] = {}

    def __enter__(self):
        # install a hook on every Linear

        assert self.checkpoint_manager.module_keys is not None, "Module keys must be set in the checkpoint manager."
        for (
            name,
            layer,
        ) in self.model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue

            if name not in self.checkpoint_manager.module_keys:
                # Skip layers that are not in the checkpoint manager's module keys
                continue

            # Save the name of the layer for later use
            layer._name = name  # type: ignore[attr-defined]

            layer._lambda = self.lambda_eigenvalues[name].to(layer.weight.device)
            layer._activation_eigenvectors = self.activation_covariance[name].to(layer.weight.device)
            layer._gradient_eigenvectors = self.gradient_covariance[name].to(layer.weight.device)

            # register forward hook to save V = X @ B^T
            fwd_hook = layer.register_forward_hook(self._save_input)
            self._fwd_hooks.append(fwd_hook)

            # register backward hook to compute P = mean(U @ V^T)
            bwd_hook = layer.register_full_backward_hook(self._collect_and_transform)
            self._bwd_hooks.append(bwd_hook)

        return self

    def _save_input(self, module: nn.Module, inp: tuple, _):
        # x: [N, S, I]
        x = inp[0].detach()
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        module._inputs = x

    # @torch.compile
    def _collect_and_transform(self, module, _, grad_out):
        if module.bias is None:
            raise RuntimeError(
                f"Module {module._name} does not have a bias. Cannot compute gradients with current EK_FAC implementation."
            )

        G_out = grad_out[0]  # [N, S, O] - gradients w.r.t. outputs
        X = module._inputs  # [N, S, I] - saved inputs

        # Compute full gradient matrix for each sequence
        # G_out: [N, S, O], X: [N, S, I] -> G: [N, O, I]
        G_weights = G_out.transpose(-2, -1) @ X  # [N, O, S] @ [N, S, I] -> [N, O, I]

        G_bias = G_out.sum(dim=-2).unsqueeze(-1)  # [N, O, 1] - sum over sequence length

        G = torch.cat([G_weights, G_bias], dim=-1)

        # Apply custom transformation: A @ G @ B.T
        if (
            module._activation_eigenvectors is not None
            and module._gradient_eigenvectors is not None
            and module._lambda is not None
        ):
            # G: [N, O, I], A: [O, O], B: [I, I]
            # Result: [N, O, I]

            try:
                gradient = torch.matmul(
                    module._gradient_eigenvectors.t(),
                    torch.matmul(G, module._activation_eigenvectors),
                )
                gradient.mul_(module._lambda)
                gradient = torch.matmul(
                    module._gradient_eigenvectors,
                    torch.matmul(G, module._activation_eigenvectors.t()),
                )
            except RuntimeError as e:
                print("-" * 60)
                print(
                    f"{G.shape}, {module._gradient_eigenvectors.shape}, {module._activation_eigenvectors.shape}, {module._lambda.shape} "
                )
                raise RuntimeError(f"Error transforming gradients for layer {module._name}: {e}")

        # Store the transformed gradients
        # Shape is [N, O, I] or [N, O, I] after transformation
        self._buffers[module._name] = G

    def __exit__(self, exc_type, exc, tb):
        # clean up secret attributes

        for name, layer in self.model.named_modules():
            attrs_to_remove = [
                "_lambda",
                "_activation_cov",
                "_gradient_cov",
                "_inputs",
                "_name",
            ]
            for attr in attrs_to_remove:
                if hasattr(layer, attr):
                    delattr(layer, attr)

        for h in self._fwd_hooks + self._bwd_hooks:
            h.remove()
        # clean up hooks
        for h in self._fwd_hooks:
            h.remove()
        for h in self._bwd_hooks:
            h.remove()

        return False

    def flattened_grads(self) -> Tensor:
        """
        Returns the flattened gradients collected during the context manager.
        """
        # concatenate all the flattened [N, p*q] chunks → [N, total]
        return torch.cat([buf.flatten(1) for buf in self._buffers.values()], dim=1)

    def get_layer_grads(self, layer_name: str) -> Tensor:
        """Get transformed gradients for a specific layer. Shape: [N, O, I]"""
        if layer_name not in self._buffers:
            raise KeyError(f"No gradients collected for layer '{layer_name}'")
        return self._buffers[layer_name]

    def get_all_layer_grads(self) -> dict[str, Tensor]:
        """Get transformed gradients for all layers as a dictionary."""
        return dict(self._buffers)
