import json
import os
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from datasets import Dataset
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from tqdm.auto import trange
from transformers import PreTrainedModel

from .data import MemmapDataset

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}


class Normalizer(ABC):
    """
    Base class for normalizers that can be used to scale gradients.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the NORMALIZER_TYPES dict."""
        super().__init_subclass__(**kwargs)
        NORMALIZER_TYPES[cls.__name__] = cls

    @staticmethod
    def from_state_dict(state_dict: dict[str, str | Tensor]) -> "Normalizer":
        """
        Create a normalizer instance from a state dictionary.
        The state dictionary should contain the class name and the tensors.
        """
        class_name = state_dict.pop("__class__")
        assert isinstance(class_name, str), "Expected '__class__' to be a string"

        if (cls := NORMALIZER_TYPES.get(class_name)) is None:
            raise ValueError(f"Unknown normalizer class: '{class_name}'")

        return cls(**state_dict)

    @abstractmethod
    def normalize_(self, grad: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Normalize gradients in-place, adding a small epsilon to avoid division by zero.
        """

    def state_dict(self) -> dict[str, str | Tensor]:
        """
        Return the state of the normalizer as a dictionary of tensors.
        This is used for saving and loading the normalizer.
        """
        tensors = {k: v for k, v in self.__dict__.items() if isinstance(v, Tensor)}
        return {
            "__class__": self.__class__.__name__,
            **tensors,
        }


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
        assert (
            self.avg_sq.ndim == 2
        ), f"Expected 2D tensor for avg_sq, got {self.avg_sq.ndim}D"

        # Compute row and column means
        return AdafactorNormalizer(
            row=self.avg_sq.mean(dim=1),  # shape [O]
            col=self.avg_sq.mean(dim=0),  # shape [I]
        )


@dataclass
class GradientProcessor:
    """Configuration for processing and compressing gradients."""

    normalizers: Mapping[str, Normalizer] = field(default_factory=dict)
    """
    Dictionary of normalizers for each matrix-valued parameter in the model. The keys
    should match the names of the parameters in the model. If a parameter does not have
    a normalizer, it will be skipped.
    """

    preconditioners: Mapping[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of preconditioners for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    projection_dim: int | None = None
    """Number of rows and columns to project the gradients to. If `None`, keep the
    original shape of the gradients."""

    projection_seed: int = 42
    """Seed for generating the random projection matrices."""

    def estimate_preconditioners(
        self,
        model: PreTrainedModel,
        data: Dataset | MemmapDataset,
        num_examples: int = 1000,
    ):
        """
        Estimate preconditioners from data. Overwrites the `preconditioners` field.
        """
        preconditioners = {}
        rank = dist.get_rank() if dist.is_initialized() else 0

        for i in trange(num_examples, position=rank):
            example = send_to_device(data[i], model.device)

            x = torch.as_tensor(example["input_ids"], device=model.device).unsqueeze(0)
            with GradientCollector(model, self) as mgr:
                model(x, labels=x).loss.backward()
                model.zero_grad()

            for name, g in mgr.collected_grads.items():
                if g is None or g.numel() == 0:
                    continue

                # Skip vector-valued parameters since they are negligible
                if g.ndim < 2:
                    continue

                # Compute the outer product of the flattened gradient
                g = g.flatten()
                preconditioner = preconditioners.get(name, None)
                if preconditioner is None:
                    preconditioners[name] = torch.outer(g, g) / num_examples
                else:
                    preconditioner.addmm_(g[:, None], g[None], alpha=1 / num_examples)

        # Sanity check
        assert preconditioners, "num_examples must be > 0"

        # Reduce the preconditioners across processes if needed
        if dist.is_initialized():
            for preconditioner in preconditioners.values():
                dist.all_reduce(preconditioner)
                preconditioner /= dist.get_world_size()

        self.preconditioners = preconditioners

    @classmethod
    def load(
        cls,
        path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "GradientProcessor":
        """
        Load the normalizers and preconditioners from a file.
        """
        cfg_path = os.path.join(path, "processor_config.json")
        norm_path = os.path.join(path, "normalizers.pth")
        precond_path = os.path.join(path, "preconditioners.pth")

        # Load configuration
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Load normalizers
        norm_state = torch.load(
            norm_path,
            map_location=map_location,
            weights_only=True,
        )
        normalizers = {
            name: Normalizer.from_state_dict(state)
            for name, state in norm_state.items()
        }

        return cls(
            normalizers=normalizers,
            preconditioners=torch.load(
                precond_path,
                map_location=map_location,
                weights_only=True,
            ),
            projection_dim=cfg.get("projection_dim"),
            projection_seed=cfg.get("projection_seed", 42),
        )

    def save(self, path: str):
        """
        Save the normalizers and preconditioners to a file.
        """
        os.makedirs(path, exist_ok=True)

        cfg_path = os.path.join(path, "processor_config.json")
        norm_path = os.path.join(path, "normalizers.pth")
        precond_path = os.path.join(path, "preconditioners.pth")

        # Save configuration separately
        cfg = asdict(self)
        del cfg["normalizers"]
        del cfg["preconditioners"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Save normalizers
        norm_state = {
            name: normalizer.state_dict()
            for name, normalizer in self.normalizers.items()
        }
        torch.save(norm_state, norm_path)
        torch.save(self.preconditioners, precond_path)


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

    processor: GradientProcessor = field(default_factory=GradientProcessor)
    """Configuration for processing and compressing gradients."""

    closure: Callable | None = None
    """Closure to call on the gradient as it is collected. If provided, we will not
    store the gradient after the closure is called."""

    def __post_init__(self):
        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        # We actually take advantage of the fact that modern Python dicts are ordered
        # so that we can both keep track of the order in which the hooks are called
        # and also use the names of the layers as keys for the normalizers.
        self.collected_grads: dict[str, Tensor] = {}

    def __enter__(self):
        generator = None

        # install a hook on every Linear
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue

            if generator is None:
                generator = ProjectionGenerator(
                    layer.weight.device,
                    seed=self.processor.projection_seed,
                )

            # Save the name of the layer for later use
            layer._name = name  # type: ignore[attr-defined]

            o, i = layer.out_features, layer.in_features
            p = self.processor.projection_dim

            if p is None:
                # TODO: Make this more efficient, don't actually materialize eye
                A, B = torch.eye(o, device=layer.weight.device), torch.eye(
                    i, device=layer.weight.device
                )
            else:
                A, B = generator.next_projection(p, p, o, i)

            if norm := self.processor.normalizers.get(name):
                # In the case of Adafactor, we can normalize the projection matrices
                # directly because the normalizer matrix is rank-1.
                if isinstance(norm, AdafactorNormalizer):
                    # Compare to the normalize_ method in AdafactorNormalizer
                    r, c = norm.row.add(1e-30), norm.col.add(1e-30)
                    denom = r.mean()

                    a, b = denom.sqrt() * r.rsqrt_(), c.rsqrt_()
                    A *= a.unsqueeze(0)
                    B *= b.unsqueeze(0)

                # In the case of Adam, we need to use a slower code path
                elif isinstance(norm, AdamNormalizer):
                    layer._exp_avg_sq = norm.avg_sq
                else:
                    raise ValueError(f"Unsupported normalizer type: {type(norm)}")

            layer._A_proj = A
            layer._B_proj = B

            # register forward hook to save V = X @ B^T
            fwd_hook = layer.register_forward_hook(self._save_input)
            self._fwd_hooks.append(fwd_hook)

            # register backward hook to compute P = sum(U @ V^T)
            bwd_hook = layer.register_full_backward_hook(self._process_grad)
            self._bwd_hooks.append(bwd_hook)

        return self

    def _save_input(self, module: nn.Module, inp: tuple, _):
        """Save the input to the module for later use in the backward pass."""
        x = inp[0].detach()
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        module._inputs = x

    def _process_grad(self, module, _, grad_out):
        """Process the incoming gradient wrt the output of the module."""
        G = grad_out[0]  # [N, S, O]

        # Slow code path for Adam
        if hasattr(module, "_exp_avg_sq"):
            # Materialize the full gradient for every sequence in the batch
            P = G.mT @ module._inputs  # [N, O, S] @ [N, S, I] → [N, O, I]

            # Normalize the gradients using the second moment matrix
            P /= module._exp_avg_sq.sqrt().add_(1e-8)

            # Project the gradients to the lower-dimensional space
            P = module._A_proj @ P @ module._B_proj.T  # [N, p, q]
        else:
            # With Adafactor, we can immediately project the incoming gradients and the
            # saved inputs to the lower-dimensional space. This makes the outer product
            # we're about to do much cheaper and more memory-efficient.
            V = module._inputs @ module._B_proj.T  # [N, S, q]
            U = G @ module._A_proj.T  # [N, S, p]

            # The gradient for each token is an outer product. The gradient for a whole
            # sequence is the sum of these outer products, which is equivalent to a
            # matrix multiplication contracting along the sequence axis S.
            # TODO: This approach will not work when we start supporting reduction along
            # documents of variable length inside each "sequence."
            P = U.mT @ V  # [N, p, S] @ [N, S, q] → [N, p, q]

        if self.closure is not None:
            self.closure(module._name, P)
        else:
            self.collected_grads[module._name] = P

        # Save memory ASAP
        del module._inputs

    def __exit__(self, exc_type, exc, tb):
        # clean up secret attributes
        for layer in self.model.modules():
            if hasattr(layer, "_A_proj"):
                del layer._A_proj
            if hasattr(layer, "_B_proj"):
                del layer._B_proj
            if hasattr(layer, "_exp_avg_sq"):
                del layer._exp_avg_sq
            if hasattr(layer, "_inputs"):
                del layer._inputs
            if hasattr(layer, "_name"):
                del layer._name

        # clean up hooks
        for h in self._fwd_hooks:
            h.remove()
        for h in self._bwd_hooks:
            h.remove()

        return False

    def flattened_grads(self) -> Tensor:
        """Concatenate and flatten all the collected gradients into a single tensor."""
        # concatenate all the flattened [N, p*q] chunks → [N, total]
        return torch.cat(
            [buf.flatten(1) for buf in self.collected_grads.values()], dim=1
        )
