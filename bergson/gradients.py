import hashlib
import json
import os
import random
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Mapping
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset
from natsort import natsorted
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from tqdm.auto import trange
from transformers import PreTrainedModel

from .data import MemmapDataset, pad_and_tensor, to_nested_tensor

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
    def normalize_(
        self,
        grad: Tensor,
        fisher_fourth_root: bool = False,
        eps: float = 1e-8,
    ) -> Tensor:
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
    def normalize_(
        self,
        grad: Tensor,
        fisher_fourth_root: bool = False,
        eps: float = 1e-30,
    ) -> Tensor:
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
        if fisher_fourth_root:
            a = denom.pow(0.25) * r.pow(-0.25)
            b = c.pow(-0.25)
        else:
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
    def normalize_(
        self,
        grad: Tensor,
        fisher_fourth_root: bool = False,
        eps: float = 1e-8,
    ) -> Tensor:
        """Normalize the gradients by the square root of the second moments."""
        # Adam-style epsilon is added outside the square root
        denom = self.avg_sq.pow(0.25) if fisher_fourth_root else self.avg_sq.sqrt()
        return grad.div_(denom.add_(eps))

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

    fisher_fourth_root: bool = False
    """
    Whether to use the fourth root of the inverse Fisher information matrix when
    normalizing gradients. This means any inner product between normalized gradients
    will implicitly use the square root of the inverse Fisher, rather than the inverse
    Fisher itself.
    """

    projection_dim: int | None = None
    """Number of rows and columns to project the gradients to. If `None`, keep the
    original shape of the gradients."""

    def estimate_preconditioners(
        self,
        model: PreTrainedModel,
        data: Dataset | MemmapDataset,
        *,
        batches: list[slice] | None = None,
        max_documents: int | None = None,
        target_modules: set[str] | None = None,
    ):
        """
        Estimate preconditioners from data. Overwrites the `preconditioners` field.
        """
        preconditioners = {}
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

        def callback(name: str, g: torch.Tensor):
            # Compute the outer product of the flattened gradient
            g = g.flatten(1).float()
            preconditioner = preconditioners.get(name, None)
            if preconditioner is None:
                preconditioners[name] = g.mT @ g
            else:
                preconditioner.addmm_(g.mT, g)

        N = 0
        total = (max_documents or len(batches)) // world_size
        pbar = trange(total, disable=rank != 0, desc="Estimating preconditioners")

        for sl in batches:
            batch = data[sl]

            # Update progress
            n = len(range(*sl.indices(len(data))))
            pbar.update(n)

            N += n
            if total and N >= total:
                break

            with GradientCollector(
                model.base_model,
                self,
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

        # Sanity check
        pbar.close()
        assert preconditioners, "num_examples must be > 0"

        for name, preconditioner in preconditioners.items():
            preconditioner /= N  # Normalize by the number of examples

            # Reduce the preconditioners across processes if needed
            if dist.is_initialized():
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
    """Object that deterministically yields pseudorandom projection matrices."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ):
        self.dtype = dtype
        self.history: dict[tuple[str, int, int], torch.Tensor] = {}
        self.prng = torch.Generator(device).manual_seed(0)

        # Check that the PRNG implementation has not changed from what we expect.
        # If this test fails, we may need to start using NumPy's PRNG instead, or
        # explicitly save the random projection matrices.
        res = torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            (),
            generator=self.prng,
            device=device,
        )
        expected = {
            "cpu": 900450186894289456,
            "cuda": 7361108121267848589,
        }
        if (val := expected.get(device.type)) is not None:
            assert res == val, f"Expected PRNG to return {val} for seed 0, got {res}"
        else:
            raise ValueError(f"Unsupported device type: {device.type}")

    def projection(
        self,
        name: str,
        m: int,
        n: int,
        side: Literal["left", "right"],
    ) -> Tensor:
        """Return the `side` projection matrix for parameter `name` of shape [m, n]."""
        # Seed the PRNG with the name of the layer and what "side" we are projecting
        message = bytes(f"{name}/{side}", "utf-8")
        digest = hashlib.md5(message).digest()
        seed = int.from_bytes(digest, byteorder="big") % (2**63 - 1)

        A = torch.randn(
            m,
            n,
            device=self.prng.device,
            dtype=self.dtype,
            generator=self.prng.manual_seed(seed),
        )
        A /= A.norm(dim=1, keepdim=True)
        return A


@dataclass
class GradientCollector(ContextDecorator):
    """
    Adds forward and backward hooks to `model` that efficiently collect per-sequence
    gradients for all the matrix-valued parameters, randomly projecting them using a
    fixed seed to compress them into lower-dimensional blocks of shape [p×q]. We use
    a dictionary of `AdafactorNormalizer` to scale the gradients by the second moments
    of the parameters, which are expected to be precomputed and passed in.

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

    target_modules: set[str] | None = None
    """
    List of parameter names to collect gradients for. Should consist only of weight
    matrices in `nn.Linear` modules. If `None`, the gradients for all weight matrices
    will be collected.
    """
    
    sequence_lengths: torch.Tensor | None = None
    """
    Sequence lengths of valid elements for the current batch. If provided, we will
    convert the gradients to nested tensors.
    """
    
    move_to_cpu: bool = False
    """
    Whether to collect gradients on CPU. This will use non-blocking transfers.
    """

    def __post_init__(self):
        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        self.collected_grads: dict[str, Tensor] = {}
        self.generator: ProjectionGenerator | None = None

    def __enter__(self):
        # install a hook on every Linear
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue

            if self.target_modules is not None and name not in self.target_modules:
                continue

            dtype = layer.weight.dtype
            if not dtype.is_floating_point:
                # bitsandbytes uses fp16 as the computation dtype
                dtype = torch.float16

            if self.generator is None:
                self.generator = ProjectionGenerator(
                    layer.weight.device,
                    dtype,
                )

            # Save the name of the layer for later use
            layer._name = name  # type: ignore[attr-defined]

            # register forward hook to save V = X @ B^T
            fwd_hook = layer.register_forward_hook(partial(self._save_input, sequence_lengths=self.sequence_lengths))
            self._fwd_hooks.append(fwd_hook)

            # register backward hook to compute P = sum(U @ V^T)
            bwd_hook = layer.register_full_backward_hook(partial(self._process_grad, sequence_lengths=self.sequence_lengths))
            self._bwd_hooks.append(bwd_hook)

        return self

    @torch.no_grad
    def _save_input(self, module: nn.Module, inp: tuple, _, sequence_lengths: Tensor | None = None):
        """Save the input to the module for later use in the backward pass."""        
        x = inp[0].detach()
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"
        
        x = to_nested_tensor(x, sequence_lengths)

        # Pre-scale the input by the Adafactor column stats
        norm = self.processor.normalizers.get(module._name)
        if isinstance(norm, AdafactorNormalizer):
            b = norm.col.add(1e-30)
            if self.processor.fisher_fourth_root:
                b.pow_(-0.25)
            else:
                b.rsqrt_()

            x = x * b.type_as(x)  # [N, S, I] * [I] → [N, S, I]

        # If we're not using AdamNormalizer, we can randomly project the input here
        # to save memory, rather than waiting until the backward pass.
        p = self.processor.projection_dim
        if p is not None and not isinstance(norm, AdamNormalizer):
            assert self.generator is not None, "Generator must be initialized"

            i = module.in_features
            x = x @ self.generator.projection(module._name, p, i, "right").T

        module._inputs = x

    @torch.no_grad
    def _process_grad(self, module: nn.Module, _, grad_out, sequence_lengths: Tensor | None = None):
        """Process the incoming gradient wrt the output of the module."""
        # Sanity checks
        assert isinstance(module, nn.Linear), "Expected a Linear module"
        assert self.generator is not None, "Generator must be initialized"
        G = grad_out[0]  # [N, S, O]
        I = module._inputs  # [N, S, I/q]

        # Save memory ASAP
        del module._inputs
        
        G = to_nested_tensor(G, sequence_lengths)

        p = self.processor.projection_dim
        o, i = module.out_features, module.in_features

        # Pre-scale G by the Adafactor row statistics
        norm = self.processor.normalizers.get(module._name)
        if isinstance(norm, AdafactorNormalizer):
            # Compare to the normalize_ method in AdafactorNormalizer
            r = norm.row.add(1e-30)

            if self.processor.fisher_fourth_root:
                a = r.mean().pow(0.25) * r.pow(-0.25)
            else:
                a = r.mean().sqrt() * r.rsqrt_()

            G = G * a.type_as(G)  # [N, S, O] * [O] → [N, S, O]

        # For Adam, we need to materialize the full gradient and then project
        if isinstance(norm, AdamNormalizer):
            P = G.mT @ I  # [N, O, S] @ [N, S, I] → [N, O, I]

            # Normalize the gradients using the second moment matrix
            P /= norm.avg_sq.sqrt().add_(1e-8)

            # Project the gradients to the lower-dimensional space
            if p is not None:
                A = self.generator.projection(module._name, p, o, "left")
                B = self.generator.projection(module._name, p, i, "right")
                P = A @ P @ B.T  # [N, p, q]
                del A, B

        # Both Adafactor and no normalizer, we can project G first
        else:
            if p is not None:
                A = self.generator.projection(module._name, p, o, "left")
                G = G @ A.T  # [N, S, p]
                del A

            P = G.mT @ I  # [N, O/p, S] @ [N, S, I/q] → [N, O/p, I/q]
        
        del G, I

        torch.cuda.empty_cache()
        if self.closure is not None:
            self.closure(module._name, P)
        else:
            if self.move_to_cpu:
                P = P.to(device="cpu", non_blocking=True)
            self.collected_grads[module._name] = P

    def __exit__(self, exc_type, exc, tb):
        # clean up secret attributes
        for layer in self.model.modules():
            if hasattr(layer, "_inputs"):
                del layer._inputs
            if hasattr(layer, "_name"):
                del layer._name

        # clean up hooks
        for h in self._fwd_hooks:
            h.remove()
        for h in self._bwd_hooks:
            h.remove()

        # Naturally sort the collected gradients by name
        self.collected_grads = {
            k: self.collected_grads[k] for k in natsorted(self.collected_grads)
        }
        return False

    def flattened_grads(self) -> Tensor:
        """Concatenate and flatten all the collected gradients into a single tensor."""
        # concatenate all the flattened [N, p*q] chunks → [N, total]
        return torch.cat(
            [buf.flatten(1) for buf in self.collected_grads.values()], dim=1
        )
