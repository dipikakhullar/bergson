import json
import os
from dataclasses import asdict, dataclass, field
from typing import Mapping

import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from datasets import Dataset
from torch import Tensor
from tqdm.auto import trange
from transformers import PreTrainedModel

from bergson.data import MemmapDataset
from bergson.gradients import GradientCollector, Normalizer

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}


@dataclass
class CovarianceProcessor:
    """Configuration for processing and compressing gradients."""

    normalizers: Mapping[str, Normalizer] = field(default_factory=dict)
    """
    Dictionary of normalizers for each matrix-valued parameter in the model. The keys
    should match the names of the parameters in the model. If a parameter does not have
    a normalizer, it will be skipped.
    """

    gradient_covariance: Mapping[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of preconditioners for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    activation_covariance: Mapping[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of preconditioners for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    sampling_seed: int = 42
    """Seed for generating the random projection matrices."""

    def compute_covariances(
        self,
        model: PreTrainedModel,
        data: Dataset | MemmapDataset,
        num_examples: int = 1000,
    ):
        """
        Estimate preconditioners from data. Overwrites the `preconditioners` field.
        """
        gradient_covariances = {}
        activation_covariances = {}
        rank = dist.get_rank() if dist.is_initialized() else 0

        for i in trange(num_examples, position=rank):
            example = send_to_device(data[i], model.device)

            x = torch.as_tensor(example["input_ids"], device=model.device).unsqueeze(0)
            with GradientCollector(model, self) as mgr:
                pass
                logits = model(x).logits
                sampled_loss = logits.sum()
                sampled_loss.backward()

                model.zero_grad()

            for name, g in mgr.collected_grads.items():
                if g is None or g.numel() == 0:
                    continue

                # Skip vector-valued parameters since they are negligible
                # TODO: Make this use named module parameters
                if g.ndim < 2:
                    continue

                # Compute the outer product of the flattened gradient

                gradient_covariance = gradient_covariances.get(name, None)
                activation_covariance = activation_covariances.get(name, None)

                if gradient_covariance is None:
                    gradient_covariances[name] = torch.outer(g, g) / num_examples
                else:
                    assert isinstance(gradient_covariance, Tensor), (
                        f"Invalid type for gradient_covariance: {type(gradient_covariance)}"
                    )
                    gradient_covariance.addmm_(g[:, None], g[None], alpha=1 / num_examples)

                if activation_covariance is None:
                    activation_covariances[name] = torch.outer(g, g) / num_examples
                else:
                    activation_covariance.addmm_(g[:, None], g[None], alpha=1 / num_examples)

        # Sanity check
        assert activation_covariances, "num_examples must be > 0"
        assert gradient_covariances, "num_examples must be > 0"

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
    ) -> "CovarianceProcessor":
        """
        Load the normalizers and preconditioners from a file.
        """
        cfg_path = os.path.join(path, "processor_config.json")
        norm_path = os.path.join(path, "normalizers.pth")
        gradient_covariance_path = os.path.join(path, "gradient_covariance.safetensors")
        activation_covariance_path = os.path.join(path, "activation_covariance.safetensors")

        # Load configuration
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Load normalizers
        norm_state = torch.load(
            norm_path,
            map_location=map_location,
            weights_only=True,
        )
        normalizers = {name: Normalizer.from_state_dict(state) for name, state in norm_state.items()}

        return cls(
            normalizers=normalizers,
            gradient_covariance=torch.load(
                gradient_covariance_path,
                map_location=map_location,
                weights_only=True,
            ),
            activation_covariance=torch.load(
                activation_covariance_path,
                map_location=map_location,
                weights_only=True,
            ),
            sampling_seed=cfg.get("sampling_seed", 42),
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
        norm_state = {name: normalizer.state_dict() for name, normalizer in self.normalizers.items()}
        torch.save(norm_state, norm_path)
        torch.save(self.preconditioners, precond_path)
