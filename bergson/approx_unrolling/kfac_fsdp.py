#!/usr/bin/env python3
"""
Minimal Working Example: FSDP2 + K-FAC Covariance Estimation

This script demonstrates:
1. Setting up a toy transformer model
2. Applying FSDP2 sharding
3. Collecting activations and gradients for K-FAC
4. Computing and saving covariance matrices using DTensor

Run with: torchrun --nproc_per_node=2 fsdp_kfac_mwe.py
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from tqdm import trange

# =============================================================================
# Toy Model Definition
# =============================================================================


class ToyTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Simple transformer block (no real attention)
        x = x + self.attn(self.norm1(x))
        x = x + self.ff2(torch.relu(self.ff1(self.norm2(x))))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_model: int = 512, n_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ToyTransformerBlock(d_model) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# =============================================================================
# K-FAC Covariance Collector
# =============================================================================


@dataclass
class KFACCollector:
    """Collects activations and output gradients for K-FAC covariance estimation."""

    model: nn.Module
    device_mesh: Optional[DeviceMesh] = None

    def __post_init__(self):
        self.activation_hooks = []
        self.gradient_hooks = []
        self.activations = {}
        self.output_gradients = {}

    def __enter__(self):
        # Register hooks on all Linear layers
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Forward hook to capture activations
            def save_activation(name=name):
                def hook(module, input, output):
                    # Store input activations
                    x = input[0].detach()  # [batch_size, seq_len, input_dim]
                    self.activations[name] = x

                return hook

            # Backward hook to capture output gradients
            def save_output_grad(name=name):
                def hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        # Store output gradients
                        grad = grad_output[0].detach()  # [batch_size, seq_len, output_dim]
                        self.output_gradients[name] = grad

                return hook

            fwd_handle = module.register_forward_hook(save_activation())
            bwd_handle = module.register_full_backward_hook(save_output_grad())

            self.activation_hooks.append(fwd_handle)
            self.gradient_hooks.append(bwd_handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up hooks
        for handle in self.activation_hooks + self.gradient_hooks:
            handle.remove()
        return False

    def compute_covariances(self) -> tuple[Dict[str, DTensor], Dict[str, DTensor]]:
        """Compute A and G covariance matrices for all Linear layers."""

        A_matrices = {}
        G_matrices = {}

        for name in self.activations.keys():
            if name not in self.output_gradients:
                continue

            # Get activations and gradients
            acts = self.activations[name]  # [batch, seq, input_dim]
            grads = self.output_gradients[name]  # [batch, seq, output_dim]

            # Flatten batch and sequence dimensions
            acts_flat = acts.reshape(-1, acts.shape[-1])  # [batch*seq, input_dim]
            grads_flat = grads.reshape(-1, grads.shape[-1])  # [batch*seq, output_dim]

            # Compute full covariance matrices
            A_full = torch.mm(acts_flat.T, acts_flat) / acts_flat.shape[0]  # [input_dim, input_dim]
            G_full = torch.mm(grads_flat.T, grads_flat) / grads_flat.shape[0]  # [output_dim, output_dim]

            # Convert to DTensor (sharded) if device_mesh is available
            if self.device_mesh is not None:
                A_dt = self._to_dtensor(A_full)
                G_dt = self._to_dtensor(G_full)
                A_matrices[name] = A_dt
                G_matrices[name] = G_dt
            else:
                A_matrices[name] = A_full
                G_matrices[name] = G_full

        return A_matrices, G_matrices

    def _to_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """Convert tensor to sharded DTensor."""
        return DTensor.from_local(tensor, device_mesh=self.device_mesh, placements=[Shard(0)], run_check=False)


# =============================================================================
# K-FAC Covariance Estimator
# =============================================================================


@dataclass
class KFACEstimator:
    """Estimates K-FAC covariance matrices over multiple examples."""

    model: nn.Module
    device_mesh: Optional[DeviceMesh] = None
    activation_covs: Dict[str, DTensor] = field(default_factory=dict)
    gradient_covs: Dict[str, DTensor] = field(default_factory=dict)

    def estimate(self, dataloader, num_examples: int = 100):
        """Estimate covariances over multiple examples."""

        rank = dist.get_rank() if dist.is_initialized() else 0
        examples_processed = 0

        with trange(num_examples, desc=f"Rank {rank}", position=rank) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if examples_processed >= num_examples:
                    break

                # Move to device
                if torch.cuda.is_available():
                    batch = batch.cuda()

                # Collect activations and gradients
                with KFACCollector(self.model, self.device_mesh) as collector:
                    # Forward + backward pass
                    output = self.model(batch)
                    loss = output.sum()  # Dummy loss
                    loss.backward()

                    # Compute covariances for this example
                    A_batch, G_batch = collector.compute_covariances()

                # Accumulate covariances
                for name in A_batch.keys():
                    if name not in self.activation_covs:
                        self.activation_covs[name] = A_batch[name] / num_examples  # type:ignore type checker thinks dividing by num_examples is casts DTensor to Tensor
                        self.gradient_covs[name] = G_batch[name] / num_examples  # type:ignore
                    else:
                        self.activation_covs[name] += A_batch[name] / num_examples  # type:ignore
                        self.gradient_covs[name] += G_batch[name] / num_examples  # type:ignore

                examples_processed += 1
                pbar.update(1)

                # Clear gradients
                self.model.zero_grad()

    def save_covariances(self, path: str):
        """Save covariance matrices to disk."""
        os.makedirs(path, exist_ok=True)

        rank = dist.get_rank() if dist.is_initialized() else 0
        is_distributed = dist.is_initialized()

        if rank == 0:
            print(f"Saving covariances to {path}...")

        # ALL RANKS must participate in DTensor.full_tensor() calls
        A_full = {}
        G_full = {}

        for name in self.activation_covs.keys():
            if isinstance(self.activation_covs[name], DTensor):
                # All ranks participate in the all-gather
                A_full[name] = self.activation_covs[name].full_tensor().cpu()
                G_full[name] = self.gradient_covs[name].full_tensor().cpu()
            else:
                A_full[name] = self.activation_covs[name].cpu()
                G_full[name] = self.gradient_covs[name].cpu()

        # Only rank 0 saves to disk
        if rank == 0:
            torch.save(A_full, os.path.join(path, "activation_covariances.pth"))
            torch.save(G_full, os.path.join(path, "gradient_covariances.pth"))
            print(f"Saved {len(A_full)} activation and gradient covariance matrices")

        # Ensure all ranks finish before continuing
        if is_distributed:
            dist.barrier()


# =============================================================================
# Main Training Script
# =============================================================================


def setup_distributed():
    """Initialize distributed training."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True
    return False


def create_toy_data(batch_size: int = 8, seq_len: int = 128, vocab_size: int = 1000):
    """Create toy training data."""
    return torch.randint(0, vocab_size, (batch_size, seq_len))


def main():
    print("Starting FSDP2 + K-FAC MWE...")

    # Setup
    is_distributed = setup_distributed()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    # Create model
    model = ToyTransformer(vocab_size=1000, d_model=256, n_layers=2)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Apply FSDP2 if distributed
    device_mesh = None
    if is_distributed:
        world_size = dist.get_world_size()
        device_mesh = DeviceMesh("cuda", list(range(world_size)))

        print(f"Applying FSDP2 with {world_size} GPUs...")

        # Apply FSDP2 bottom-up
        for layer in model.layers:
            assert isinstance(layer.attn, nn.Linear), "Layer must be a Linear module"
            assert isinstance(layer.ff1, nn.Linear), "Layer must be a Linear module"
            assert isinstance(layer.ff2, nn.Linear), "Layer must be a Linear module"
            fully_shard(layer.attn)
            fully_shard(layer.ff1)
            fully_shard(layer.ff2)
            fully_shard(layer)

        fully_shard(model.embedding)
        fully_shard(model.output)
        fully_shard(model)

    # Move to device
    model = model.to(device)

    # Create estimator
    estimator = KFACEstimator(model, device_mesh)

    # Create toy dataloader
    def toy_dataloader(num_batches: int = 50):
        for _ in range(num_batches):
            yield create_toy_data().to(device)

    # Estimate covariances
    print("Estimating K-FAC covariances...")
    estimator.estimate(toy_dataloader(), num_examples=20)

    # Save results
    estimator.save_covariances("./kfac_output")

    # Print summary
    if dist.get_rank() == 0 if is_distributed else True:
        print("\nSummary of collected covariances:")
        for name in estimator.activation_covs.keys():
            A_shape = estimator.activation_covs[name].shape
            G_shape = estimator.gradient_covs[name].shape
            print(f"  {name}: A{A_shape}, G{G_shape}")

    print("Done!")


if __name__ == "__main__":
    main()
