import os

import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, default_data_collator
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from bergson.approx_unrolling.language_task import LanguageModelingTask
from bergson.approx_unrolling.model_checkpoints import PythiaCheckpoints
from bergson.approx_unrolling.pile_data import get_pile_dataset


def setup_distributed():
    """Properly initialize distributed training"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Set CUDA device for current rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size()


# Main execution
if __name__ == "__main__":
    # 1. Initialize distributed FIRST
    rank, world_size = setup_distributed()

    model_name = "EleutherAI/pythia-14m"

    # 2. Initialize your checkpoint manager and task
    all_checkpoints = [[1000]]
    pythia_checkpoints_manager = PythiaCheckpoints(all_checkpoints, model_name)
    pythia_checkpoints_manager.save_models(overwrite=False)

    assert pythia_checkpoints_manager.module_keys is not None

    task = LanguageModelingTask(module_keys=pythia_checkpoints_manager.module_keys)
    pythia_checkpoints_manager.module_keys = task.get_influence_tracked_modules()

    # 3. Load dataset
    train_dataset = get_pile_dataset(model_str=model_name, step=0, max_samples=4000)

    if rank == 0:
        print(f"Loaded {len(train_dataset)} samples from the Pile dataset.")

    # 4. Load model with proper device handling
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Option A: Load directly to GPU (for smaller models)
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=f"step{0}",
        force_download=True,
        device_map=None,  # Don't use device_map with FSDP
    )

    # Option B: Meta device initialization (for larger models - commented out)
    # config = GPTNeoXConfig.from_pretrained(model_name)
    # with torch.device("meta"):
    #     model = GPTNeoXForCausalLM(config)

    if rank == 0:
        print("-*" * 20)
        print(f"Loaded model {model_name} for training.")

    # 5. Apply FSDP2 wrapping BEFORE moving to device
    if dist.is_initialized() and world_size > 1:
        # Apply FSDP2 to transformer layers first
        for module in model.modules():
            if isinstance(module, GPTNeoXLayer):
                fully_shard(module)

        # Then wrap the entire model
        model = fully_shard(model)

        if rank == 0:
            print("-*" * 20)
            print("Applied FSDP2 sharding to model layers.")
    else:
        # Single GPU case - just move to device
        model = model.to(device)
        if rank == 0:
            print("-*" * 20)
            print("Moved model to single GPU.")

    # 6. Ensure model is on correct device
    if hasattr(model, "device"):
        model_device = model.device
    else:
        # For FSDP models, get device from parameters
        model_device = next(model.parameters()).device

    if rank == 0:
        print(f"Model device: {model_device}")

    # 7. Create dataloader with proper distributed sampler
    if dist.is_initialized() and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        batch_size_per_gpu = 32 // world_size  # Adjust batch size for distributed
    else:
        sampler = SequentialSampler(train_dataset)
        batch_size_per_gpu = 32

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        collate_fn=default_data_collator,
        pin_memory=True,  # Better performance
        num_workers=2,  # Parallel data loading
    )

    # 8. Training loop with proper error handling
    model.train()

    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, position=rank, disable=rank != 0)):
            # Send batch to model's device
            batch = send_to_device(batch, model_device)

            # Zero gradients
            model.zero_grad()

            # Forward pass
            try:
                loss = task.compute_train_loss(batch=batch, model=model, sample=False)

                # Check for NaN loss
                if torch.isnan(loss):
                    if rank == 0:
                        print(f"Warning: NaN loss detected at batch {batch_idx}")
                    continue

                # Backward pass
                loss.backward()

                # Optional: Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if rank == 0 and batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"OOM error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    except Exception as e:
        if rank == 0:
            print(f"Training error: {e}")
        raise

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


# Alternative approach with meta device initialization for large models
def create_model_with_meta_device(model_name, revision="step0"):
    """
    Alternative approach using meta device initialization
    Useful for very large models that don't fit in GPU memory initially
    """
    from transformers import GPTNeoXConfig

    # Load config
    config = GPTNeoXConfig.from_pretrained(model_name)

    # Create model on meta device
    with torch.device("meta"):
        model = GPTNeoXForCausalLM(config)

    # Apply FSDP2
    for module in model.modules():
        if isinstance(module, GPTNeoXLayer):
            fully_shard(module)
    model = fully_shard(model)

    # Move to actual device
    model.to_empty(device="cuda")

    # Load pretrained weights
    # Note: This requires careful handling of the state dict
    # You may need to use model.load_state_dict() or DCP APIs

    return model


# Mixed precision version
def create_model_with_mixed_precision(model_name, revision="step0"):
    """
    Create model with FSDP2 and mixed precision for better performance
    """
    # Mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,  # Forward/backward in bfloat16
        reduce_dtype=torch.float32,  # Gradient reduction in float32
    )

    # Load model
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,  # Start with float32, FSDP will cast
        device_map=None,
    )

    # Apply FSDP2 with mixed precision
    for module in model.modules():
        if isinstance(module, GPTNeoXLayer):
            fully_shard(module, mp_policy=mp_policy)

    model = fully_shard(model, mp_policy=mp_policy)

    return model
