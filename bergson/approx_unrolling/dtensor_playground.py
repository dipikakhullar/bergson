import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import (  # type: ignore
    Shard,
    distribute_tensor,
    init_device_mesh,  # type: ignore
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)

if not dist.is_initialized():
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

rank = dist.get_rank() if dist.is_initialized() else 0


# print(f"Rank {rank} initialized on device {torch.cuda.current_device()}")
# print("world size", int(os.environ["WORLD_SIZE"]))
dist.barrier()

mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))
big_tensor = torch.rand(8001, 8000)

if rank == 0:
    print("-*" * 100)

# dist.barrier()
# Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.

my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
dist.barrier()

if rank == 0:
    print("-" * 50)

print(my_dtensor.to_local().shape, rank)
print(my_dtensor.to_local()[0, 0], rank)
# # mul_tensor = torch.tensor([2])
# # mul_dtensor = distribute_tensor(mul_tensor, mesh, [Shard(dim=0)])

# # another_test_tensor = torch.tensor(0, device=rank)

test_tensor = torch.zeros_like(my_dtensor)

new_dtensor = my_dtensor * test_tensor

# wait for all processes to finish
dist.barrier()

if rank == 0:
    print("Post edit", new_dtensor)

gathered_tensor = new_dtensor.full_tensor()  # type: ignore


dist.destroy_process_group()

print(gathered_tensor[0, 0])
