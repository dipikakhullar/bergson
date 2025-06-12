import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from torch import autocast
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, default_data_collator

from bergson.approx_unrolling.language_task import LanguageModelingTask
from bergson.approx_unrolling.model_checkpoints import PythiaCheckpoints
from bergson.approx_unrolling.pile_data import get_pile_dataset

model_name = "EleutherAI/pythia-14m"

all_checkpoints = [[1000]]

pythia_checkpoints_manager = PythiaCheckpoints(all_checkpoints, model_name)
pythia_checkpoints_manager.save_models(overwrite=False)

assert pythia_checkpoints_manager.module_keys is not None

task = LanguageModelingTask(module_keys=pythia_checkpoints_manager.module_keys)

pythia_checkpoints_manager.module_keys = task.get_influence_tracked_modules()

model = pythia_checkpoints_manager.load_checkpoint(checkpoint=1000, device="cuda")
train_dataset = get_pile_dataset(model_str=model_name, step=0, max_samples=10000)


model = GPTNeoXForCausalLM.from_pretrained(
    model_name,
    revision=f"step{0}",
    device_map="auto",
    force_download=True,
)
rank = dist.get_rank() if dist.is_initialized() else 0


dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=SequentialSampler(data_source=train_dataset),
    collate_fn=default_data_collator,
)

for batch in tqdm(dataloader, position=rank):
    batch = send_to_device(batch, model.device)

    model.zero_grad()
    with autocast(
        device_type=str(model.device),
        enabled=True,
        dtype=torch.bfloat16,
    ):
        loss = task.compute_train_loss(batch=batch, model=model, sample=False)
    loss.backward()
