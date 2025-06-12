import json

import torch
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

path = (
    "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/rank32_correct/checkpoint-338"
)


def load():
    adapter_config_path = path + "/adapter_config.json"
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)

    print("Adapter config:", adapter_config)
    base_model_name = adapter_config.get("base_model_name_or_path", "Unknown")
    print(f"Base model: {base_model_name}")

    with torch.device("meta"):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")

    for module in base_model.modules():
        if isinstance(module, GPTNeoXLayer):
            fully_shard(module)
    base_model = fully_shard(base_model)

    base_model.to_empty(device="cuda")


if __name__ == "__main__":
    # adapter_config_path = path + "/adapter_config.json"
    # with open(adapter_config_path, "r") as f:
    #     adapter_config = json.load(f)

    # print("Adapter config:", adapter_config)
    # base_model_name = adapter_config.get("base_model_name_or_path", "Unknown")
    # print(f"Base model: {base_model_name}")

    # base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")

    # model = PeftModel.from_pretrained(base_model, path)
    load()
