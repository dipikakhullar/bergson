from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from quelle.hessians.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]


class LanguageModelingTask(Task):
    def __init__(
        self,
        module_keys: List[str] = [],
        track_attention: bool = True,
        track_mlp: bool = True,
        track_custom: List[str] = [],
    ):
        """
        Initialize the LanguageModelingTask with customized influence tracking.

        Args:
            num_layers: Number of transformer layers to track
            track_attention: Whether to track attention modules
            track_mlp: Whether to track MLP modules
            custom_modules: Additional custom modules to track
            **kwargs: Other parameters for the parent class
        """
        super().__init__()
        self.track_attention = track_attention
        self.track_mlp = track_mlp
        self.track_custom = track_custom
        self.module_keys = module_keys

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # We could also compute the log-likelihood or averaged margin.
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Will track everything if no modules are specified."""
        total_modules = []

        for m in self.module_keys:
            if any(x in m.lower() for x in ["dropout", "layernorm", "act"]):
                continue
            if (
                "attention." in m.lower() or "attn." in m.lower()
            ) and self.track_attention:
                total_modules.append(m)
            if "mlp." in m.lower() and self.track_mlp:
                total_modules.append(m)

        if self.track_custom:
            total_modules += self.track_custom

        return total_modules if len(total_modules) > 0 else None

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]
