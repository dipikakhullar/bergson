import torch

from .data import load_index
from .gradients import GradientProcessor
from .utils import assert_type


class Attributor:
    def __init__(
        self,
        index_path: str,
        device: int | None = None,
    ):
        # Contains the actual inputs fed into the model to produce the gradients
        self.index = load_index(index_path)

        # Load the gradient processor
        map_loc = f"cuda:{device}" if device is not None else "cpu"
        self.processor = GradientProcessor.load(index_path, map_location=map_loc)

        # Load the whole column of gradients into memory
        idx_th = self.index.with_format("torch", device=map_loc, dtype=torch.float16)
        self.grads = assert_type(torch.Tensor, idx_th["gradient"])

    @torch.compile
    def search(self, queries: torch.Tensor, k: int) -> torch.return_types.topk:
        """
        Search for the `k` nearest examples in the index based on the query or queries.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        return torch.topk(queries @ self.grads.mT, k)
