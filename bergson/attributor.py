import torch

from .data import load_gradients
from .gradients import GradientProcessor


class Attributor:
    def __init__(
        self,
        index_path: str,
        device: torch.device | str = "cpu",
    ):
        # Map the gradients into memory (very fast)
        mmap = load_gradients(index_path)

        # Load them onto the desired device (slow)
        self.grads = torch.from_numpy(mmap).to(device, copy=True)

        # In-place normalize for numerical stability
        self.grads /= self.grads.norm(dim=1, keepdim=True)

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

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
