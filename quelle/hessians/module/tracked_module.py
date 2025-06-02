from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from accelerate.utils.dataclasses import BaseEnum
from torch import nn

from quelle.hessians.arguments import FactorArguments
from quelle.hessians.module.tracker.base import BaseTracker
from quelle.hessians.module.tracker.factor import CovarianceTracker, LambdaTracker
from quelle.hessians.utils.constants import (
    COVARIANCE_FACTOR_NAMES,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    LAMBDA_FACTOR_NAMES,
    PRECONDITIONED_GRADIENT_TYPE,
)


class ModuleMode(str, BaseEnum):
    """Enum representing a module's mode for factor computation.

    This enum indicates which factors need to be computed during
    forward and backward passes.
    """

    DEFAULT = "default"
    COVARIANCE = "covariance"
    LAMBDA = "lambda"
    PRECONDITION_GRADIENT = "precondition_gradient"
    GRADIENT_AGGREGATION = "gradient_aggregation"


class TrackedModule(nn.Module):
    """A wrapper class for PyTorch modules to compute influence factors.

    This class extends `nn.Module` to add functionality for tracking and computing
    various influence-related metrics.
    """

    SUPPORTED_MODULES: Dict[Type[nn.Module], Any] = {}

    def __init_subclass__(cls, module_type: Type[nn.Module] = None, **kwargs: Any) -> None:
        """Automatically registers subclasses as supported modules.

        Args:
            module_type (Type[nn.Module], optional):
                The type of module this subclass supports.
            **kwargs:
                Additional keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        if module_type is not None:
            cls.SUPPORTED_MODULES[module_type] = cls

    def __init__(
        self,
        name: str,
        original_module: nn.Module,
        factor_args: Optional[FactorArguments] = None,
        per_sample_gradient_process_fnc: Optional[Callable] = None,
    ) -> None:
        """Initializes an instance of the `TrackedModule` class.

        Args:
            name (str):
                The original name of the module.
            original_module (nn.Module):
                The original module to be wrapped.
            factor_args (FactorArguments, optional):
                Arguments for computing factors.
            per_sample_gradient_process_fnc (Callable, optional):
                Optional function to post-process per-sample gradients.
        """
        super().__init__()

        self.name = name
        self.original_module = original_module

        assert isinstance(self.original_module.weight.dtype, torch.dtype)
        self._constant: torch.Tensor = nn.Parameter(
            torch.zeros(
                1,
                dtype=self.original_module.weight.dtype,
                requires_grad=True,
            )
        )
        self.current_mode = ModuleMode.DEFAULT
        self.factor_args = FactorArguments() if factor_args is None else factor_args
        self.per_sample_gradient_process_fnc = per_sample_gradient_process_fnc

        self._trackers = {
            ModuleMode.DEFAULT: BaseTracker(self),
            ModuleMode.COVARIANCE: CovarianceTracker(self),
            ModuleMode.LAMBDA: LambdaTracker(self),
        }

        self.attention_mask: Optional[torch.Tensor] = None
        self.gradient_scale: float = 1.0
        self.storage: Dict[str, Optional[Union[torch.Tensor, PRECONDITIONED_GRADIENT_TYPE]]] = {}
        self.einsum_path: Optional[List[int]] = None
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initializes storage for various factors."""

        # Storage for activation and pseudo-gradient covariance matrices #
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            self.storage[covariance_factor_name]: Optional[torch.Tensor] = None

        # Storage for eigenvectors and eigenvalues #
        for eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
            self.storage[eigen_factor_name]: Optional[torch.Tensor] = None

        # Storage for lambda matrices #
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            self.storage[lambda_factor_name]: Optional[torch.Tensor] = None

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Performs a forward pass of the tracked module.

        This method should have identical behavior to that of the original module.

        Args:
            inputs (torch.Tensor):
                Input tensor to the module.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.

        Returns:
            torch.Tensor:
                The output of the forward pass.
        """
        outputs = self.original_module(inputs, *args, **kwargs)
        if outputs.requires_grad:
            return outputs
        return outputs + self._constant

    def update_factor_args(self, factor_args: FactorArguments) -> None:
        """Updates the factor arguments.

        Args:
            factor_args (FactorArguments):
                New factor arguments to set.
        """
        self.factor_args = factor_args

    def get_factor(self, factor_name: str) -> Optional[torch.Tensor]:
        """Retrieves a factor by name from storage.

        Args:
            factor_name (str):
                The name of the factor to retrieve.

        Returns:
            Optional[torch.Tensor]:
                The requested factor, or `None` if not found.
        """
        if factor_name not in self.storage or self.storage[factor_name] is None:
            return None
        return self.storage[factor_name]

    def release_factor(self, factor_name: str) -> None:
        """Releases a factor from memory.

        Args:
            factor_name (str):
                The name of the factor to release.
        """
        if factor_name not in self.storage or self.storage[factor_name] is None:
            return None
        del self.storage[factor_name]
        self.storage[factor_name] = None

    def set_factor(self, factor_name: str, factor: Any) -> None:
        """Sets a factor in storage.

        Args:
            factor_name (str):
                The name of the factor to set.
            factor (Any):
                The factor value to store.
        """
        if factor_name in self.storage:
            self.storage[factor_name] = factor

    def set_mode(self, mode: ModuleMode, release_memory: bool = False) -> None:
        """Sets the operating mode of the `TrackedModule`.

        This method changes the current mode and manages associated trackers and memory.

        Args:
            mode (ModuleMode):
                The new mode to set.
            release_memory (bool):
                Whether to release memory for all trackers.
        """
        self._trackers[self.current_mode].release_hooks()
        self.einsum_path = None
        self.current_mode = mode

        if release_memory:
            for _, tracker in self._trackers.items():
                tracker.release_memory()

        self._trackers[self.current_mode].register_hooks()

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor] = None) -> None:
        """Sets the attention mask for activation covariance computations.

        Args:
            attention_mask (torch.Tensor, optional):
                The attention mask to set.
        """
        self.attention_mask = attention_mask

    def set_gradient_scale(self, scale: float = 1.0) -> None:
        """Sets the scale of the gradient obtained from `GradScaler`.

        Args:
            scale (float):
                The scale factor to set.
        """
        self.gradient_scale = scale

    def finalize_iteration(self) -> None:
        """Finalizes statistics for the current iteration."""
        self._trackers[self.current_mode].finalize_iteration()

    def exist(self) -> bool:
        """Checks if the desired statistics are available.

        Returns:
            bool:
                `True` if statistics exist, `False` otherwise.
        """
        return self._trackers[self.current_mode].exist()

    def synchronize(self, num_processes: int) -> None:
        """Synchronizes statistics across multiple processes.

        Args:
            num_processes (int):
                The number of processes to synchronize across.
        """
        self._trackers[self.current_mode].synchronize(num_processes=num_processes)

    def truncate(self, keep_size: int) -> None:
        """Truncates stored statistics to a specified size.

        Args:
            keep_size (int):
                The number of dimension to keep.
        """
        self._trackers[self.current_mode].truncate(keep_size=keep_size)

    def accumulate_iterations(self) -> None:
        """Accumulates (or prepares to accumulate) statistics across multiple iterations."""
        self._trackers[self.current_mode].accumulate_iterations()

    def finalize_all_iterations(self) -> None:
        """Finalizes statistics after all iterations."""
        self._trackers[self.current_mode].finalize_all_iterations()

    @abstractmethod
    def get_flattened_activation(self, input_activation: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """Returns the flattened activation tensor and the number of stacked activations.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, int]]:
                The flattened activation tensor and the number of stacked activations. The flattened
                activation is a 2-dimensional matrix with dimension `activation_num x activation_dim`.
        """
        raise NotImplementedError("Subclasses must implement the `get_flattened_activation` method.")

    @abstractmethod
    def get_flattened_gradient(self, output_gradient: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """Returns the flattened output gradient tensor.

        Args:
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the
                PyTorch's backward hook.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, int]]:
                The flattened output gradient tensor and the number of stacked gradients. The flattened
                gradient is a 2-dimensional matrix  with dimension `gradient_num x gradient_dim`.
        """
        raise NotImplementedError("Subclasses must implement the `get_flattened_gradient` method.")

    @abstractmethod
    def compute_summed_gradient(self, input_activation: torch.Tensor, output_gradient: torch.Tensor) -> torch.Tensor:
        """Returns the summed gradient tensor.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the PyTorch's backward hook.

        Returns:
            torch.Tensor:
                The aggregated gradient tensor.
        """
        raise NotImplementedError("Subclasses must implement the `compute_summed_gradient` method.")

    @abstractmethod
    def compute_per_sample_gradient(
        self, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        """Returns the flattened per-sample gradient tensor. For a brief introduction to
        per-sample gradient, see https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the PyTorch's backward hook.

        Returns:
            torch.Tensor:
                The per-sample gradient tensor.
        """
        raise NotImplementedError("Subclasses must implement the `compute_per_sample_gradient` method.")
