from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

import torch
from accelerate.utils.dataclasses import BaseEnum

from bergson.hessians.utils.constants import (
    ACTIVATION_EIGENVECTORS_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_MATRIX_NAME,
)

STORAGE_TYPE = Dict[str, Any]


class FactorStrategy(str, BaseEnum):
    """Strategies for computing preconditioning factors."""

    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    KFAC = "kfac"
    EKFAC = "ekfac"


class FactorConfig(metaclass=ABCMeta):
    """Configurations for each supported factor strategy."""

    CONFIGS: Dict[FactorStrategy, Any] = {}

    def __init_subclass__(cls, factor_strategy: Optional[FactorStrategy] = None, **kwargs) -> None:
        """Registers all subclasses of `FactorConfig`."""
        super().__init_subclass__(**kwargs)
        if factor_strategy is not None:
            assert factor_strategy in [strategy.value for strategy in FactorStrategy]
            cls.CONFIGS[factor_strategy] = cls()

    @property
    @abstractmethod
    def requires_covariance_matrices(self) -> bool:
        """Returns `True` if the strategy requires computing covariance matrices."""
        raise NotImplementedError("Subclasses must implement the `requires_covariance_matrices` property.")

    @property
    @abstractmethod
    def requires_eigendecomposition(self) -> bool:
        """Returns `True` if the strategy requires performing Eigendecomposition."""
        raise NotImplementedError("Subclasses must implement the `requires_eigendecomposition` property.")

    @property
    @abstractmethod
    def requires_lambda_matrices(self) -> bool:
        """Returns `True` if the strategy requires computing Lambda matrices."""
        raise NotImplementedError("Subclasses must implement the `requires_lambda_matrices` property.")

    @property
    @abstractmethod
    def requires_eigendecomposition_for_lambda(self) -> bool:
        """Returns `True` if the strategy requires loading Eigendecomposition results, before computing
        Lambda matrices."""
        raise NotImplementedError("Subclasses must implement the `requires_eigendecomposition_for_lambda` property.")

    @property
    @abstractmethod
    def requires_covariance_matrices_for_precondition(self) -> bool:
        """Returns `True` if the strategy requires loading covariance matrices, before computing
        preconditioned gradient."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_covariance_matrices_for_precondition` property."
        )

    @property
    @abstractmethod
    def requires_eigendecomposition_for_precondition(self) -> bool:
        """Returns `True` if the strategy requires loading Eigendecomposition results, before computing
        preconditioned gradient."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_eigendecomposition_for_precondition` property."
        )

    @property
    @abstractmethod
    def requires_lambda_matrices_for_precondition(self) -> bool:
        """Returns `True` if the strategy requires loading Lambda matrices, before computing
        the preconditioned gradient."""
        raise NotImplementedError("Subclasses must implement the `requires_lambda_matrices_for_precondition` property.")


class Identity(FactorConfig, factor_strategy=FactorStrategy.IDENTITY):
    """Applies no preconditioning to the gradient."""

    @property
    def requires_covariance_matrices(self) -> bool:
        return False

    @property
    def requires_eigendecomposition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return False

    @property
    def requires_lambda_matrices(self) -> bool:
        return False

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return False

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return False

    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: STORAGE_TYPE,
    ) -> torch.Tensor:
        del storage
        return gradient


class Diagonal(FactorConfig, factor_strategy=FactorStrategy.DIAGONAL):
    """Applies diagonal preconditioning to the gradient."""

    @property
    def requires_covariance_matrices(self) -> bool:
        return False

    @property
    def requires_eigendecomposition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return False

    @property
    def requires_lambda_matrices(self) -> bool:
        return True

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return False

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return True


class Kfac(FactorConfig, factor_strategy=FactorStrategy.KFAC):
    """Applies KFAC preconditioning to the gradient.

    See https://arxiv.org/pdf/1503.05671.pdf for details.
    """

    @property
    def requires_covariance_matrices(self) -> bool:
        return True

    @property
    def requires_eigendecomposition(self) -> bool:
        return True

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return False

    @property
    def requires_lambda_matrices(self) -> bool:
        return False

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return True

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return False

    @torch.no_grad()
    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: STORAGE_TYPE,
    ) -> torch.Tensor:
        activation_eigenvectors = storage[ACTIVATION_EIGENVECTORS_NAME].to(device=gradient.device)
        gradient_eigenvectors = storage[GRADIENT_EIGENVECTORS_NAME].to(device=gradient.device)
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(device=gradient.device)
        gradient = torch.matmul(gradient_eigenvectors.t(), torch.matmul(gradient, activation_eigenvectors))
        gradient.mul_(lambda_matrix)
        gradient = torch.matmul(gradient_eigenvectors, torch.matmul(gradient, activation_eigenvectors.t()))
        return gradient


class Ekfac(FactorConfig, factor_strategy=FactorStrategy.EKFAC):
    """Applies EKFAC preconditioning to the gradient.

    See https://arxiv.org/pdf/1806.03884.pdf for details.
    """

    @property
    def requires_covariance_matrices(self) -> bool:
        return True

    @property
    def requires_eigendecomposition(self) -> bool:
        return True

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return True

    @property
    def requires_lambda_matrices(self) -> bool:
        return True

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return True

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return True
