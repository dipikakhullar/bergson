import itertools
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from safetensors.torch import load_file
from transformers import GPTNeoXForCausalLM

from quelle.approx_unrolling.logger_config import get_logger

logger = get_logger(__name__)


class ModelCheckpointManager(ABC):
    """
    Abstract base class for managing model checkpoints.
    Handles saving/loading models with organized directory structure.
    """

    def __init__(
        self,
        all_checkpoints: List[List[int]],
        model_name: str,
        lr: Union[List[List[float]], float],
        dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the checkpoint manager.

        Args:
        """
        self.model_name = model_name
        self.model_dir = ".models" / Path(model_name)
        if dir is not None:
            dir = Path(dir)
            self.model_dir = dir / self.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.all_checkpoints = all_checkpoints
        self.module_keys = None
        self.lr = lr
        if isinstance(lr, int):
            self.lr = [[lr] * len(segment) for segment in all_checkpoints]

    @abstractmethod
    def load_models(self, checkpoint: int, *args, **kwargs) -> torch.nn.Module:
        """
        Abstract method to load a model
        """
        pass

    def save_models(
        self,
        overwrite: bool = False,
    ):
        """
        Save a model to the checkpoints directory.

        Args:
            model: PyTorch model to save
            model_name: Name for the saved model
            metadata: Additional metadata to save with the model
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            overwrite: Whether to overwrite existing checkpoint

        Returns:
            Path to the saved checkpoint
        """
        # Create model-specific directory

        for checkpoint in itertools.chain(*self.all_checkpoints):
            checkpoint_path = self.model_dir / f"checkpoint_{checkpoint}"
            # Check if file exists and handle overwrite

            if checkpoint_path.exists():
                if overwrite:
                    logger.warning(
                        f"Checkpoint {checkpoint_path} already exists. Overwriting..."
                    )
                    shutil.rmtree(checkpoint_path)
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                else:
                    logger.info(
                        f"Checkpoint {checkpoint_path} already exists, will continue with cache.Set overwrite=True to replace."
                    )
                    # return self.model_dir need to save module_keys to avoid reloading everything always
            else:
                checkpoint_path.mkdir(parents=True, exist_ok=True)

            try:
                loaded_model = self.load_models(checkpoint)
                torch.save(loaded_model, checkpoint_path / "model.pt")
                logger.info(f"Saved checkpoint to {str(checkpoint_path / 'model.pt')}")
            except Exception as e:
                logger.error(f"Failed to save model at checkpoint {checkpoint}: {e}")
                raise

        self.module_keys = [m[0] for m in loaded_model.named_modules()]  # type: ignore

        logger.info(f"Saved models to {self.model_dir}")
        return self.model_dir

    def load_checkpoint(self, checkpoint: int, device: str = "cpu") -> torch.nn.Module:
        """
        Load a saved checkpoint.

        Args:
            model_name: Name of the model
            checkpoint_name: Specific checkpoint file name (if None, loads latest)

        Returns:
            Model loaded from the checkpoint
        """
        model_checkpoint_dir = self.model_dir / f"checkpoint_{checkpoint}" / "model.pt"

        try:
            model = torch.load(
                model_checkpoint_dir, map_location=device, weights_only=False
            )  # Alternative way to load model?
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_checkpoint_dir}: {e}")
            raise

    def load_cache(
        self, segment: int, device: str = "cpu"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load a model from a specific checkpoint.

        Args:
            checkpoint: Checkpoint number to load
            device: Device to load the model on (default: 'cpu')

        Returns:
            Loaded PyTorch model
        """

        MATRIX_NAMES = [
            "activation_eigenvectors",
            "gradient_eigenvectors",
            "lambda_matrix",
        ]

        segment_path = (
            self.model_dir
            / f"segment_{segment}"
            / "influence_results"
            / "factors_ekfac_half"
        )

        cache_dict = {}

        for name in MATRIX_NAMES:
            matrix_path = segment_path / f"{name}.safetensors"
            matrix = load_file(matrix_path, device=device)
            cache_dict[name] = matrix

        return cache_dict


class PythiaCheckpoints(ModelCheckpointManager):
    """
    Checkpoint manager for Pythia models.
    """

    def __init__(
        self,
        all_checkpoints: List[List[int]],
        model_name: str,
        lr: Union[List[List[float]], float] = 1e-3,
    ):
        self.model_name = model_name
        super().__init__(all_checkpoints, model_name, lr)

    def load_models(self, checkpoint: int) -> torch.nn.Module:
        """
        Load a Pythia model from a specific checkpoint.
        """

        model = GPTNeoXForCausalLM.from_pretrained(
            self.model_name,
            revision=f"step{checkpoint}",
            device_map="cpu",
            force_download=True,
        )

        return model


if __name__ == "__main__":
    all_checkpoints = [[2000, 3000, 4000], [6000]]
    model_name = "EleutherAI/pythia-14m"

    pythia_manager = PythiaCheckpoints(all_checkpoints, model_name)
    pythia_manager.save_models(overwrite=True)
