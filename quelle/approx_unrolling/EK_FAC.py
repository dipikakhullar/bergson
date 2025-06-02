import argparse
import gc
import itertools
import logging
import os
import shutil
from typing import Dict

import torch
from hessians.analyzer import Analyzer, prepare_model
from hessians.arguments import FactorArguments
from hessians.utils.common.factor_arguments import all_low_precision_factor_arguments
from hessians.utils.dataset import DataLoaderKwargs
from language_task import LanguageModelingTask
from safetensors.torch import load_file, save_file
from torch.utils import data
from tqdm import tqdm
from transformers import default_data_collator

from quelle.approx_unrolling.logger_config import get_logger
from quelle.approx_unrolling.model_checkpoints import (
    ModelCheckpointManager,
)
from quelle.approx_unrolling.utils import TensorDict

BATCH_TYPE = Dict[str, torch.Tensor]
logger = get_logger(__name__)


def compute_EK_FAC(
    model: torch.nn.Module,
    task: LanguageModelingTask,
    train_dataset: data.Dataset,
    output_dir: str,
    EK_FAC_args: argparse.Namespace,
    factor_args: FactorArguments,
    factors_name: str,
):
    logging.basicConfig(level=logging.INFO)

    # Define task and prepare model.
    model = prepare_model(model, task)

    if EK_FAC_args.use_compile:
        model = torch.compile(model)  # type: ignore

    analyzer = Analyzer(
        analysis_name="",
        output_dir=output_dir,
        model=model,
        task=task,
        profile=EK_FAC_args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=EK_FAC_args.per_device_batch_size,
        factor_args=factor_args,
        initial_per_device_batch_size_attempt=64,
        dataloader_kwargs=dataloader_kwargs,
    )


def compute_EK_FAC_hessians(
    model: torch.nn.Module,
    task: LanguageModelingTask,
    train_dataset: data.Dataset,
    output_dir: str,
    EK_FAC_args: argparse.Namespace,
    factor_args: FactorArguments,
    factors_name: str,
):
    logging.basicConfig(level=logging.INFO)

    # Define task and prepare model.
    model = prepare_model(model, task)

    if EK_FAC_args.use_compile:
        model = torch.compile(model)  # type: ignore

    analyzer = Analyzer(
        analysis_name="",
        output_dir=output_dir,
        model=model,
        task=task,
        profile=EK_FAC_args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=EK_FAC_args.per_device_batch_size,
        factor_args=factor_args,
        initial_per_device_batch_size_attempt=32,
        dataloader_kwargs=dataloader_kwargs,
    )


def compute_EK_FAC_checkpoints(
    checkpoint_manager: ModelCheckpointManager,
    task: LanguageModelingTask,
    train_dataset: data.Dataset,
    EK_FAC_args: argparse.Namespace,
    device="cuda",
):
    """Loads the model from the checkpoint directory and computes the EK-FAC matrix."""

    # Compute influence factors.
    factors_name = EK_FAC_args.factor_strategy
    factor_args = FactorArguments(strategy=EK_FAC_args.factor_strategy)  # type:ignore
    if EK_FAC_args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(
            strategy=EK_FAC_args.factor_strategy, dtype=torch.bfloat16
        )
        factors_name += "_half"
    if EK_FAC_args.use_compile:
        factors_name += "_compile"

    # Check if the model directory exists
    if not os.path.exists(checkpoint_manager.model_dir):
        raise FileNotFoundError(
            f"Model directory {checkpoint_manager.model_dir} does not exist."
        )

    for checkpoint in itertools.chain(*checkpoint_manager.all_checkpoints):
        checkpoint_path = (
            checkpoint_manager.model_dir / f"checkpoint_{checkpoint}" / "model.pt"
        )
        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file {checkpoint_path} does not exist."
            )

        # Load the model from the checkpoint

        loaded_checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint, device=device
        )

        output_dir = (
            checkpoint_manager.model_dir
            / f"checkpoint_{checkpoint}"
            / "influence_results"
        )

        # compute_EK_FAC(
        #     model=loaded_checkpoint,
        #     output_dir=str(output_dir),
        #     task=task,
        #     train_dataset=train_dataset,
        #     EK_FAC_args=EK_FAC_args,
        #     factor_args=factor_args,
        #     factors_name=factors_name,
        # )

        compute_EK_FAC_hessians(
            model=loaded_checkpoint,
            output_dir=str(output_dir),
            task=task,
            train_dataset=train_dataset,
            EK_FAC_args=EK_FAC_args,
            factor_args=factor_args,
            factors_name=factors_name,
        )

        logger.info(
            f"Computed EK-FAC for checkpoint {checkpoint} and saved to {output_dir} \n"
            + "-" * 50
        )

    compute_average_covariance(
        checkpoint_manager=checkpoint_manager, EK_FAC_args=EK_FAC_args
    )

    compute_eigendecomposition(
        checkpoint_manager=checkpoint_manager, EK_FAC_args=EK_FAC_args
    )

    compute_lambda_matrices(
        checkpoint_manager=checkpoint_manager,
        EK_FAC_args=EK_FAC_args,
        task=task,
        factor_args=factor_args,
        factors_name=factors_name,
        train_dataset=train_dataset,
    )

    compute_lambda_matrices_average(
        checkpoint_manager=checkpoint_manager, EK_FAC_args=EK_FAC_args
    )


def compute_average_covariance(
    checkpoint_manager: ModelCheckpointManager, EK_FAC_args: argparse.Namespace
):
    """Computes the average covariance matrices bar{A} and bar{S} for each segment of checkpoints."""

    factor_half = "_half" if EK_FAC_args.use_half_precision else ""
    for i, segment in enumerate(checkpoint_manager.all_checkpoints):
        activations = []
        gradients = []

        for checkpoint in segment:
            checkpoint_gradient_path = (
                checkpoint_manager.model_dir
                / f"checkpoint_{checkpoint}"
                / "influence_results"
                / ("factors_ekfac" + factor_half)
                / "gradient_covariance.safetensors"
            )

            checkpoint_activation_path = (
                checkpoint_manager.model_dir
                / f"checkpoint_{checkpoint}"
                / "influence_results"
                / ("factors_ekfac" + factor_half)
                / "activation_covariance.safetensors"
            )

            gradients_covariance = TensorDict(load_file(checkpoint_gradient_path))
            gradients.append(gradients_covariance)

            activations_covariance = TensorDict(load_file(checkpoint_activation_path))

            activations.append(activations_covariance)

            # Should definitely change this, this is computing the number of tokens processed, i.e. batch_size*|data points|
            checkpoint_num_path = (
                checkpoint_manager.model_dir
                / f"checkpoint_{checkpoint}"
                / "influence_results"
                / ("factors_ekfac" + factor_half)
                / "num_activation_covariance_processed.safetensors"
            )

        segment_path = (
            checkpoint_manager.model_dir
            / f"segment_{i}"
            / "influence_results"
            / ("factors_ekfac" + factor_half)
        )
        segment_path.mkdir(parents=True, exist_ok=True)
        average_activation = sum(activations) / len(segment)
        average_gradient = sum(gradients) / len(segment)

        assert isinstance(average_activation, TensorDict)
        assert isinstance(average_gradient, TensorDict)

        save_file(
            average_activation.to_dict(),
            segment_path / "average_activation_covariance.safetensors",
        )
        save_file(
            average_gradient.to_dict(),
            segment_path / "average_gradient_covariance.safetensors",
        )

        # Should change this, see above
        assert checkpoint_num_path.exists(), (  # type:ignore
            f"{checkpoint_num_path} number file does not exist."  # type:ignore
        )

        shutil.copy2(
            checkpoint_num_path,  # type:ignore
            segment_path / "num_covariance_processed.safetensors",
        )


def compute_eigendecomposition(
    checkpoint_manager: ModelCheckpointManager,
    EK_FAC_args: argparse.Namespace,
    device="cuda",
):
    factor_half = "_half" if EK_FAC_args.use_half_precision else ""
    for i, segment in tqdm(enumerate(checkpoint_manager.all_checkpoints)):
        segment_path = (
            checkpoint_manager.model_dir
            / f"segment_{i}"
            / "influence_results"
            / ("factors_ekfac" + factor_half)
        )
        num_path = segment_path / "num_covariance_processed.safetensors"
        for type in ["activation", "gradient"]:
            # Load averages
            average_type_path = segment_path / f"average_{type}_covariance.safetensors"

            average_type = TensorDict(load_file(average_type_path, device=device))
            num_processed = TensorDict(load_file(num_path, device=device))

            # Init eigenfactors dictionary
            eigenvalue_factors = {}
            eigenvector_factors = {}

            # Perform eigen decomposition
            for key in average_type.keys():
                original_dtype = average_type[key].dtype
                average_type[key] = average_type[key].to(dtype=torch.float64)
                average_type[key].div_(num_processed[key])

                # In cases where covariance matrices are not exactly symmetric due to numerical issues.
                average_type[key] = average_type[key] + average_type[key].t()
                average_type[key].mul_(0.5)

                eigvals, eigvecs = torch.linalg.eigh(average_type[key])
                eigenvalue_factors[key] = eigvals.contiguous().to(dtype=original_dtype)
                eigenvector_factors[key] = eigvecs.contiguous().to(dtype=original_dtype)

            # Save eigenvalues and eigenvectors
            save_file(
                eigenvalue_factors,
                segment_path / f"{type}_eigenvalues.safetensors",
            )
            save_file(
                eigenvector_factors, segment_path / f"{type}_eigenvectors.safetensors"
            )
            del eigenvalue_factors, eigenvector_factors
            gc.collect()
            torch.compiler.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Computed eigendecomposition for segment_{i}.")


def compute_lambda_matrices(
    checkpoint_manager: ModelCheckpointManager,
    EK_FAC_args: argparse.Namespace,
    task: LanguageModelingTask,
    factor_args: FactorArguments,
    factors_name: str,
    train_dataset: data.Dataset,
    device: str = "cuda",
    output_dir: str = "influence_results",
):
    """Computes the lambda matrix for a specific checkpoint."""

    for i, segment in enumerate(checkpoint_manager.all_checkpoints):
        for checkpoint in segment:
            model = checkpoint_manager.load_checkpoint(
                checkpoint=checkpoint, device=device
            )
            model = prepare_model(model, task)

            if EK_FAC_args.use_compile:
                model = torch.compile(model)  # type: ignore

            result_output_dir = (
                checkpoint_manager.model_dir / f"checkpoint_{checkpoint}" / output_dir
            )

            analyzer = Analyzer(
                analysis_name="",
                output_dir=str(result_output_dir),
                model=model,
                task=task,
                profile=EK_FAC_args.profile,
            )
            # Configure parameters for DataLoader.
            dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
            analyzer.set_dataloader_kwargs(dataloader_kwargs)

            eigendecomposition_path = (
                checkpoint_manager.model_dir
                / f"segment_{i}"
                / "influence_results"
                / (
                    "factors_ekfac"
                    + ("_half" if EK_FAC_args.use_half_precision else "")
                )
            )

            analyzer.fit_lambda_matrices(
                factors_name=factors_name,
                dataset=train_dataset,
                per_device_batch_size=EK_FAC_args.per_device_batch_size,
                initial_per_device_batch_size_attempt=32,
                dataloader_kwargs=dataloader_kwargs,
                factor_args=factor_args,
                load_factors_output_dir=str(eigendecomposition_path),
            )


def compute_lambda_matrices_average(
    checkpoint_manager: ModelCheckpointManager,
    EK_FAC_args: argparse.Namespace,
    device: str = "cuda",
):
    """Computes the lambda matrices for each segment of checkpoints."""

    for i, segment in tqdm(enumerate(checkpoint_manager.all_checkpoints)):
        segment_path = (
            checkpoint_manager.model_dir
            / f"segment_{i}"
            / "influence_results"
            / ("factors_ekfac" + ("_half" if EK_FAC_args.use_half_precision else ""))
        )

        lambda_matrices = []
        for checkpoint in segment:
            lambda_matrix_path = (
                checkpoint_manager.model_dir
                / f"checkpoint_{checkpoint}"
                / "influence_results"
                / (
                    "factors_ekfac"
                    + ("_half" if EK_FAC_args.use_half_precision else "")
                )
                / "lambda_matrix.safetensors"
            )

            lambda_matrix = TensorDict(load_file(lambda_matrix_path, device=device))
            lambda_matrices.append(lambda_matrix)

        # Compute the average lambda matrix
        average_lambda_matrix = sum(lambda_matrices) / len(lambda_matrices)

        assert isinstance(average_lambda_matrix, TensorDict)
        save_file(
            average_lambda_matrix.to_dict(),
            segment_path / "lambda_matrix.safetensors",
        )


def compute_unrolled_lambdas(
    checkpoint_manager: ModelCheckpointManager,
    EK_FAC_args: argparse.Namespace,
    device: str = "cuda",
):
    for i, segment in tqdm(enumerate(checkpoint_manager.all_checkpoints)):
        segment_path = (
            checkpoint_manager.model_dir
            / f"segment_{i}"
            / "influence_results"
            / ("factors_ekfac" + ("_half" if EK_FAC_args.use_half_precision else ""))
        )

        lambda_matrix_path = segment_path / "lambda_matrix.safetensors"
        lambda_matrix = TensorDict(load_file(lambda_matrix_path, device=device))

        mean_lr = sum(checkpoint_manager.lr[i]) / len(checkpoint_manager.lr[i])  # type:ignore
        segment_len = segment[-1] - segment[0]
        unrolled_lambda = lambda_matrix._apply_unary(
            lambda t: torch.exp(t * mean_lr * segment_len)
        )

        save_file(
            unrolled_lambda.to_dict(),
            segment_path / "unrolled_lambda_matrix.safetensors",
        )


def prepare_hessians(
    checkpoint_manager: ModelCheckpointManager,
    EK_FAC_args: argparse.Namespace,
):
    # lambda_matrices = []
    # lambda_matrices_exp = []
    # # load lambda_matrices from the checkpoints

    # for checkpoint in checkpoint_manager.checkpoint_list:
    #     checkpoint_path = (
    #         checkpoint_manager.checkpoints_dir
    #         / checkpoint_manager.model_name
    #         / f"checkpoint_{checkpoint}"
    #         / "CHANGE_THIS"
    #         / "lambda_matrix.safetensors"
    #     )
    #     # Check if the checkpoint file exists
    #     if not os.path.exists(checkpoint_path):
    #         raise FileNotFoundError(
    #             f"Checkpoint file {checkpoint_path} does not exist."
    #         )
    #     lambda_matrices.append(load_file(checkpoint_path))

    # for lambda_matrix in lambda_matrices:
    #     lambda_matrix_exp = torch.exp(lambda_matrix)
    #     lambda_matrices_exp.append(lambda_matrix_exp)

    pass
