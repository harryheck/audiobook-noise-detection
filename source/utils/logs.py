# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.


"""
This module handles the logging and summary writing for the project.
"""

import datetime
import os
import shutil
from pathlib import Path, PosixPath
from typing import Any, Dict, Optional, Union
import tensorflow as tf
import tensorflow.summary as summary
from tensorboard.plugins.hparams import api as hp



if __name__ == "__main__":
    import config
else:
    from utils import config

class CustomSummaryWriter:
    """
    A custom TensorBoard writer for logging hyperparameters and metrics.
    """

    def __init__(self, log_dir, params=None, metrics=None):
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.params = params
        self.metrics = metrics if metrics else {}

        if params:
            self._log_hyperparameters(params, self.metrics)

    def _log_hyperparameters(self, params, metrics):
        """
        Logs hyperparameters and initial metrics to TensorBoard.
        """
        params = params.flattened_copy()  # Flatten nested parameters
        with self.file_writer.as_default():
            hp.hparams(params)  # Log hyperparameters
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(metric_name, metric_value, step=1)

    def log_scalar(self, name, value, step):
        """
        Logs scalar values to TensorBoard.
        """
        with self.file_writer.as_default():
            tf.summary.scalar(name, value, step=step)
        self.file_writer.flush()

    def step(self):
        """
        Flushes the file writer.
        """
        self.file_writer.flush()


def return_tensorboard_path() -> PosixPath:
    """
    Returns the path to the TensorBoard logs directory for the current experiment.
    The path is constructed using the default directory, current datetime, and DVC experiment name.

    Returns:
        PosixPath: The path to the TensorBoard logs directory.
    """
    default_dir = config.get_env_variable("DEFAULT_DIR")
    dvc_exp_name = config.get_env_variable("DVC_EXP_NAME")
    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    tensorboard_path = Path(
        f"{default_dir}/logs/tensorboard/{current_datetime}_{dvc_exp_name}"
    )
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    return tensorboard_path


def copy_tensorboard_logs() -> str:
    """
    Copies the TensorBoard logs specific to the current experiment from the host directory
    to the temporary experiment directory.

    Returns:
        str: The name of the copied directory.
    """
    default_dir = config.get_env_variable("DEFAULT_DIR")
    dvc_exp_name = config.get_env_variable("DVC_EXP_NAME")

    tensorboard_logs_source = Path(f"{default_dir}/logs/tensorboard")
    tensorboard_logs_destination = Path(f"exp_logs/tensorboard")
    tensorboard_logs_destination.mkdir(parents=True, exist_ok=True)
    for f in tensorboard_logs_source.iterdir():
        if f.is_dir() and f.name.endswith(dvc_exp_name):
            dir_name = f.name
            shutil.copytree(
                f, tensorboard_logs_destination / f.name, dirs_exist_ok=True
            )
            print(
                f"TensorBoard log '{f.name}' copied to '{tensorboard_logs_destination / f.name}'"
            )
            return f.name
    print("No TensorBoard logs found. Skipping copying.")
    return f"no_tensorboard_logs_{dvc_exp_name}"


def copy_slurm_logs(dir_name) -> None:
    """
    Copies the SLURM logs specific to the current experiment from the host directory
    to the temporary experiment directory.

    If the SLURM_JOB_ID is not found, the copying process is skipped.

    Args:
        dir_name (str): The name of the directory to copy the SLURM logs to.

    Raises:
        ValueError: If the directory name does not end with the DVC experiment name.
    """
    default_dir = config.get_env_variable("DEFAULT_DIR")
    current_slurm_job_id = config.get_env_variable("SLURM_JOB_ID")
    dvc_exp_name = config.get_env_variable("DVC_EXP_NAME")

    if dir_name is None:
        raise ValueError("Directory name is None.")
    elif not dir_name.endswith(dvc_exp_name):
        raise ValueError(f"Directory '{dir_name}' does not end with '{dvc_exp_name}'")

    if current_slurm_job_id:
        slurm_logs_source = Path(f"{default_dir}/logs/slurm")
        slurm_logs_destination = Path(f"exp_logs/slurm/{dir_name}")
        slurm_logs_destination.mkdir(parents=True, exist_ok=True)
        if current_slurm_job_id is not None:
            for f in slurm_logs_source.iterdir():
                if f.is_file() and f.name.endswith(current_slurm_job_id + ".out"):
                    shutil.copy(f, slurm_logs_destination)
        print(f"SLURM log 'slurm-{current_slurm_job_id}.out' copied to {slurm_logs_destination / f.name}.")
    else:
        print("No SLURM_JOB_ID found. Skipping SLURM logs copying.")


def main():
    """Main function to copy SLURM and TensorBoard logs."""
    dir_name = copy_tensorboard_logs()
    copy_slurm_logs(dir_name=dir_name)


if __name__ == "__main__":
    main()
