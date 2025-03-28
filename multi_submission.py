#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, learning_rate, weight_scaling, dilation):
    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S train.learning_rate={learning_rate} -S train.weight_scaling={weight_scaling} -S model.conv2d_dilation={dilation}",
    }
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    learning_rate_list = [0.00005, 0.00001, 0.0005, 0.0001]
    weight_scaling_list = [0.5, 0.7, 0.9, 1.0]
    dilation_list = [1, 2, 4]

    # Iterate over a cartesian product parameter grid of the test_split and batch_size lists
    for learning_rate, batch_size, dilation in itertools.product(learning_rate_list, weight_scaling_list, dilation_list):
        submit_batch_job(arguments, learning_rate, batch_size, dilation)