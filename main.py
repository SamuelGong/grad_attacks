# Reference:
# https://github.com/JonasGeiping/breaching/blob/main/examples/TAG%20-%20Optimization-based%20Attack%20-%20FL-Transformer%20for%20Causal%20LM.ipynb
# https://github.com/JonasGeiping/breaching/blob/main/examples/APRIL%20%20-%20Analytic%20Attack%20-%20Vision%20Transformer%20on%20ImageNet.ipynb

import os
import sys
import time
import json
import logging
from train import get_gradient
from attack import gradient_attack
from config import load_configurations
from utils import (
    set_log, fix_randomness, prepare_data, git_status
)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


def logging_meta_info(config):
    set_log(config.log_file)
    pid = os.getpid()
    print(f"PID: {pid}")
    git_branch, git_commit = git_status()
    logging.info(f"Git branch: {git_branch}, Git commit: {git_commit}")
    logging.info(f"Configuration:\n{json.dumps(config, indent=4)}")
    print(f"Log file: {config.log_file}")


def main(args):
    begin_time = time.perf_counter()
    config = load_configurations(args[0])
    logging_meta_info(config)

    # Step 1: obtain the ground truth data to be recovered, e.g., a sentence
    fix_randomness(config.global_seed)
    dataset, ground_truth_data, auxiliary = prepare_data(config=config)

    # Step 2: simulating the training and obtain the gradient
    gradient, model = get_gradient(
        config=config,
        dataset=dataset,
        auxiliary=auxiliary
    )

    # Step 3: launching the attack
    reconstructed_data = gradient_attack(
        config=config,
        model=model,
        auxiliary=auxiliary,
        target_gradient=gradient,
        ground_truth_data=ground_truth_data  # only for evaluation use
    )

    end_time = time.perf_counter()
    duration = round(end_time - begin_time, 2)
    logging.info(f"Done in {duration}s.")


if __name__ == "__main__":
    main(sys.argv[1:])
