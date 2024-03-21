import argparse
import json
import os
import random

import numpy as np
import torch

from pistacchio_simulator.FederatedDataset.Utils.preferences import Preferences
from pistacchio_simulator.FederatedDataset.partition_dataset import FederatedDataset


def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True




def main() -> None:
    """Based on the preferences, this function generates the dataset

    Raises:
        InvalidSplitType: If the split type is not valid
    """
    parser = argparse.ArgumentParser(description="Experiments from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        metavar="N",
        help="Config file",
    )
    args = parser.parse_args()
    config = None
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    config = Preferences(**config)

    seed_everything(42)
    

    FederatedDataset.generate_partitioned_dataset(config=config)


if __name__ == "__main__":
    main()
