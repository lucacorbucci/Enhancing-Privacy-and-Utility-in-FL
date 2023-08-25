import argparse
import json

from FederatedDataset.partition_dataset import FederatedDataset
from FederatedDataset.Utils.preferences import Preferences


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

    FederatedDataset.generate_partitioned_dataset(config=config)


if __name__ == "__main__":
    main()
