import argparse
import json

from pistacchio_simulator.DataSplit.generate_dataset import generate_splitted_dataset
from pistacchio_simulator.Utils.preferences import Preferences


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
    config = Preferences.generate_from_json(config)

    generate_splitted_dataset(config=config)


if __name__ == "__main__":
    main()
