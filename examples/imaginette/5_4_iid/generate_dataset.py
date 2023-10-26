import argparse
import json

from pistacchio_simulator.FederatedDataset.partition_dataset import FederatedDataset
from pistacchio_simulator.FederatedDataset.Utils.preferences import Preferences


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

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ],
    )
    train_dataset = CelebaDataset(
        csv_path="../data/celeba/train_smiling.csv",
        image_path="../data/celeba/img_align_celeba",
        transform=transform,
        debug=True,
    )
    test_dataset = CelebaDataset(
        csv_path="../data/celeba/test_smiling.csv",
        image_path="../data/celeba/img_align_celeba",
        transform=transform,
        debug=True,
    )

    FederatedDataset.generate_partitioned_dataset(
        config=config, train_ds=train_dataset, test_ds=test_dataset
    )


if __name__ == "__main__":
    main()
