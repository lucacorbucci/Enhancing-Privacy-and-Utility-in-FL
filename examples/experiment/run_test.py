import argparse
import json
import sys

from loguru import logger

from pistacchio.Experiments.run_experiments import Experiment


def main() -> None:
    """The main function of the program.
    It parses the preferences and launch the experiment
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
    try:
        with open(args.config, "r", encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError:
        logger.error("Config file not found")
        sys.exit()

    Experiment.run(config)


if __name__ == "__main__":
    main()
    main()
