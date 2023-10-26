import argparse
import json
import sys

from loguru import logger
from pistacchio_simulator.Experiments.run_experiments import Experiment
from pistacchio_simulator.Utils.preferences import Preferences
from pydantic.tools import parse_obj_as


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
    parser.add_argument(
        "--lr_p2p",
        type=float,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--lr_server",
        type=float,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--batch_size_p2p",
        type=int,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--local_training_epochs_p2p",
        type=int,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        metavar="N",
        help="Config file",
    )
    parser.add_argument(
        "--fl_rounds_P2P",
        type=int,
        default=None,
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

    preferences = parse_obj_as(Preferences, config)
    if preferences.wandb_config.sweep:
        if preferences.p2p_config:
            if args.lr_p2p:
                print("SETTING THE LEARNING RATE P2P")
                preferences.p2p_config.lr = args.lr_p2p
            if args.local_training_epochs_p2p:
                print("SETTING THE NUMBER OF LOCAL STEPS P2P")
                preferences.p2p_config.local_training_epochs = (
                    args.local_training_epochs_p2p
                )
            if args.fl_rounds_P2P:
                print("SETTING THE NUMBER OF FL ROUNDS P2P")
                preferences.p2p_config.fl_rounds = args.fl_rounds_P2P
            if args.batch_size_p2p:
                print("SETTING THE BATCH SIZE P2P")
                preferences.p2p_config.batch_size = args.batch_size_p2p

        if preferences.server_config:
            if args.lr_server:
                print("SETTING THE LEARNING RATE SERVER")
                preferences.server_config.lr = args.lr_server
            if preferences.server_config.differential_privacy:
                if args.noise_multiplier:
                    print("SETTING THE NOISE MULTIPLIER")
                    preferences.server_config.noise_multiplier = args.noise_multiplier
                if args.max_grad_norm:
                    print("SETTING THE MAX GRAD NORM")
                    preferences.hyperparameters_config.max_grad_norm = (
                        args.max_grad_norm
                    )
            if args.batch_size:
                print("SETTING THE BATCH SIZE")
                preferences.server_config.batch_size = args.batch_size

        if args.optimizer:
            print("SETTING THE OPTIMIZER")
            preferences.hyperparameters_config.optimizer = args.optimizer

    Experiment.run(config, preferences)


if __name__ == "__main__":
    main()
