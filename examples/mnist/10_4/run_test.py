import argparse
import json
import sys

import torch
from loguru import logger
from opacus import PrivacyEngine
from pydantic.tools import parse_obj_as

from pistacchio_simulator.Experiments.run_experiments import Experiment
from pistacchio_simulator.Utils.data_loader import DataLoader
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.utils import Utils


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

    if args.epsilon:
        # We need to understand the noise that we need to add based
        # on the epsilon that we want to guarantee
        max_noise = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for cluster_name in range(preferences.num_clusters):
            for node_name in range(preferences.num_nodes):
                model_noise = Utils.get_model(preferences=preferences).to(device)
                print(device)

                # get the training dataset of one of the clients
                train_set = DataLoader().load_splitted_dataset(
                    f"{preferences.data_split_config.store_path}/cluster_{cluster_name}_node{node_name}_private_train.pt",
                )
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

                privacy_engine = PrivacyEngine(accountant="rdp")
                optimizer_noise = Utils.get_optimizer(
                    preferences, model_noise, phase=Phase.SERVER
                )
                (
                    _,
                    private_optimizer,
                    _,
                ) = privacy_engine.make_private_with_epsilon(
                    module=model_noise,
                    optimizer=optimizer_noise,
                    data_loader=train_loader,
                    epochs=preferences.server_config.fl_rounds
                    * preferences.server_config.local_training_epochs,
                    target_epsilon=preferences.server_config.epsilon,
                    target_delta=preferences.hyperparameters_config.delta,
                    max_grad_norm=preferences.hyperparameters_config.max_grad_norm,
                )
            max_noise = max(max_noise, private_optimizer.noise_multiplier)
            print(
                f"Cluster {cluster_name} Node {node_name} - {(args.num_rounds // 10) * args.epochs} -- {private_optimizer.noise_multiplier}"
            )

        train_parameters.noise_multiplier = max_noise
        train_parameters.epsilon = None
        print(
            f">>>>> FINALE {(args.num_rounds // 10) * args.epochs} -- {train_parameters.noise_multiplier}"
        )

    Experiment.run(config, preferences)


if __name__ == "__main__":
    main()
