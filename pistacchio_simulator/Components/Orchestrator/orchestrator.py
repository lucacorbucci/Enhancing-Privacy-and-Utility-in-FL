import copy
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import multiprocess
import numpy as np
import torch
from loguru import logger
from multiprocess import set_start_method
from pistacchio_simulator.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_simulator.Models.mnist import MnistNet
from pistacchio_simulator.Utils.learning import Learning
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.utils import Utils
from torch import nn

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)


def start_train(phase, preferences, weights):
    node = FederatedNode(
        node_id=0,
        node_name="cluster_0_node_0",
        cluster_id=0,
        preferences=preferences,
        phase=Phase.SERVER,
        weights=weights,
    )
    weights, metrics = node.train_local_model(phase)
    return (node, node.node_id, node.cluster_id, metrics, weights)


@dataclass(frozen=True, eq=True)
class NodeInfo:
    node_name: str
    node_id: int
    cluster_id: int


class Orchestrator:
    def __init__(
        self,
        preferences: Preferences,
        model: nn.Module,
    ) -> None:
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        self.preferences = preferences
        self.model = model
        self.federated_model = None
        self.validation_set = None
        self.pool_size = (
            preferences.data_split_config.num_nodes
            * preferences.data_split_config.num_clusters
        )
        self.p2p_phase = True if self.preferences.p2p_config else False
        self.server_phase = True if self.preferences.server_config else False
        self.sampled_nodes = self.preferences.pool_size
        self.local_epochs_server = (
            preferences.server_config.local_training_epochs if self.server_phase else 0
        )
        self.fl_rounds_server = (
            preferences.server_config.fl_rounds if self.server_phase else 0
        )
        self.local_epochs_p2p = (
            preferences.p2p_config.local_training_epochs if self.p2p_phase else 0
        )
        self.fl_rounds_p2p = preferences.p2p_config.fl_rounds if self.p2p_phase else 0
        self.p2p_training = True if preferences.p2p_config else None
        self.total_epsilon = 0
        self.model_weights = None
        set_start_method("spawn")
        self.p2p_weights = {}
        self.store_path = f"../data/{self.preferences.dataset}/nodes_data/"
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
        os.makedirs(self.store_path)

    def launch_orchestrator(self) -> None:
        self.load_validation_data()

        # self.federated_model = self.create_model()

        self.nodes = self.create_nodes()

        # We want to be sure that the number of nodes that we
        # sample at each iteration is always less or equal to the
        # total number of nodes.
        self.sampled_nodes = min(self.sampled_nodes, len(self.nodes))
        if self.preferences.wandb:
            self.wandb = Utils.configure_wandb(
                group="Orchestrator", preferences=self.preferences
            )

        print(f"P2P {self.p2p_phase} - Server {self.server_phase}")
        if self.p2p_phase:
            # In this case we want to perform a P2P training
            self.orchestrate_nodes(phase=Phase.P2P)

        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
        os.makedirs(self.store_path)

        if self.server_phase:
            # In this case we want to perform a server training
            self.orchestrate_nodes(phase=Phase.SERVER)
        logger.debug("Training finished")

        if self.preferences.wandb:
            Utils.finish_wandb(wandb_run=self.wandb)

    def create_nodes(
        self,
    ) -> list[FederatedNode]:
        """This function creates the nodes that will be used
        in the federated learning process.

        Returns
            list[FederatedNode]: the list of the nodes
        """
        nodes = []
        for cluster_id in range(self.preferences.data_split_config.num_clusters):
            for node_id in range(self.preferences.data_split_config.num_nodes):
                node_name = f"cluster_{cluster_id}_node_{node_id}"

                nodes.append(NodeInfo(node_name, node_id, cluster_id))

        return nodes

    def orchestrate_nodes(self, phase: Phase) -> None:
        logger.info(
            f"Orchestrating {phase} phase - {self.fl_rounds_p2p if phase == Phase.P2P else self.fl_rounds_server} Iterations"
        )
        iterations = self.fl_rounds_p2p if phase == Phase.P2P else self.fl_rounds_server

        data_test = torch.load("../data/mnist/federated_data/server_validation_set.pt")

        test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
        self.model.to("cuda:0")
        weights = {}

        with multiprocess.Pool(self.pool_size) as pool:
            for iteration in range(iterations):
                metrics_list = []
                nodes_to_select = copy.deepcopy(self.nodes)
                while len(nodes_to_select) > 0:
                    to_select = min(self.sampled_nodes, len(nodes_to_select))
                    sampled_nodes = random.sample(nodes_to_select, to_select)

                    nodes_to_select = list(set(nodes_to_select) - set(sampled_nodes))
                    if phase == Phase.P2P and self.p2p_weights:
                        weights = self.p2p_weights
                    else:
                        weights = Utils.get_parameters(self.model)

                    results = [
                        pool.apply_async(
                            start_train,
                            (
                                Phase.SERVER,
                                self.preferences,
                                weights[node.cluster_id]
                                if phase == Phase.P2P
                                else weights,
                            ),
                        )
                        for node in sampled_nodes
                    ]

                    for result in results:
                        (
                            node,
                            node_id,
                            cluster_id,
                            metrics,
                            weights,
                        ) = result.get()

                        if phase == Phase.P2P:
                            if cluster_id not in weights:
                                weights[cluster_id] = {}
                            weights[cluster_id][node_id] = copy.deepcopy(
                                weights,
                            )
                        else:
                            ttt = copy.deepcopy(weights)
                            metrics_list.append(metrics)

                if phase == Phase.P2P:
                    for cluster_id in weights:
                        avg = Utils.compute_average(weights[cluster_id])
                        self.p2p_weights["cluster_id"] = avg
                else:
                    print(type(weights))
                    # avg = Utils.compute_average(weights)
                    Utils.set_params(self.model, ttt)

                    aggregated_training_accuracy = np.mean(
                        [metric["accuracy"].item() for metric in metrics_list]
                    )
                    aggregated_training_loss = np.mean(
                        [metric["loss"].item() for metric in metrics_list]
                    )
                    aggregated_epsilon = max(
                        [metric["epsilon"] for metric in metrics_list]
                    )

                    Utils.log_metrics_to_wandb(
                        wandb_run=self.wandb,
                        metrics={
                            "Train loss": aggregated_training_loss,
                            "Train accuracy": aggregated_training_accuracy,
                            "Fl Round": iteration + self.fl_rounds_p2p,
                            "EPSILON": aggregated_epsilon,
                            "FL Round Server": iteration,
                        },
                    )

                logger.debug("Computed the average")
                if phase == Phase.SERVER:
                    self.evaluate(iteration=iteration)

    def evaluate(self, iteration: int) -> None:
        logger.debug("Computing metrics...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        (
            loss,
            accuracy,
            fscore,
            precision,
            recall,
            test_accuracy_per_class,
        ) = Learning.evaluate_model(
            model=self.model,
            test_loader=self.validation_set,
            device=self.device,
        )

        metrics = {
            "test loss": loss,
            "test accuracy": accuracy,
            "test fscore": fscore,
            "test precision": precision,
            "test recall": recall,
            "test_accuracy_per_class": test_accuracy_per_class,
            "FL_round": iteration + self.fl_rounds_p2p,
            "FL Round Server": iteration,
        }
        logger.debug(metrics)
        logger.debug("Metrics computed")
        logger.debug("Logging the metrics on wandb")
        if self.preferences.wandb:
            Utils.log_metrics_to_wandb(wandb_run=self.wandb, metrics=metrics)
        logger.debug("Metrics logged")

    def load_validation_data(self) -> None:
        """This function loads the validation data from disk."""
        data: torch.utils.data.DataLoader[Any] = None
        print(
            f"{self.preferences.data_split_config.store_path}/{self.preferences.data_split_config.server_validation_set}"
        )
        # Load the validation set dataset using the pytorch function
        data = torch.load(
            f"{self.preferences.data_split_config.store_path}/{self.preferences.data_split_config.server_validation_set}"
        )

        self.validation_set = torch.utils.data.DataLoader(
            data,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        if self.preferences.debug:
            targets = []
            for _, data in enumerate(self.validation_set, 0):
                targets.append(data[-1])
            targets = [item.item() for sublist in targets for item in sublist]
            logger.info(f"Validation set: {Counter(targets)}")
