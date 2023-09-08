import copy
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import multiprocess
import numpy as np
import torch
from loguru import logger
from multiprocess import set_start_method
from multiprocess.pool import ThreadPool
from pistacchio_simulator.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_simulator.Models.federated_model import FederatedModel
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


# def start_nodes(node, model):
#     new_node = copy.deepcopy(node)
#     new_node.federated_model = new_node.init_federated_model(model)

#     return new_node


def start_train(node, phase, preferences, model):
    node = FederatedNode(
        node_id=node.node_id,
        node_name=node.node_name,
        cluster_id=node.cluster_id,
        preferences=preferences,
        phase=phase,
        model=model,
    )
    weights, metrics = node.train_local_model(phase)
    return (node, node.node_id, node.cluster_id, weights, metrics)


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
        self.local_epochs_server = preferences.server_config.local_training_epochs
        self.fl_rounds_server = preferences.server_config.fl_rounds
        self.local_epochs_p2p = preferences.p2p_config.local_training_epochs
        self.fl_rounds_p2p = preferences.p2p_config.fl_rounds
        self.p2p_training = True if preferences.p2p_config else None
        self.total_epsilon = 0
        self.model_weights = None
        set_start_method("spawn")
        self.p2p_weights = {}

    def launch_orchestrator(self) -> None:
        self.load_validation_data()
        self.federated_model = self.create_model()

        self.nodes = self.create_nodes()

        # We want to be sure that the number of nodes that we
        # sample at each iteration is always less or equal to the
        # total number of nodes.
        self.sampled_nodes = min(self.sampled_nodes, len(self.nodes))
        if self.preferences.wandb:
            self.wandb = Utils.configure_wandb(
                group="Orchestrator", preferences=self.preferences
            )
        if self.p2p_phase:
            # In this case we want to perform a P2P training
            self.orchestrate_nodes(phase=Phase.P2P)

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
        all_nodes = self.nodes
        metrics = []

        for iteration in range(iterations):
            nodes_to_select = copy.copy(all_nodes)
            all_nodes = []
            logger.info(f"Iteration {iteration}")
            weights = {}
            with multiprocess.Pool(self.pool_size) as pool:
                while len(nodes_to_select) > 0:
                    to_select = min(self.sampled_nodes, len(nodes_to_select))
                    sampled_nodes = random.sample(nodes_to_select, to_select)
                    node_models = []
                    for node in sampled_nodes:
                        node_model = copy.deepcopy(self.model)
                        if phase == Phase.P2P and self.p2p_weights:
                            weights_model = self.p2p_weights
                        else:
                            weights_model = self.federated_model.get_weights()

                        node_federated_model = FederatedModel(
                            dataset_name=self.preferences.dataset,
                            node_name=node.node_name,
                            model=node_model,
                            preferences=self.preferences,
                        )
                        node_federated_model.update_weights(weights_model)
                        node_models.append(node_federated_model)
                    nodes_to_select = list(set(nodes_to_select) - set(sampled_nodes))

                    results = [
                        pool.apply_async(
                            start_train,
                            (node, Phase.P2P, self.preferences, model),
                        )
                        for node, model in zip(sampled_nodes, node_models)
                    ]

                    count = 0
                    for result in results:
                        count += 1
                        node, node_id, cluster_id, model_weights, metric = result.get()
                        all_nodes.append(node)

                        if phase == Phase.P2P:
                            if cluster_id not in weights:
                                weights[cluster_id] = {}
                            weights[cluster_id][node_id] = copy.deepcopy(
                                model_weights.weights
                            )
                        else:
                            weights[node_id] = copy.deepcopy(model_weights.weights)
                            metrics.append(metric)

            if phase == Phase.P2P:
                for cluster_id in weights:
                    avg = Utils.compute_average(weights[cluster_id])
                    self.p2p_weights["cluster_id"] = avg
            else:
                self.federated_model.update_weights(avg)

            logger.debug("Computed the average")

    # def orchestrate_server_phase(self):
    #     logger.info("Orchestrating Server phase...")

    #     metrics = []
    #     all_nodes = self.nodes

    #     for iteration in range(self.fl_rounds_server):
    #         nodes_to_select = copy.copy(all_nodes)
    #         logger.info(f"Iterazione {iteration}")
    #         all_nodes = []
    #         weights = {}
    #         with multiprocess.Pool(self.pool_size) as pool:
    #             while len(nodes_to_select) > 0:
    #                 to_select = min(self.sampled_nodes, len(nodes_to_select))
    #                 sampled_nodes = random.sample(nodes_to_select, to_select)
    #                 for node in sampled_nodes:
    #                     node_model = copy.deepcopy(self.model)
    #                     node.load_data()
    #                     node_federated_model = FederatedModel(
    #                         dataset_name=self.preferences.dataset,
    #                         node_name=node.node_id,
    #                         model=node_model,
    #                         preferences=self.preferences,
    #                     )
    #                     node_federated_model.update_weights = (
    #                         self.federated_model.get_weights()
    #                     )

    #                     node.federated_model = node_federated_model

    #                 # remove the sampled nodes from the list of nodes to be selected
    #                 nodes_to_select = [
    #                     node for node in nodes_to_select if node not in sampled_nodes
    #                 ]
    #                 results = [
    #                     pool.apply_async(start_train, (node, Phase.SERVER))
    #                     for node in sampled_nodes
    #                 ]

    #                 count = 0
    #                 for result in results:
    #                     count += 1
    #                     node, node_id, cluster_id, model_weights, metric = result.get()
    #                     all_nodes.append(node)

    #                     weights[node_id] = copy.deepcopy(model_weights.weights)
    #                     metrics.append(metric)

    #         aggregated_training_accuracy = np.mean(
    #             [metric["accuracy"].item() for metric in metrics]
    #         )
    #         aggregated_training_loss = np.mean(
    #             [metric["loss"].item() for metric in metrics]
    #         )
    #         aggregated_epsilon = max([metric["epsilon"] for metric in metrics])

    #         Utils.log_metrics_to_wandb(
    #             wandb_run=self.wandb,
    #             metrics={
    #                 "train loss": aggregated_training_loss,
    #                 "train accuracy": aggregated_training_accuracy,
    #                 "FL ROUND": iteration,
    #                 "EPSILON": aggregated_epsilon,
    #             },
    #         )
    #         avg = Utils.compute_average(weights)
    #         for node in self.nodes:
    #             node.federated_model.update_weights(avg)
    #         self.federated_model.update_weights(avg)
    #         logger.debug("Computed the average")

    #         self.log_metrics(iteration=iteration)
    #         logger.debug("Computed the average")

    def log_metrics(self, iteration: int) -> None:
        logger.debug("Computing metrics...")
        (
            loss,
            accuracy,
            fscore,
            precision,
            recall,
            test_accuracy_per_class,
        ) = self.federated_model.evaluate_model()
        metrics = {
            "test loss": loss,
            "test accuracy": accuracy,
            "test fscore": fscore,
            "test precision": precision,
            "test recall": recall,
            "test_accuracy_per_class": test_accuracy_per_class,
            "FL ROUND": iteration,
        }
        logger.debug(metrics)
        logger.debug("Metrics computed")
        logger.debug("Logging the metrics on wandb")
        if self.preferences.wandb:
            Utils.log_metrics_to_wandb(wandb_run=self.wandb, metrics=metrics)
        logger.debug("Metrics logged")

    def create_model(self) -> FederatedModel:
        """This function creates and initialize the model that
        we'll use on the server for the validation.
        """
        logger.debug("Creating model")
        orchestrator_model = copy.deepcopy(self.model)
        model = FederatedModel(
            dataset_name=self.preferences.dataset,
            node_name="server",
            model=orchestrator_model,
            preferences=self.preferences,
        )
        if model:
            model.trainloader = self.validation_set
            model.testloader = self.validation_set

            if self.preferences.server_config.differential_privacy:
                model.net, _, _ = model.init_differential_privacy(phase=Phase.SERVER)

        logger.debug("Model created")
        return model

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
            batch_size=16,
            shuffle=False,
            num_workers=0,
        )

        if self.preferences.debug:
            targets = []
            for _, data in enumerate(self.validation_set, 0):
                targets.append(data[-1])
            targets = [item.item() for sublist in targets for item in sublist]
            logger.info(f"Validation set: {Counter(targets)}")
