import copy
import random
import sys
from collections import Counter
from typing import Any
import numpy as np 
import multiprocess
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


def start_nodes(node, model):
    new_node = copy.deepcopy(node)
    new_node.federated_model = new_node.init_federated_model(model)

    return new_node


def start_train(node, phase):

    if not node.federated_model.diff_privacy_initialized:
        print(f"INIZIALIZZO DI NUOVO {node.federated_model.diff_privacy_initialized}")
        node.federated_model.init_differential_privacy(phase=phase)
    weights, metrics = node.train_local_model(phase)
    return (node, node.node_id, node.cluster_id, weights, metrics)


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
        set_start_method("spawn")

    def launch_orchestrator(self) -> None:
        self.load_validation_data()
        self.federated_model = self.create_model()
        self.nodes = self.create_nodes()
        self.start_nodes()

        if self.p2p_training:
            grouped_nodes = self.group_nodes()
        
        # We want to be sure that the number of nodes that we
        # sample at each iteration is always less or equal to the
        # total number of nodes.
        self.sampled_nodes = min(self.sampled_nodes, len(self.nodes))
        if self.preferences.wandb:
            self.wandb = Utils.configure_wandb(
                group="Orchestrator", preferences=self.preferences
            )
        self.orchestrate_nodes(grouped_nodes=grouped_nodes)
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
                new_node = FederatedNode(
                    node_id=f"{node_id}_cluster_{cluster_id}",
                    preferences=self.preferences,
                    cluster_id=cluster_id,
                )
                nodes.append(new_node)

        return nodes

    def group_nodes(self) -> dict[str, list[FederatedNode]]:
        """This function groups the nodes by cluster_id.

        Returns
            dict[str, list[FederatedNode]]: a dictionary where the
                key is the cluster_id and the value is a list of nodes
                that belong to that cluster.
        """
        grouped_nodes = {}
        for node in self.nodes:
            if node.cluster_id not in grouped_nodes:
                grouped_nodes[node.cluster_id] = []
            grouped_nodes[node.cluster_id].append(node)

        return grouped_nodes

    def start_nodes(self) -> None:
        logger.debug("Starting nodes...")
        model_list = [copy.deepcopy(self.model) for _ in range(len(self.nodes))]

        with multiprocess.Pool(self.pool_size) as pool:
            results = [
                pool.apply_async(start_nodes, (node, model))
                for node, model in zip(self.nodes, model_list)
            ]
            self.nodes = []
            for result in results:
                node = result.get()
                self.nodes.append(node)

        logger.debug("Nodes started")

    def orchestrate_p2p_phase(self, grouped_nodes: dict[str, list[FederatedNode]]):
        logger.info(f"Orchestrating P2P phase - {self.fl_rounds_p2p} Iterations")
        all_nodes = self.nodes

        for iteration in range(self.fl_rounds_p2p):
            nodes_to_select = copy.copy(all_nodes)
            all_nodes = []
            logger.info(f"Iteration {iteration}")
            weights = {}
            with multiprocess.Pool(self.pool_size) as pool:
                while len(nodes_to_select) > 0:
                    to_select = min(self.sampled_nodes, len(nodes_to_select))
                    sampled_nodes = random.sample(nodes_to_select, to_select)

                    nodes_to_select = [node for node in nodes_to_select if node not in sampled_nodes]
                
                    results = [
                        pool.apply_async(start_train, (node, Phase.P2P))
                        for node in sampled_nodes
                    ]

                    count = 0
                    for result in results:
                        count += 1
                        node, node_id, cluster_id, model_weights, metric = result.get()
                        all_nodes.append(node)
                        if cluster_id not in weights:
                            weights[cluster_id] = {}
                        weights[cluster_id][node_id] = copy.deepcopy(model_weights.weights)

            for cluster_id in weights:
                avg = Utils.compute_average(weights[cluster_id])
                for node in self.nodes:
                    if node.cluster_id == cluster_id:
                        node.federated_model.update_weights(avg)
            logger.debug("Computed the average")

    def orchestrate_server_phase(self):
        logger.info("Orchestrating Server phase...")

        metrics = []
        all_nodes = self.nodes

        for iteration in range(self.fl_rounds_server):
            nodes_to_select = copy.copy(all_nodes)
            logger.info(f"Iterazione {iteration}")
            all_nodes = []
            weights = {}
            with multiprocess.Pool(self.pool_size) as pool:
                while len(nodes_to_select) > 0:
                    to_select = min(self.sampled_nodes, len(nodes_to_select))
                    sampled_nodes = random.sample(nodes_to_select, to_select)
                    # remove the sampled nodes from the list of nodes to be selected
                    nodes_to_select = [node for node in nodes_to_select if node not in sampled_nodes]
                    results = [
                        pool.apply_async(start_train, (node, Phase.SERVER))
                        for node in sampled_nodes
                    ]

                    count = 0
                    for result in results:
                        count += 1
                        node, node_id, cluster_id, model_weights, metric = result.get()
                        all_nodes.append(node)

                        weights[node_id] = copy.deepcopy(model_weights.weights)
                        metrics.append(metric)


            aggregated_training_accuracy = np.mean([metric["accuracy"].item() for metric in metrics])
            aggregated_training_loss = np.mean([metric["loss"].item() for metric in metrics])
            aggregated_epsilon = max([metric["epsilon"] for metric in metrics])

            Utils.log_metrics_to_wandb(wandb_run=self.wandb, metrics={
                "train loss": aggregated_training_loss,
                "train accuracy": aggregated_training_accuracy,
                "FL ROUND": iteration,
                "EPSILON": aggregated_epsilon,
            })
            avg = Utils.compute_average(weights)
            for node in self.nodes:
                node.federated_model.update_weights(avg)
            self.federated_model.update_weights(avg)
            logger.debug("Computed the average")

            self.log_metrics(iteration=iteration)
            logger.debug("Computed the average")


    def orchestrate_nodes(
        self,
        grouped_nodes: dict[str, list[FederatedNode]] = None,
    ) -> None:
        logger.debug("Orchestrating nodes...")

        if self.p2p_phase and grouped_nodes:
            # In this case we want to perform a P2P training
            self.orchestrate_p2p_phase(grouped_nodes)

        if self.server_phase:
            # In this case we want to perform a server training
            self.orchestrate_server_phase()
        logger.debug("Training finished")

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

    def create_model(self) -> None:
        """This function creates and initialize the model that
        we'll use on the server for the validation.
        """
        logger.debug("Creating model")
        orchestrator_model = copy.deepcopy(self.model)
        model = FederatedModel(
            dataset_name=self.preferences.dataset,
            node_name="server",
            preferences=self.preferences,
        )
        if model:
            model.init_model(net=orchestrator_model)
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
