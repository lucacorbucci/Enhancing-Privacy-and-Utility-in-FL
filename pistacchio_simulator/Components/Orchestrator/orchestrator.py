import copy
import gc
import random
import sys
import time
from collections import Counter
from concurrent.futures import wait
from typing import Any

import dill
import multiprocess
import torch
from loguru import logger
from multiprocess.pool import ThreadPool
from torch import nn

from pistacchio_simulator.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_simulator.Models.federated_model import FederatedModel
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.utils import Utils


logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)


def start_nodes(node, model, communication_queue):
    new_node = copy.deepcopy(node)
    new_node.federated_model = new_node.init_federated_model(model)

    communication_queue.put(new_node)
    return "OK"


def start_train(node):
    weights = node.train_local_model()
    return (node.node_id, node.cluster_id, weights)


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

        self.sampled_nodes_server = (
            self.preferences.pool_size
        )
        self.local_epochs_server = preferences.server_config.local_training_epochs
        self.fl_rounds_server = preferences.server_config.fl_rounds
        self.local_epochs_p2p = preferences.p2p_config.local_training_epochs
        self.fl_rounds_p2p = preferences.p2p_config.fl_rounds
        self.p2p_training = True if preferences.p2p_config else None

    def launch_orchestrator(self) -> None:
        self.load_validation_data()
        self.federated_model = self.create_model()
        self.nodes = self.create_nodes()
        self.start_nodes()

        if self.p2p_training:
            grouped_nodes = self.group_nodes()
            if self.preferences.p2p_config.differential_privacy:
                for node in self.nodes:
                    node.federated_model.init_differential_privacy(phase=Phase.P2P)

        if self.preferences.server_config.differential_privacy:
            for node in self.nodes:
                node.federated_model.init_differential_privacy(phase=Phase.SERVER)

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
        manager = multiprocess.Manager()
        communication_queue = manager.Queue()

        with multiprocess.Pool(self.pool_size) as pool:
            results = [
                pool.apply_async(start_nodes, (node, model, communication_queue))
                for node, model in zip(self.nodes, model_list)
            ]
            self.nodes = []
            for result in results:
                _ = result.get()
                self.nodes.append(communication_queue.get())

        logger.debug("Nodes started")

    def orchestrate_p2p_phase(self, grouped_nodes: list[FederatedNode]):
         with ThreadPool(self.pool_size) as pool:
            for iteration in range(self.fl_rounds_p2p):
                logger.info(f"Iterazione {iteration}")
                weights = {}
                sampled_nodes = random.sample(self.nodes, self.sampled_nodes)
                results = [
                    pool.apply_async(start_train, (node,)) for node in sampled_nodes
                ]

                ready = []

                count = 0
                for result in results:
                    logger.debug(f"Popping {count}")
                    count += 1
                    node_id, cluster_id, model_weights = result.get()

                    if cluster_id not in weights:
                        weights[cluster_id] = {}
                    weights[cluster_id][node_id] = copy.deepcopy(model_weights.weights)

                for cluster_id in weights: 
                    avg = Utils.compute_average(weights[cluster_id])
                    for node in self.nodes:
                        if node.cluster_id == cluster_id:
                            node.federated_model.update_weights(avg)
                logger.debug("Computed the average")
                #self.log_metrics(iteration=iteration)

    def orchestrate_server_phase(self):
        with ThreadPool(self.pool_size) as pool:
            for iteration in range(self.fl_rounds_server):
                logger.info(f"Iterazione {iteration}")
                weights = {}
                sampled_nodes = random.sample(self.nodes, self.sampled_nodes)
                results = [
                    pool.apply_async(start_train, (node,)) for node in sampled_nodes
                ]

                ready = []

                count = 0
                for result in results:
                    logger.debug(f"Popping {count}")
                    count += 1
                    node_id, cluster_id, model_weights = result.get()

                    weights[node_id] = copy.deepcopy(model_weights.weights)

                avg = Utils.compute_average(weights)
                for node in self.nodes:
                    node.federated_model.update_weights(avg)
                self.federated_model.update_weights(avg)
                logger.debug("Computed the average")
                self.log_metrics(iteration=iteration)

    def orchestrate_nodes(
        self,
        grouped_nodes: Optional[list[FederatedNode]] = None,
    ) -> None:
        logger.debug("Orchestrating nodes...")

        if self.p2p_phase and grouped_nodes:
            # In this case we want to perform a P2P training
            orchestrate_p2p_phase(grouped_nodes)
        
        if self.server_phase:
            # In this case we want to perform a server training
            orchestrate_server_phase()
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
            true_positive_rate,
            false_positive_rate,
        ) = self.federated_model.evaluate_model()
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "fscore": fscore,
            "precision": precision,
            "recall": recall,
            "test_accuracy_per_class": test_accuracy_per_class,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "epoch": iteration,
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
            dataset_name=self.preferences.dataset_name,
            node_name="server",
            preferences=self.preferences,
        )
        if model:
            model.init_model(net=orchestrator_model)
            model.trainloader = self.validation_set
            model.testloader = self.validation_set

            if self.preferences.server_config["differential_privacy_server"]:
                model.net, _, _ = model.init_differential_privacy(phase=Phase.SERVER)

        logger.debug("Model created")
        return model

    def load_validation_data(self) -> None:
        """This function loads the validation data from disk."""
        data: torch.utils.data.DataLoader[Any] = None
        with open(
            (
                f"../data/{self.preferences.dataset_name}/federated_split"
                f"/server_validation/{self.preferences.data_split_config['server_validation_set']}"
            ),
            "rb",
        ) as file:
            data = dill.load(file)
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
