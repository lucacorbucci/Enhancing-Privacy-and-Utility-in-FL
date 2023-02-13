import copy
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import dill
import torch
from loguru import logger
from torch import nn

from pistacchio.Components.FederatedNode.federated_node import FederatedNode
from pistacchio.Models.federated_model import FederatedModel
from pistacchio.Utils.phases import Phase
from pistacchio.Utils.preferences import Preferences
from pistacchio.Utils.utils import Utils


logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)


def start_nodes(node, model):
    node.start_node(model)
    return node


def start_train(node):
    weights = node.train_local_model()
    return (node.node_id, weights)


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
        self.total_num_nodes = (
            preferences.data_split_config["num_nodes"]
            * preferences.data_split_config["num_clusters"]
        )
        self.pool_size = 4
        self.iterations = 3

    def launch_orchestrator(self) -> None:
        self.load_validation_data()
        self.federated_model = self.create_model()

        self.nodes = self.create_nodes()
        self.start_nodes()
        self.orchestrate_nodes()

    def create_nodes(
        self,
    ) -> list[FederatedNode]:

        nodes = []
        for cluster_id in range(self.preferences.data_split_config["num_clusters"]):
            for node_id in range(self.preferences.data_split_config["num_nodes"]):

                new_node = FederatedNode(
                    node_id=f"{node_id}_cluster_{cluster_id}",
                    preferences=self.preferences,
                )
                nodes.append(new_node)
        return nodes

    def start_nodes(self) -> None:

        logger.debug("Starting nodes...")
        model_list = [copy.deepcopy(self.model) for _ in range(len(self.nodes))]

        with ProcessPoolExecutor(self.pool_size) as executor:
            results = [
                executor.submit(start_nodes, node, model)
                for node, model in zip(self.nodes, model_list)
            ]
            self.nodes = []
            for result in results:
                self.nodes.append(result.result())

        logger.debug("Nodes started")

    def orchestrate_nodes(
        self,
    ) -> None:
        logger.debug("Orchestrating nodes...")
        model_list = [copy.deepcopy(self.model) for _ in range(len(self.nodes))]

        for iteration_number in range(self.iterations):
            weights = {}
            with ThreadPoolExecutor(self.pool_size) as executor:
                results = [executor.submit(start_train, node) for node in self.nodes]
                for result in results:
                    node_id, model_weights = result.result()
                    weights[node_id] = model_weights.weights

                Utils.compute_average(weights)
            self.compute_metrics()
        logger.debug("Training finished")

    def compute_metrics(self) -> None:
        logger.debug("Computing metrics...")
        (
            loss,
            accuracy,
            fscore,
            precision,
            recall,
            test_accuracy_per_class,
        ) = self.federated_model.evaluate_model()
        logger.debug("Metrics computed")

    def create_model(self) -> None:
        """This function creates and initialize the model that
        we'll use on the server for the validation.
        """
        logger.debug("Creating model")
        model = FederatedModel(
            dataset_name=self.preferences.dataset_name,
            node_name="server",
            preferences=self.preferences,
        )
        if model:
            model.init_model(net=self.model)
            model.trainloader = self.validation_set
            model.testloader = self.validation_set

            if self.preferences.server_config["differential_privacy_server"]:
                model.net, _, _ = model.init_differential_privacy(
                    phase=Phase.SERVER,
                )

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
            num_workers=8,
        )

        if self.preferences.debug:
            targets = []
            for _, data in enumerate(self.validation_set, 0):
                targets.append(data[-1])
            targets = [item.item() for sublist in targets for item in sublist]
            logger.info(f"Validation set: {Counter(targets)}")
