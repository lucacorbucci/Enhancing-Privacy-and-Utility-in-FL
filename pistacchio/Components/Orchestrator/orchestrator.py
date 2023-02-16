import copy
import sys
from collections import Counter
from typing import Any
import time
import dill
import torch
from loguru import logger
from torch import nn
from concurrent.futures import wait
from multiprocess.pool import ThreadPool
from pistacchio.Utils.phases import Phase
import random
import multiprocess
# import torch.multiprocessing 

from pistacchio.Components.FederatedNode.federated_node import FederatedNode
from pistacchio.Models.federated_model import FederatedModel
from pistacchio.Utils.phases import Phase
from pistacchio.Utils.preferences import Preferences
from pistacchio.Utils.utils import Utils
import gc

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
        self.pool_size = 25
        self.iterations = 2
        self.sampled_nodes = 100

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
        
        for node in self.nodes:
            node.federated_model.init_differential_privacy(phase=Phase.SERVER, node_id=node.node_id)
        logger.debug("Nodes started")

    
    def orchestrate_nodes(
        self,
    ) -> None:
        logger.debug("Orchestrating nodes...")


        with ThreadPool(self.pool_size) as pool:

            for iteration in range(self.iterations):
                logger.info(f"Iterazione {iteration}")
                weights = {}
                sampled_nodes = random.sample(self.nodes, self.sampled_nodes)
                results = [pool.apply_async(start_train, (node,)) for node in sampled_nodes]

                ready = []

                # while True:
                #     import time
                #     time.sleep(1)
                #     # catch exception if results are not ready yet
                #     try:
                #         ready = [result.ready() for result in results]
                #         successful = [result.successful() for result in results]
                #     except Exception:
                #         continue
                #     # exit loop if all tasks returned success
                #     if all(successful):
                #         break
                #     # raise exception reporting exceptions received from workers
                #     if all(ready) and not all(successful):
                #         raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')

                count = 0
                for result in results:
                    logger.debug(f"Popping {count}")
                    count += 1
                    node_id, model_weights = result.get()

                    weights[node_id] = copy.deepcopy(model_weights.weights)


                avg = Utils.compute_average(weights)
                for node in self.nodes:
                    node.federated_model.update_weights(avg)

                logger.debug("Computed the average")
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
                model.net, _, _ = model.init_differential_privacy(
                    phase=Phase.SERVER,
                    node_id="Orchestrator"
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
            num_workers=0,
        )

        if self.preferences.debug:
            targets = []
            for _, data in enumerate(self.validation_set, 0):
                targets.append(data[-1])
            targets = [item.item() for sublist in targets for item in sublist]
            logger.info(f"Validation set: {Counter(targets)}")
