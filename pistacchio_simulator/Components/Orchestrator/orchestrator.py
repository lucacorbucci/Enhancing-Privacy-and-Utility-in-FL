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
from torch import nn

from pistacchio_simulator.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_simulator.Models.mnist import MnistNet
from pistacchio_simulator.Utils.learning import Learning
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.utils import Utils


logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)


def start_train(phase, preferences, weights, node_id, node_name, cluster_id):
    node = FederatedNode(
        node_id=node_id,
        node_name=node_name,
        cluster_id=cluster_id,
        preferences=preferences,
        phase=phase,
        weights=weights,
    )
    weights, metrics, num_examples = node.train_local_model(phase)
    return (node, node.node_id, node.cluster_id, metrics, weights, num_examples)


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
        self.pool_size = self.preferences.pool_size
        self.p2p_phase = bool(self.preferences.p2p_config)
        self.server_phase = bool(self.preferences.server_config)
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
        set_start_method("spawn")
        self.p2p_weights = {}
        self.store_path = f"{self.preferences.data_split_config.store_path}/nodes_data/"
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
        os.makedirs(self.store_path)
        self.cluster_nodes = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def launch_orchestrator(self) -> None:
        self.load_validation_data()

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
            logger.info("Orchestrating P2P phase")
            # In this case we want to perform a P2P training
            self.orchestrate_nodes(phase=Phase.P2P)

        if self.server_phase:
            logger.info("Orchestrating Server phase")
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
            if cluster_id not in self.cluster_nodes:
                self.cluster_nodes[cluster_id] = []
            for node_id in range(self.preferences.data_split_config.num_nodes):
                node_name = f"cluster_{cluster_id}_node_{node_id}"
                node = NodeInfo(node_name, node_id, cluster_id)
                nodes.append(node)
                self.cluster_nodes[cluster_id].append(node)
            if self.p2p_phase:
                self.p2p_weights[cluster_id] = Utils.get_parameters(self.model)

        return nodes

    def orchestrate_nodes(self, phase: Phase) -> None:
        logger.info(
            f"Orchestrating {phase} phase - {self.fl_rounds_p2p if phase == Phase.P2P else self.fl_rounds_server} Iterations"
        )
        iterations = self.fl_rounds_p2p if phase == Phase.P2P else self.fl_rounds_server

        self.model.to("cuda:0")

        with multiprocess.Pool(self.pool_size) as pool:
            for iteration in range(iterations):
                metrics_list = []
                all_weights = []
                all_weights_p2p = {}
                num_examples_list_p2p = {}
                num_examples_list = []
                nodes_to_select = copy.deepcopy(self.nodes)
                while len(nodes_to_select) > 0:
                    to_select = min(self.sampled_nodes, len(nodes_to_select))
                    sampled_nodes = random.sample(nodes_to_select, to_select)

                    nodes_to_select = list(set(nodes_to_select) - set(sampled_nodes))
                    if phase == Phase.P2P and self.p2p_weights or (phase == Phase.SERVER and iteration == 0 and self.p2p_phase):
                        weights = self.p2p_weights
                    else:
                        weights = Utils.get_parameters(self.model)

                    results = [
                        pool.apply_async(
                            start_train,
                            (
                                phase,
                                self.preferences,
                                weights[node.cluster_id]
                                if phase == Phase.P2P or (phase == Phase.SERVER and iteration == 0 and self.p2p_phase)
                                else weights,
                                node.node_id,
                                node.node_name,
                                node.cluster_id,
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
                            num_examples,
                        ) = result.get()

                        if phase == Phase.P2P:
                            if cluster_id not in all_weights_p2p:
                                all_weights_p2p[cluster_id] = []
                            all_weights_p2p[cluster_id].append(
                                copy.deepcopy(
                                    weights,
                                )
                            )
                            if cluster_id not in num_examples_list_p2p:
                                num_examples_list_p2p[cluster_id] = []
                            num_examples_list_p2p[cluster_id].append(num_examples)
                        else:
                            all_weights.append(weights)
                            num_examples_list.append(num_examples)
                            metrics_list.append(metrics)

                if phase == Phase.P2P:

                    for cluster_id in all_weights_p2p:
                        aggregated_weights = Utils.aggregate_weights(
                            all_weights_p2p[cluster_id],
                            num_examples_list_p2p[cluster_id],
                        )
                        self.p2p_weights[cluster_id] = aggregated_weights
                else:
                    aggregated_weights = Utils.aggregate_weights(
                        all_weights,
                        num_examples_list,
                    )

                    Utils.set_params(self.model, aggregated_weights)

                    aggregated_training_accuracy = np.mean(
                        [metric["accuracy"].item() for metric in metrics_list]
                    )
                    aggregated_training_loss = np.mean(
                        [metric["loss"].item() for metric in metrics_list]
                    )
                    aggregated_epsilon = max(
                        [metric["epsilon"] for metric in metrics_list]
                    )

                    if self.preferences.wandb:
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
                    self.log_metrics(
                        metrics,
                    )
                else:
                    pass
                    # self.evaluate_p2p(iteration=iteration)

    # def evaluate_p2p(self, iteration: int):
    #     for cluster_id, weights in self.p2p_weights.items():
    #         losses = []
    #         accuracies = []
    #         fscores = []
    #         precisions = []
    #         recalls = []
    #         cluster_model = copy.deepcopy(self.model)
    #         Utils.set_params(cluster_model, self.p2p_weights[cluster_id])
    #         for node in self.cluster_nodes[cluster_id]:
    #             data_test_node = torch.load(
    #                 torch.load(
    #                     f"../data/{self.preferences.dataset}/federated_data/{node.node_name}_test.pt"
    #                 )
    #             )
    #             test_node = torch.utils.data.DataLoader(
    #                 data_test_node,
    #                 batch_size=128,
    #                 shuffle=False,
    #                 num_workers=0,
    #             )
    #             (
    #                 loss,
    #                 accuracy,
    #                 fscore,
    #                 precision,
    #                 recall,
    #                 _,
    #             ) = Learning.evaluate_model(
    #                 model=cluster_model,
    #                 test_loader=test_node,
    #                 device=self.device,
    #             )
    #             losses.append(loss)
    #             accuracies.append(accuracy)
    #             fscores.append(fscore)
    #             precisions.append(precision)
    #             recalls.append(recall)

    #             metrics = {
    #                 f"test loss {node.node_name}": loss,
    #                 f"test accuracy {node.node_name}": accuracy,
    #                 f"test fscore {node.node_name}": fscore,
    #                 f"test precision {node.node_name}": precision,
    #                 f"test recall {node.node_name}": recall,
    #                 f"FL Round P2P {node.node_name}": iteration,
    #             }
    #             self.log_metrics(metrics)
    #         metrics = {
    #             f"test loss {cluster_id}": np.mean(losses),
    #             f"test accuracy {cluster_id}": np.mean(accuracies),
    #             f"test fscore {cluster_id}": np.mean(fscores),
    #             f"test precision {cluster_id}": np.mean(precisions),
    #             f"test recall {cluster_id}": np.mean(recalls),
    #             f"FL Round P2P {cluster_id}": iteration,
    #         }
    #         self.log_metrics(metrics)

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

    def log_metrics(self, metrics: dict):
        logger.debug(metrics)
        logger.debug("Metrics computed")
        logger.debug("Logging the metrics on wandb")
        if self.preferences.wandb:
            Utils.log_metrics_to_wandb(wandb_run=self.wandb, metrics=metrics)
        logger.debug("Metrics logged")
