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
from opacus import PrivacyEngine
from pistacchio_simulator.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_simulator.Utils.data_loader import DataLoader
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
        self.preferences = preferences
        self.model = model
        self.federated_model = None
        self.test_set = None
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
        if self.p2p_phase:
            noise = self.get_current_noise(self.preferences.p2p_config, phase=Phase.P2P)
            self.preferences.p2p_config.noise_multiplier = noise
            self.preferences.p2p_config.epsilon = None

        if self.server_phase:
            noise = self.get_current_noise(
                self.preferences.server_config, phase=Phase.SERVER
            )
            self.preferences.server_config.noise_multiplier = noise
            self.preferences.server_config.epsilon = None

        self.load_test_data()

        self.nodes = self.create_nodes()

        # We want to be sure that the number of nodes that we
        # sample at each iteration is always less or equal to the
        # total number of nodes.
        self.sampled_nodes = min(self.sampled_nodes, len(self.nodes))
        if self.preferences.wandb:
            self.wandb = Utils.configure_wandb(
                group="Orchestrator", preferences=self.preferences
            )

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
            test_loader=self.test_set,
            device=self.device,
        )

        # This is the case in which we perform a training that involves
        # both the P2P and the Server phase
        if self.preferences.task_type == "p2p_and_server":
            if self.p2p_phase:
                logger.info("Orchestrating P2P phase")
                # In this case we want to perform a P2P training
                self.orchestrate_nodes(phase=Phase.P2P)
            if self.server_phase:
                logger.info("Orchestrating Server phase")
                # In this case we want to perform a server training
                self.orchestrate_nodes(phase=Phase.SERVER)

        # In this case we perform a training that involves
        # both the P2P and the Server phase but we want to
        # perform the server phase first and then the P2P
        elif self.preferences.task_type == "inverse_p2p_and_server":
            if self.server_phase:
                logger.info("Orchestrating Server phase")
                # In this case we want to perform a server training
                self.orchestrate_nodes(phase=Phase.SERVER)
            if self.p2p_phase:
                logger.info("Orchestrating P2P phase")
                # In this case we want to perform a P2P training
                self.orchestrate_nodes(phase=Phase.P2P)
        # In this case we only perform P2P
        elif self.preferences.task_type == "p2p":
            logger.info("Orchestrating P2P phase")
            # In this case we want to perform a P2P training
            self.orchestrate_nodes(phase=Phase.P2P)
        # This is the classic FL with the server
        elif self.preferences.task_type == "fl":
            logger.info("Orchestrating Server phase")
            # In this case we want to perform a server training
            self.orchestrate_nodes(phase=Phase.SERVER)

        logger.debug("Training finished")

        if self.preferences.wandb:
            Utils.finish_wandb(wandb_run=self.wandb)

    def get_current_noise(self, configuration_phase, phase: str):
        if not configuration_phase.epsilon:
            if configuration_phase.noise_multiplier:
                return configuration_phase.noise_multiplier
            return 0
        # We need to understand the noise that we need to add based
        # on the epsilon that we want to guarantee
        max_noise = 0
        for cluster_name in range(self.preferences.data_split_config.num_clusters):
            for node_name in range(self.preferences.data_split_config.num_nodes):
                if self.preferences.public_private_experiment and phase == Phase.P2P:
                    if self.preferences.dataset_p2p:
                        # When I have the P2P phase and during that phase I don't use differential privacy
                        # then I have to use the public dataset. Instead, when I use DP, I'll use the
                        # private training dataset even during the P2P phase
                        dataset = DataLoader().load_splitted_dataset(
                            f"{self.preferences.data_split_config.store_path}/cluster_{cluster_name}_node_{node_name}_{self.preferences.dataset_p2p}_train.pt",
                        )
                    else:
                        dataset = DataLoader().load_splitted_dataset(
                            f"{self.preferences.data_split_config.store_path}/cluster_{cluster_name}_node_{node_name}_public_train.pt",
                        )
                elif (
                    self.preferences.public_private_experiment and phase == Phase.SERVER
                ):
                    if self.preferences.dataset_server:
                        dataset = DataLoader().load_splitted_dataset(
                            f"{self.preferences.data_split_config.store_path}/cluster_{cluster_name}_node_{node_name}_{self.preferences.dataset_server}_train.pt",
                        )
                    else:
                        dataset = DataLoader().load_splitted_dataset(
                            f"{self.preferences.data_split_config.store_path}/cluster_{cluster_name}_node_{node_name}_private_train.pt",
                        )
                else:
                    dataset = DataLoader().load_splitted_dataset(
                        f"{self.preferences.data_split_config.store_path}/cluster_{cluster_name}_node_{node_name}_train.pt",
                    )

            model_noise = Utils.get_model(preferences=self.preferences).to(self.device)

            # get the training dataset of one of the clients
            dataset_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.preferences.p2p_config.batch_size
                if phase == Phase.P2P
                else self.preferences.server_config.batch_size,
                shuffle=False,
                num_workers=0,
            )

            privacy_engine = PrivacyEngine(accountant="rdp")
            optimizer_noise = Utils.get_optimizer(
                preferences=self.preferences,
                model=model_noise,
                phase=phase,
            )

            epochs = (
                self.preferences.server_config.local_training_epochs
                * self.preferences.server_config.fl_rounds
                if phase == Phase.SERVER
                else self.preferences.p2p_config.local_training_epochs
                * self.preferences.p2p_config.fl_rounds
            )

            (
                _,
                private_optimizer,
                _,
            ) = privacy_engine.make_private_with_epsilon(
                module=model_noise,
                optimizer=optimizer_noise,
                data_loader=dataset_loader,
                epochs=epochs,
                target_epsilon=self.preferences.p2p_config.epsilon
                if phase == Phase.P2P
                else self.preferences.server_config.epsilon,
                target_delta=self.preferences.hyperparameters_config.delta,
                max_grad_norm=self.preferences.hyperparameters_config.max_grad_norm,
            )
            max_noise = max(max_noise, private_optimizer.noise_multiplier)
            print(
                f"Node {node_name} Cluster {cluster_name} -- {private_optimizer.noise_multiplier}"
            )

        print(f">>>>> FINALE {max_noise}")

        return max_noise

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
            # if self.p2p_phase:
            #     self.p2p_weights[cluster_id] = Utils.get_parameters(self.model)

        return nodes

    def orchestrate_nodes(self, phase: Phase) -> None:
        logger.info(
            f"Orchestrating {phase} phase - {self.fl_rounds_p2p if phase == Phase.P2P else self.fl_rounds_server} Iterations - {self.local_epochs_p2p if phase == Phase.P2P else self.local_epochs_server} Local Epochs - Learning Rate {self.preferences.p2p_config.lr if phase == Phase.P2P else self.preferences.server_config.lr}"
        )
        iterations = self.fl_rounds_p2p if phase == Phase.P2P else self.fl_rounds_server

        self.model.to("cuda:0")
        aggregated_epsilon = 0

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
                    if (phase == Phase.P2P and iteration != 0) or (
                        phase == Phase.SERVER
                        and iteration == 0
                        and self.p2p_phase
                        and self.p2p_weights != {}
                    ):
                        print(
                            f"Using P2P weights in {iteration} iteration, {phase} phase"
                        )
                        weights = self.p2p_weights
                        if phase == Phase.SERVER:
                            logger.info("Server Phase with P2P weights")
                    else:
                        # In this case we get the weights from the model
                        # that is currently on the server. This could happen when
                        # we perform a server training or when we are in the first
                        # iteration of the P2P training and we want to use the weights
                        # that we have computed in the server phase (inverted_p2P_and_server)
                        weights = Utils.get_parameters(self.model)
                        if phase == Phase.P2P:
                            logger.info(
                                "Starting the P2P Phase with weights computed in the Server Phase"
                            )

                    results = [
                        pool.apply_async(
                            start_train,
                            (
                                phase,
                                self.preferences,
                                weights[node.cluster_id]
                                if (phase == Phase.P2P and iteration != 0)
                                or (
                                    phase == Phase.SERVER
                                    and iteration == 0
                                    and self.p2p_phase
                                    and self.p2p_weights != {}
                                )
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
                            metrics["cluster_id"] = cluster_id
                            metrics["node_id"] = node_id
                            metrics_list.append(metrics)

                        else:
                            all_weights.append(weights)
                            num_examples_list.append(num_examples)
                            metrics_list.append(metrics)

                if phase == Phase.P2P:
                    validation_accuracy_clusters = []
                    test_accuracy_clusters = []
                    for cluster_id in all_weights_p2p:
                        aggregated_weights = Utils.aggregate_weights(
                            all_weights_p2p[cluster_id],
                            num_examples_list_p2p[cluster_id],
                        )
                        self.p2p_weights[cluster_id] = aggregated_weights
                        metrics_cluster = []
                        for metric in metrics_list:
                            if metric["cluster_id"] == cluster_id:
                                metrics_cluster.append(metric)

                        aggregated_validation_accuracy_cluster = np.mean(
                            [
                                metric["validation_accuracy"].item()
                                for metric in metrics_list
                            ],
                        )
                        aggregated_test_accuracy_cluster = np.mean(
                            [metric["test_accuracy"].item() for metric in metrics_list],
                        )
                        validation_accuracy_clusters.append(
                            aggregated_validation_accuracy_cluster
                        )
                        test_accuracy_clusters.append(aggregated_test_accuracy_cluster)
                        if self.preferences.wandb:
                            Utils.log_metrics_to_wandb(
                                wandb_run=self.wandb,
                                metrics={
                                    "FL Round P2P": iteration,
                                    "FL Round": iteration,
                                    f"Aggregated Test Accuracy cluster {cluster_id}": aggregated_test_accuracy_cluster,
                                    f"Aggregated Validation Accuracy {cluster_id}": aggregated_validation_accuracy_cluster,
                                },
                            )

                    aggregated_epsilon = max(
                        [metric["epsilon"] for metric in metrics_list],
                    )
                    aggregated_training_accuracy = np.mean(
                        [metric["accuracy"].item() for metric in metrics_list]
                    )
                    aggregated_training_loss = np.mean(
                        [metric["loss"].item() for metric in metrics_list]
                    )
                    aggregated_validation_loss = np.mean(
                        [metric["validation_loss"].item() for metric in metrics_list]
                    )
                    aggregated_test_loss = np.mean(
                        [metric["test_loss"].item() for metric in metrics_list]
                    )
                    aggregated_validation_accuracy = np.mean(
                        [
                            metric["validation_accuracy"].item()
                            for metric in metrics_list
                        ]
                    )
                    aggregated_test_accuracy = np.mean(
                        [metric["test_accuracy"].item() for metric in metrics_list]
                    )

                    if self.preferences.wandb:
                        Utils.log_metrics_to_wandb(
                            wandb_run=self.wandb,
                            metrics={
                                "Train loss": aggregated_training_loss,
                                "Train accuracy": aggregated_training_accuracy,
                                "EPSILON": aggregated_epsilon,
                                "FL Round P2P": iteration,
                                "FL Round": iteration,
                                "Average Nodes Test Loss": aggregated_test_loss,
                                "Average Nodes Test Accuracy": aggregated_test_accuracy,
                                "Average Nodes Validation Loss": aggregated_validation_loss,
                                "Average Nodes Validation Accuracy": aggregated_validation_accuracy,
                                "Average Cluster Test Accuracy": np.mean(
                                    test_accuracy_clusters
                                ),
                                "Average Cluster Validation Accuracy": np.mean(
                                    validation_accuracy_clusters
                                ),
                                "Custom Metric": np.mean(
                                    aggregated_validation_accuracy
                                ),
                            },
                        )
                else:
                    aggregated_weights = Utils.aggregate_weights(
                        all_weights,
                        num_examples_list,
                    )
                    logger.debug("Aggregated all the weights during Server phase")

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
                    aggregated_validation_loss = np.mean(
                        [metric["validation_loss"].item() for metric in metrics_list]
                    )
                    aggregated_test_loss = np.mean(
                        [metric["test_loss"].item() for metric in metrics_list]
                    )
                    aggregated_validation_accuracy = np.mean(
                        [
                            metric["validation_accuracy"].item()
                            if isinstance(metric["validation_accuracy"], torch.Tensor)
                            else metric["validation_accuracy"]
                            for metric in metrics_list
                        ]
                    )
                    aggregated_test_accuracy = np.mean(
                        [metric["test_accuracy"].item() for metric in metrics_list]
                    )

                    if self.preferences.wandb:
                        Utils.log_metrics_to_wandb(
                            wandb_run=self.wandb,
                            metrics={
                                "Train loss": aggregated_training_loss,
                                "Train accuracy": aggregated_training_accuracy,
                                "FL_round": iteration + self.fl_rounds_p2p,
                                "EPSILON": aggregated_epsilon,
                                "FL Round Server": iteration,
                                "FL Round": iteration + self.fl_rounds_p2p,
                                "Aggregated Test Loss": aggregated_test_loss,
                                "Aggregated Test Accuracy": aggregated_test_accuracy,
                                "Aggregated Validation Loss": aggregated_validation_loss,
                                "Aggregated Validation Accuracy": aggregated_validation_accuracy,
                                "Custom Metric": np.mean(
                                    aggregated_validation_accuracy,
                                ),
                            },
                        )

                # logger.debug("Computed the average")
                # if phase == Phase.SERVER:
                #     (
                #         loss,
                #         accuracy,
                #         fscore,
                #         precision,
                #         recall,
                #         test_accuracy_per_class,
                #     ) = Learning.evaluate_model(
                #         model=self.model,
                #         test_loader=self.test_set,
                #         device=self.device,
                #     )

                #     metrics = {
                #         "test loss": loss,
                #         "test accuracy": accuracy,
                #         "test fscore": fscore,
                #         "test precision": precision,
                #         "test recall": recall,
                #         "test_accuracy_per_class": test_accuracy_per_class,
                #         "FL_round": iteration + self.fl_rounds_p2p,
                #         "FL Round Server": iteration,
                #     }

                #     self.log_metrics(
                #         metrics,
                #     )

                #     (
                #         loss_validation,
                #         accuracy_validation,
                #         _,
                #         _,
                #         _,
                #         _,
                #     ) = Learning.evaluate_model(
                #         model=self.model,
                #         test_loader=self.validation_set,
                #         device=self.device,
                #     )

                #     metrics = {
                #         "Loss_validation": loss_validation,
                #         "Accuracy_validation": accuracy_validation,
                #         "custom_metric": loss_validation
                #         + self.fl_rounds_p2p
                #         + self.local_epochs_p2p,
                #         "FL_round": iteration + self.fl_rounds_p2p,
                #         "FL Round Server": iteration,
                #     }
                #     self.log_metrics(
                #         metrics,
                #     )

                # else:
                #     # pass
                #     self.evaluate_p2p(
                #         iteration=iteration, aggregated_epsilon=aggregated_epsilon
                #     )

    # def evaluate_p2p(self, iteration: int, aggregated_epsilon: float):
    #     average_validation_loss = []
    #     average_accuracy_validation = []
    #     for cluster_id, weights in self.p2p_weights.items():
    #         losses = []
    #         accuracies = []
    #         fscores = []
    #         precisions = []
    #         recalls = []
    #         cluster_model = copy.deepcopy(self.model)
    #         Utils.set_params(cluster_model, self.p2p_weights[cluster_id])
    #         data_validation_node = torch.load(
    #             f"{self.preferences.data_split_config.store_path}/validation_cluster_{cluster_id}.pt"
    #         )
    #         validation_node = torch.utils.data.DataLoader(
    #             data_validation_node,
    #             batch_size=256,
    #             shuffle=False,
    #             num_workers=0,
    #         )
    #         (
    #             loss_validation,
    #             accuracy_validation,
    #             _,
    #             _,
    #             _,
    #             _,
    #         ) = Learning.evaluate_model(
    #             model=cluster_model,
    #             test_loader=validation_node,
    #             device=self.device,
    #         )

    #         average_validation_loss.append(loss_validation)
    #         average_accuracy_validation.append(accuracy_validation)

    #         for node in self.cluster_nodes[cluster_id]:
    #             data_test_node = torch.load(
    #                 f"{self.preferences.data_split_config.store_path}/test_cluster_{cluster_id}.pt"
    #             )

    #             test_node = torch.utils.data.DataLoader(
    #                 data_test_node,
    #                 batch_size=256,
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

    #         metrics = {
    #             f"test loss {cluster_id}": np.mean(losses),
    #             f"cluster test accuracy {cluster_id}": np.mean(accuracies),
    #             f"test fscore {cluster_id}": np.mean(fscores),
    #             f"test precision {cluster_id}": np.mean(precisions),
    #             f"test recall {cluster_id}": np.mean(recalls),
    #             f"FL Round P2P": iteration,
    #             "FL_round": iteration,
    #         }
    #         self.log_metrics(metrics)

    #         losses = []
    #         accuracies = []
    #         fscores = []
    #         precisions = []
    #         recalls = []
    #         for node in self.cluster_nodes[cluster_id]:
    #             data_test_node = torch.load(
    #                 f"{self.preferences.data_split_config.store_path}/cluster_{cluster_id}_node_{node.node_id}_test.pt"
    #             )

    #             test_node = torch.utils.data.DataLoader(
    #                 data_test_node,
    #                 batch_size=256,
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
    #                 f"FL Round P2P": iteration,
    #                 "FL_round": iteration,
    #             }
    #             self.log_metrics(metrics)
    #         metrics = {
    #             f"Average test loss {cluster_id}": np.mean(losses),
    #             f"Average cluster accuracy {cluster_id}": np.mean(accuracies),
    #             f"Average test fscore {cluster_id}": np.mean(fscores),
    #             f"Average test precision {cluster_id}": np.mean(precisions),
    #             f"Average test recall {cluster_id}": np.mean(recalls),
    #             f"FL Round P2P": iteration,
    #             "FL_round": iteration,
    #         }
    #         self.log_metrics(metrics)

    #     average_validation_loss = np.mean(average_validation_loss)
    #     average_accuracy_validation = np.mean(average_accuracy_validation)

    #     metrics = {
    #         "Loss_validation": average_validation_loss,
    #         "Accuracy_validation": average_accuracy_validation,
    #         "FL Round P2P": iteration,
    #         "FL_round": iteration,
    #         "Custom_metric": custom_metric,
    #     }
    #     self.log_metrics(metrics)

    def load_test_data(self) -> None:
        """This function loads the test data from disk."""
        data: torch.utils.data.DataLoader[Any] = None
        print(
            f"{self.preferences.data_split_config.store_path}/{self.preferences.data_split_config.server_test_set}"
        )
        # Load the test set dataset using the pytorch function
        test_data = torch.load(
            f"{self.preferences.data_split_config.store_path}/{self.preferences.data_split_config.server_test_set}"
        )
        validation_data = torch.load(
            f"{self.preferences.data_split_config.store_path}/server_validation_set.pt"
        )

        self.test_set = torch.utils.data.DataLoader(
            test_data,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        self.validation_set = torch.utils.data.DataLoader(
            validation_data,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        if self.preferences.debug:
            targets = []
            for _, data in enumerate(self.test_set, 0):
                targets.append(data[-1])
            targets = [item.item() for sublist in targets for item in sublist]
            logger.info(f"Test set: {Counter(targets)}")

    def log_metrics(self, metrics: dict):
        logger.debug(metrics)
        logger.debug("Metrics computed")
        logger.debug("Logging the metrics on wandb")
        if self.preferences.wandb:
            Utils.log_metrics_to_wandb(wandb_run=self.wandb, metrics=metrics)
        logger.debug("Metrics logged")
