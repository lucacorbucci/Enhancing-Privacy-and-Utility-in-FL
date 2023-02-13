import os
import random
import time
from collections import OrderedDict
from types import ModuleType
from typing import Any, Mapping, TypeVar

import torch
from dotenv import load_dotenv
from torch import Tensor

import wandb
from pistacchio.Utils.preferences import Preferences


TDestination = TypeVar("TDestination", bound=Mapping[str, Tensor])


class Utils:
    """Define the Utils class."""

    @staticmethod
    def compute_average(shared_data: dict) -> dict[Any, Any]:
        """This function computes the average of the weights.

        Args:
            shared_data (Dict): weights received from the other nodes of the cluster

        Returns
        -------
            OrderedDict: the average of the weights
        """
        models = list(shared_data.values())

        results: OrderedDict = OrderedDict()

        for model in models:
            for key in model:
                if results.get(key) is None:
                    if torch.cuda.is_available():
                        model[key] = model[key].to("cuda:0")

                    results[key] = model[key]
                else:
                    if torch.cuda.is_available():
                        model[key] = model[key].to("cuda:0")
                    results[key] = results[key].add(model[key])

        for key in results:
            results[key] = torch.div(results[key], len(models))
        return results

    @staticmethod
    def compute_distance_from_mean(shared_data: dict, average_weights: dict) -> dict:
        """This function takes as input the weights received from all the nodes and
        the average computed by the server.
        It computes the distance between the average and
        the weights received from each node per each layer.
        Then it computes the mean of the distances per each layers.
        The function returns a dictionary containing the mean of
        the distances per each node.

        Args:
            shared_data (Dict): the weights received from the other nodes of the cluster
            average_weights (Dict): the average of the weights

        Returns
        -------
            Dict: a dictionary containing the mean of the distances per each node
        """
        distances = {}
        for node_name, models in shared_data.items():
            mean = []
            for layer_name in models:

                mean.append(
                    torch.mean(
                        torch.subtract(models[layer_name], average_weights[layer_name]),
                    ),
                )

            distances[node_name] = torch.mean(torch.stack(mean))

        return distances

    @staticmethod
    def get_current_time() -> str:
        """This function is used to get the current time.

        Returns
        -------
            str: current time
        """
        return str(time.strftime("%H_%M_%S", time.localtime()))

    @staticmethod
    def get_run_name(preferences: Preferences) -> str:
        """This function is used to get the experiment name.

        Args:
            preferences (_type_): preferences for the run

        Returns
        -------
            str: experiment name
        """
        if preferences.removed_node_id:
            return f"removed_node_{str(preferences.removed_node_id)}"

        diff_private = (
            ""
            if not preferences.server_config["differential_privacy_server"]
            else "differentiallyprivate_"
        )
        noise_multiplier = (
            ""
            if not preferences.hyperparameters.get("noise_multiplier", None)
            else f"noise_multiplier_{preferences.hyperparameters['noise_multiplier']}_"
        )
        max_grad_norm = (
            ""
            if not preferences.hyperparameters.get("max_grad_norm", None)
            else f"max_grad_norm_{preferences.hyperparameters['max_grad_norm']}_"
        )
        return (
            ""
            + str(preferences.dataset_name)
            + "_"
            + str(preferences.data_split_config["num_nodes"])
            + "_nodes_"
            + str(preferences.data_split_config["num_clusters"])
            + "_clusters_"
            + diff_private
            + noise_multiplier
            + max_grad_norm
            + str(preferences.experiment_name)
        )

    @staticmethod
    def configure_wandb(group: str, preferences: Preferences) -> ModuleType:
        """This function is used to configure wandb.

        Args:
            group (_type_): group name
            preferences (_type_): preferences for the run


        Returns
        -------
            wandb: _description_
        """
        load_dotenv()

        wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
        wandb_entity = os.getenv("WANDB_ENTITY")
        config_dictionary = preferences.hyperparameters
        config_dictionary["num_nodes"] = preferences.data_split_config["num_nodes"]
        config_dictionary["num_clusters"] = preferences.data_split_config[
            "num_clusters"
        ]

        if preferences.p2p_config:
            config_dictionary["step_P2P"] = preferences.p2p_config[
                "num_communication_round_pre_training"
            ]
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            config=config_dictionary,
            group=group,
            tags=preferences.wandb_tags,
        )
        if wandb.run:
            wandb.run.name = Utils.get_run_name(preferences=preferences)
        return wandb

    @staticmethod
    def log_wandb_artifact(wandb_run: ModuleType, model_file: str) -> None:
        """Log an artifact to wanbd.

        Args:
            wandb_run (ModuleType): wandb object
            model_file (str): model to be logged
        """
        if wandb_run:
            artifact = wandb_run.Artifact("model", type="model")
            artifact.add_file(model_file)
            wandb_run.log_artifact(artifact)

    @staticmethod
    def log_to_wandb(
        wandb_run: ModuleType,
        accuracy: float,
        recall: float,
        precision: float,
        fscore: float,
        loss: float,
        epoch: int,
        epsilon: float,
        distances: dict,
        results: dict,
        test_accuracy_per_class: list | None = None,
        reduced_classes: list | None = None,
        percentage_underrepresented_classes: list | None = None,
    ) -> None:
        """Log metrics to wandb.

        Args:
            wandb_run (_type_): wandb object
            accuracy (_type_): accuracy of the model
            recall (): recall of the model
            precision (_type_): precision of the model
            fscore (_type_): fscore of the model
            loss (_type_): loss of the model
            epoch (_type_): number of epochs
            epsilon (_type_): privacy budget
            distances (dict): distances from the average model
            results (dict): _description_
            test_accuracy_per_class (list | None, optional): _description_.
                Defaults to None.
            reduced_classes (list | None, optional): _description_. Defaults to None.
            percentage_underrepresented_classes (list | None, optional): _description_.
                Defaults to None.
        """
        if wandb_run:
            wandb_run.log({"accuracy": accuracy, "epoch": epoch})
            wandb_run.log({"loss": loss, "epoch": epoch})
            wandb_run.log({"recall": recall, "epoch": epoch})
            wandb_run.log({"precision": precision, "epoch": epoch})
            wandb_run.log({"fscore": fscore, "epoch": epoch})
            wandb_run.log({"epsilon": epsilon, "epoch": epoch})
            wandb_run.log({"distances": distances, "epoch": epoch})
            if results:
                wandb_run.log(results)
            if test_accuracy_per_class is not None:
                wandb_run.log({"test_accuracy_per_class": test_accuracy_per_class})
            # Optional
            if reduced_classes:
                wandb_run.log({"reduced_classes": reduced_classes})
            if percentage_underrepresented_classes:
                wandb_run.log(
                    {
                        "percentage_underrepresented_classes": percentage_underrepresented_classes,
                    },
                )

    @staticmethod
    def log_epsilon_to_wandb(
        wandb_run: ModuleType,
        epsilon: float,
        accuracy: float,
        model: torch.nn.Module,
        epoch: int,
    ) -> None:
        """Log the epsilon list to wandb.

        Args:
            wandb_run (_type_): wandb object
            epsilon (_type_): privacy budget
            accuracy (_type_): accuracy of the model
            model (_type_): model name
            epoch (_type_): number of epochs
        """
        if wandb_run:
            wandb_run.log({"epsilon": epsilon, "epoch": epoch})
            wandb_run.log({"accuracy": accuracy, "epoch": epoch})

            wandb_run.watch(model, log="all", log_freq=10)

    @staticmethod
    def finish_wandb(wandb_run: ModuleType) -> None:
        """Kill wandb.

        Args:
            wandb_run (ModuleType): wandb object
        """
        wandb_run.finish()

    @staticmethod
    def change_weight_names(weights: OrderedDict, string_to_add: str) -> OrderedDict:
        """This function is used to change the name of the weights.

        Args:
            weights (OrderedDict): weights of the model
            string_to_add (str): string to add to the name of the weights

        Returns
        -------
            OrderedDict: updated weights
        """
        new_weights = OrderedDict()
        for key, value in weights.items():
            new_weights[string_to_add + key] = value
        return new_weights

    @staticmethod
    def shuffle_lists(first_list: list, second_list: list) -> "zip[Any]":
        """Shuffle two lists in the same way and return them.

        Args:
            a (list): the first list to shuffle
            b (list): the second list

        Returns
        -------
            Tuple[list, list]: The two shuffled lists
        """
        if len(first_list) != len(second_list):
            raise ValueError("The two lists must have the same length")
        if len(first_list) == 0 or len(second_list) == 0:
            raise ValueError("The two lists must not be empty")

        zipped_list = list(zip(first_list, second_list))
        random.shuffle(zipped_list)
        return zip(*zipped_list)
        return zip(*zipped_list)
