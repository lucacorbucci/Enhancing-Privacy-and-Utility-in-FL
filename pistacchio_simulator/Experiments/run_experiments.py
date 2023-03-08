from multiprocessing import set_start_method

import torch
from pistacchio_simulator.Components.Orchestrator.orchestrator import Orchestrator
from pistacchio_simulator.Exceptions.errors import InvalidDatasetErrorNameError
from pistacchio_simulator.Models.celeba import CelebaGenderNet, CelebaNet
from pistacchio_simulator.Models.fair_face import FairFace
from pistacchio_simulator.Models.fashion_mnist import FashionMnistNet
from pistacchio_simulator.Models.imaginette import Imaginette
from pistacchio_simulator.Models.mnist import MnistNet
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.task import Task, TaskType
from torch import nn


class Experiment:
    @staticmethod
    def get_model(preferences: Preferences) -> nn.Module:
        """This function is used to get the model.

        Returns
        -------
            nn.Module: the model
        """
        model = None
        if preferences.dataset_name == "mnist":
            model = MnistNet()
        elif preferences.dataset_name == "cifar10":
            model = Experiment.get_model_to_fine_tune()
            preferences.fine_tuning = True
        elif preferences.dataset_name == "celeba":
            model = CelebaNet()
        elif preferences.dataset_name == "celeba_gender":
            model = CelebaGenderNet()
        elif preferences.dataset_name == "fashion_mnist":
            model = FashionMnistNet()
        elif preferences.dataset_name == "imaginette":
            model = Imaginette().get_model_to_fine_tune()
            preferences.fine_tuning = True
        elif preferences.dataset_name == "fair_face":
            model = FairFace.get_model_to_fine_tune()
            preferences.fine_tuning = True
        else:
            raise InvalidDatasetErrorNameError("Invalid dataset name")
        return model

    @staticmethod
    def run_federated_learning_experiment(preferences: Preferences) -> None:
        """Run a federated learning experiment.

        Args:
            preferences (Preferences): _description_
        """
        model = Experiment.get_model(preferences)

        orchestrator = Orchestrator(
            preferences=preferences,
            model=model,
        )
        orchestrator.launch_orchestrator()

    @staticmethod
    def run_contribution_experiment(preferences: Preferences) -> None:
        iterations = (
            preferences.data_split_config["num_clusters"]
            * preferences.data_split_config["num_nodes"]
            + 1
        )
        for iteration in range(iterations):
            model = Experiment.get_model(preferences)

            orchestrator = Orchestrator(
                preferences=preferences, model=model, iteration=iteration
            )
            orchestrator.launch_orchestrator()

    @staticmethod
    def run_fairness_experiment(preferences: Preferences) -> None:
        """Run an experiment to verify the fairness of the
        federated learning model.

        Args:
            preferences (Preferences): preferences specified in the config file

        Raises
        ------
            NotImplementedError: _description_
        """
        model = Experiment.get_model(preferences)
        Experiment.launch_classic_experiment(preferences, model)

    @staticmethod
    def run(config: dict) -> int:
        """This function is used to prepare all the things
        that are needed to run an experiment.

        Args:
            config (dict): preferences file

        Raises
        ------
            InvalidDatasetErrorNameError: if the dataset name is not valid

        Returns
        -------
            int: 0 if everything went well
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")

        preferences = Preferences.generate_from_json(config)
        if preferences.task.task_type == TaskType.FEDERATEDLEARNING:
            Experiment.run_federated_learning_experiment(preferences)
        elif preferences.task.task_type == TaskType.CONTRIBUTION:
            Experiment.run_contribution_experiment(preferences)
        elif preferences.task.task_type == TaskType.FAIRNESS:
            Experiment.run_fairness_experiment(preferences)
        return 0
