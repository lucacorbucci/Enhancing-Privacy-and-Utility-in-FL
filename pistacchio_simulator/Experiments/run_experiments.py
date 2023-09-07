
import torch
from pydantic.tools import parse_obj_as
from torch import nn

from pistacchio_simulator.Components.Orchestrator.orchestrator import Orchestrator
from pistacchio_simulator.Exceptions.errors import InvalidDatasetNameError
from pistacchio_simulator.Models.celeba import CelebaGenderNet, CelebaNet
from pistacchio_simulator.Models.fashion_mnist import FashionMnistNet
from pistacchio_simulator.Models.mnist import MnistNet
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.task import Task, TaskType


class Experiment:
    @staticmethod
    def get_model(preferences: Preferences) -> nn.Module:
        """This function is used to get the model.

        Returns
        -------
            nn.Module: the model
        """
        model = None
        if preferences.dataset == "mnist":
            model = MnistNet()
        elif preferences.dataset == "cifar10":
            model = Experiment.get_model_to_fine_tune()
            preferences.fine_tuning = True
        elif preferences.dataset == "celeba":
            model = CelebaNet()
        elif preferences.dataset == "celeba_gender":
            model = CelebaGenderNet()
        elif preferences.dataset == "fashion_mnist":
            model = FashionMnistNet()
        elif preferences.dataset == "imaginette":
            model = Experiment.get_model_to_fine_tune()
            preferences.fine_tuning = True
        else:
            raise InvalidDatasetNameError("Invalid dataset name")
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
            InvalidDatasetNameError: if the dataset name is not valid

        Returns
        -------
            int: 0 if everything went well
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # set_start_method("spawn")
        # torch.multiprocess.set_sharing_strategy("file_system")

        preferences = parse_obj_as(Preferences, config)
        task = Task(preferences.task_type)
        if task.task_type == TaskType.FEDERATEDLEARNING:
            Experiment.run_federated_learning_experiment(preferences)
        # elif task_type == TaskType.CONTRIBUTION:
        #     Experiment.run_contribution_experiment(preferences)
        # elif task_type == TaskType.FAIRNESS:
        #     Experiment.run_fairness_experiment(preferences)
        else:
            raise NotImplementedError("Task not implemented")

        return 0
