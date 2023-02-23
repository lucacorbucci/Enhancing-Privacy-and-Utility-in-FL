import sys
import warnings
from abc import ABC
from collections import Counter
from typing import Any, Generic, Mapping, TypeVar

import numpy as np
import torch
from loguru import logger
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn, optim
import gc
from pistacchio_simulator.Exceptions.errors import (
    InvalidDatasetError,
    NotYetInitializedFederatedLearningError,
    NotYetInitializedPreferencesError,
)
from pistacchio_simulator.Utils.data_loader import DataLoader
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences


warnings.filterwarnings("ignore")

TDestination = TypeVar("TDestination", bound=Mapping[str, Any])


logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)


class FederatedModel(ABC, Generic[TDestination]):
    """This class is used to create the federated model that
    we will train. It returns a different model based on the
    dataset we want to use.
    """

    def __init__(
        self,
        dataset_name: str,
        node_name: str,
        preferences: Preferences | None = None,
    ) -> None:
        """Initialize the Federated Model.

        Args:
            dataset_name (str): Name of the dataset we want to use
            node_name (str): name of the node we are working on
            preferences (Preferences, optional): Configuration for this run. Defaults to None.

        Raises
        ------
            InvalidDatasetErrorNameError: _description_
        """
        self.device = None
        self.optimizer: optim.Optimizer = None
        self.dataset_name = dataset_name
        self.node_name = node_name
        self.mixed = False
        self.privacy_engine = None
        if node_name != "server":
            try:
                self.training_set = DataLoader().load_splitted_dataset_train(
                    f"../data/{dataset_name}/federated_split/train_set/{self.node_name}_split",
                )
                self.test_set = DataLoader().load_splitted_dataset_test(
                    f"../data/{dataset_name}/federated_split/test_set/{self.node_name}_split",
                )
            except Exception as error:
                raise InvalidDatasetError from error
        self.trainloader = None
        self.testloader = None
        self.preferences = preferences
        self.diff_privacy_initialized = False
        gpus = preferences.gpu_config
        expected_len = 11
        if len(node_name) == expected_len:
            if gpus:
                gpu_name = gpus[int(node_name[10]) % len(gpus)]
            self.device = torch.device(
                gpu_name if torch.cuda.is_available() and gpus else "cpu",
            )
            logger.debug(f"Running on {self.device}")

        self.net = None

    def init_model(self, net: nn.Module) -> None:
        """Initialize the Federated Model before the use of it.

        Args:
            net (nn.Module): model we want to use
        """
        if not self.trainloader and not self.testloader and self.node_name != "server":
            self.trainloader, self.testloader = self.load_data()

        self.add_model(net)
        if self.net:

            params_to_update = []
            for _, param in self.net.named_parameters():
                if param.requires_grad is True:
                    params_to_update.append(param)

            # self.optimizer = torch.optim.Adam(
            # self.optimizer = torch.optim.SGD(
            self.optimizer = optim.RMSprop(
                params_to_update,
                lr=self.preferences.hyperparameters["lr"],
            )

    def add_model(self, model: nn.Module) -> None:
        """This function adds the model passed as parameter
        as the model used in the FederatedModel.

        Args:
            model (nn.Module): Model we want to inject in the Federated Model
        """
        self.net = model
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

    def load_data(
        self,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load training and test dataset.

        Returns
        -------
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: training and test set

        Raises
        ------
            Exception: Preference is not initialized
        """
        if self.preferences:
            batch_size = self.preferences.hyperparameters["batch_size"]
            trainloader = torch.utils.data.DataLoader(
                self.training_set,
                batch_size=16,
                shuffle=True,
                num_workers=0,
            )

            testloader = torch.utils.data.DataLoader(
                self.test_set,
                batch_size=16,
                shuffle=False,
                num_workers=0,
            )

            if self.preferences and self.preferences.debug:
                self.print_data_stats(trainloader)
            return trainloader, testloader

        raise NotYetInitializedPreferencesError

    def print_data_stats(self, trainloader: torch.utils.data.DataLoader) -> None:
        """Debug function used to print stats about the loaded datasets.

        Args:
            trainloader (torch.utils.data.DataLoader): training set
        """
        num_examples = {
            "trainset": len(self.training_set),
            "testset": len(self.test_set),
        }
        targets = []
        for _, data in enumerate(trainloader, 0):
            targets.append(data[1])
        targets = [item.item() for sublist in targets for item in sublist]
        logger.info(f"{self.node_name}, {Counter(targets)}")
        logger.info(f"Training set size: {num_examples['trainset']}")
        logger.info(f"Test set size: {num_examples['testset']}")

    def get_weights_list(self) -> list[float]:
        """Get the parameters of the network.

        Raises
        ------
            Exception: if the model is not initialized it raises an exception

        Returns
        -------
            List[float]: parameters of the network
        """
        if self.net:
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        raise NotYetInitializedFederatedLearningError

    def get_weights(self) -> TDestination:
        """Get the weights of the network.

        Raises
        ------
            Exception: if the model is not initialized it raises an exception

        Returns
        -------
            _type_: weights of the network
        """
        if self.net:
            return self.net.state_dict()
        raise NotYetInitializedFederatedLearningError

    def update_weights(self, avg_tensors: TDestination) -> None:
        """This function updates the weights of the network.

        Raises
        ------
            Exception: _description_

        Args:
            avg_tensors (_type_): tensors that we want to use in the network
        """
        if self.net:
            self.net.load_state_dict(avg_tensors, strict=True)
        else:
            raise NotYetInitializedFederatedLearningError

    def store_model_on_disk(self) -> None:
        """This function is used to store the trained model
        on disk.

        Raises
        ------
            Exception: if the model is not initialized it raises an exception
        """
        if self.net:
            torch.save(
                self.net.state_dict(),
                "../model_results/model_" + self.node_name + ".pt",
            )
        else:
            raise NotYetInitializedFederatedLearningError

    def train(self) -> tuple[float, torch.tensor]:
        """Train the network and computes loss and accuracy.

        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized

        Returns
        -------
            Tuple[float, float]: Loss and accuracy on the training set.
        """
        if self.net:
            criterion = nn.CrossEntropyLoss()
            running_loss = 0.0
            total_correct = 0
            total = 0
            self.net = self.net.to(self.device)

            self.net.train()

            for _, (data, target) in enumerate(self.trainloader, 0):
                self.optimizer.zero_grad()

                if isinstance(data, list):
                    data = data[0]

                # if torch.cuda.is_available():
                #     gc.collect()
                #     torch.cuda.empty_cache()
                #     with torch.no_grad():
                #         torch.cuda.empty_cache()

                target = target.to(self.device)
                data = data.to(self.device)

                # forward pass, backward pass and optimization
                outputs = self.net(data)
                loss = criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == target).float().sum()
                running_loss += loss.item()
                total_correct += correct
                total += target.size(0)

                if torch.cuda.is_available():
                    # del data
                    # del target
                    # del loss
                    gc.collect()
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        torch.cuda.empty_cache()

            loss = running_loss / len(self.trainloader)
            accuracy = total_correct / total
            logger.info(f"Training loss: {loss}, accuracy: {accuracy}")

            return loss, accuracy
        raise NotYetInitializedFederatedLearningError

    def train_with_differential_privacy(self) -> tuple[float, float, float]:
        """Train the network using differential privacy and computes loss and accuracy.

        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized

        Returns
        -------
            Tuple[float, float]: Loss and accuracy on the training set.
        """
        if self.net:
            max_physical_batch_size = self.preferences.hyperparameters[
                "MAX_PHYSICAL_BATCH_SIZE"
            ]
            # Train the network
            criterion = nn.CrossEntropyLoss()
            running_loss = 0.0
            total_correct = 0
            total = 0

            self.net = self.net.to(self.device)

            self.net.train()

            with BatchMemoryManager(
                data_loader=self.trainloader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=self.optimizer,
            ) as memory_safe_data_loader:
                for _, (data, target) in enumerate(memory_safe_data_loader, 0):
                    self.optimizer.zero_grad()

                    if isinstance(data, list):
                        data = data[0]
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.net(data)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == target).float().sum()

                    loss = criterion(outputs, target)
                    running_loss += loss.item()
                    total_correct += correct
                    total += target.size(0)

                    self.optimizer.zero_grad()
                    self.net.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.net.zero_grad()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            loss = running_loss / len(self.trainloader)
            accuracy = total_correct / total
            logger.info(f"Training loss: {loss}, accuracy: {accuracy}")

            if self.node_name != "server":
                epsilon = self.privacy_engine.accountant.get_epsilon(
                    delta=self.preferences.hyperparameters["DELTA"],
                )

            return loss, accuracy, epsilon

        raise NotYetInitializedFederatedLearningError

    def evaluate_model(self) -> tuple[float, float, float, float, float, list]:
        """Validate the network on the entire test set.

        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized

        Returns
        -------
            Tuple[float, float]: loss and accuracy on the test set.
        """
        with torch.no_grad():
            if self.net:
                self.net.eval()
                criterion = nn.CrossEntropyLoss()
                test_loss = 0
                correct = 0
                total = 0
                y_pred = []
                y_true = []
                losses = []
                with torch.no_grad():
                    for data, target in self.testloader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.net(data)
                        total += target.size(0)
                        test_loss = criterion(output, target).item()
                        losses.append(test_loss)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        y_pred.append(pred)
                        y_true.append(target)

                test_loss = np.mean(losses)
                accuracy = correct / total

                y_true = [item.item() for sublist in y_true for item in sublist]
                y_pred = [item.item() for sublist in y_pred for item in sublist]

                f1score = f1_score(y_true, y_pred, average="macro")
                precision = precision_score(y_true, y_pred, average="macro")
                recall = recall_score(y_true, y_pred, average="macro")

                cm = confusion_matrix(y_true, y_pred)
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                accuracy_per_class = cm.diagonal()

                true_positives = np.diag(cm)
                num_classes = len(list(set(y_true)))

                false_positives = []
                for i in range(num_classes):
                    false_positives.append(sum(cm[:,i]) - cm[i,i])

                false_negatives = []
                for i in range(num_classes):
                    false_negatives.append(sum(cm[i,:]) - cm[i,i])

                true_negatives = []
                for i in range(num_classes):
                    temp = np.delete(cm, i, 0)   # delete ith row
                    temp = np.delete(temp, i, 1)  # delete ith column
                    true_negatives.append(sum(sum(temp)))

                denominator = [sum(x) for x in zip(false_positives, true_negatives)]
                false_positive_rate = [num/den for num, den in zip(false_positives, denominator)]

                denominator = [sum(x) for x in zip(true_positives, false_negatives)]
                true_positive_rate = [num/den for num, den in zip(true_positives, denominator)]

                return (
                    test_loss,
                    accuracy,
                    f1score,
                    precision,
                    recall,
                    accuracy_per_class,
                    true_positive_rate,
                    false_positive_rate
                )

            raise NotYetInitializedFederatedLearningError

    def init_privacy_with_epsilon(self, phase: Phase, epsilon: float) -> None:
        """Initialize differential privacy using the epsilon parameter.

        Args:
            phase (Phase): phase of the training
            EPSILON (float): epsilon parameter for differential privacy

        Raises
        ------
            Exception: Preference is not initialized
        """
        if self.preferences:
            max_grad_norm = self.preferences.hyperparameters["max_grad_norm"]
            delta = self.preferences.hyperparameters["DELTA"]

            if self.privacy_engine:
                (
                    self.net,
                    self.optimizer,
                    self.trainloader,
                ) = self.privacy_engine.make_private_with_epsilon(
                    module=self.net,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    epochs=(
                        self.preferences.server_config[
                            "num_communication_round_with_server"
                        ]
                        if phase == Phase.SERVER
                        else self.preferences.p2p_config[
                            "num_communication_round_pre_training"
                        ]
                    ),
                    target_epsilon=epsilon,
                    target_delta=delta,
                    max_grad_norm=max_grad_norm,
                )
        else:
            raise NotYetInitializedPreferencesError

    def init_privacy_with_noise(self, phase: Phase) -> None:
        """Initialize differential privacy using the noise parameter
        without the epsilon parameter.
        Noise multiplier: the more is higher the more is the noise
        Max grad: the more is higher the less private is training.

        Args:
            phase (Phase): phase of the training
        Raises:
            Exception: Preference is not initialized
        """
        if self.preferences:
            max_grad_norm = self.preferences.hyperparameters["max_grad_norm"]
            if self.privacy_engine:
                self.net.train()
                (
                    self.net,
                    self.optimizer,
                    self.trainloader,
                ) = self.privacy_engine.make_private(
                    module=self.net,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.preferences.hyperparameters[
                        "noise_multiplier"
                    ]
                    if phase == Phase.SERVER
                    else self.preferences.hyperparameters["noise_multiplier_P2P"],
                    max_grad_norm=max_grad_norm,
                )
        else:
            raise NotYetInitializedPreferencesError

    def init_differential_privacy(
        self, phase: Phase
    ) -> tuple[nn.Module, optim.Optimizer, torch.utils.data.DataLoader]:
        """Initialize the differential privacy.

        Args:
            phase (str): phase of the training

        Returns
        -------
            _type_:
        """
        if self.preferences:
            self.diff_privacy_initialized = True
            self.privacy_engine = PrivacyEngine()
            epsilon = (
                self.preferences.hyperparameters.get("EPSILON", None)
                if phase == Phase.SERVER
                else self.preferences.hyperparameters.get("EPSILON_P2P", None)
            )
            if epsilon:
                self.init_privacy_with_epsilon(phase=phase, epsilon=epsilon)
            else:
                self.init_privacy_with_noise(phase=phase)

            return self.net, self.optimizer, self.trainloader
        raise NotYetInitializedPreferencesError
