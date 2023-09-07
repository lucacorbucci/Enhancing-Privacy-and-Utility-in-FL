import sys
import warnings
from abc import ABC
from collections import Counter, OrderedDict
from typing import Any, Generic, Mapping, TypeVar

import numpy as np
import torch
from loguru import logger
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn, optim

from pistacchio_simulator.Exceptions.errors import (
    InvalidDatasetNameError,
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
            InvalidDatasetNameError: _description_
        """
        self.device = None
        self.optimizer: optim.Optimizer = None
        self.dataset_name = dataset_name
        self.node_name = node_name
        self.mixed = False
        self.privacy_engine = None
        if node_name != "server":
            cluster_id = node_name.split("_")[-1]
            node_id = node_name.split("_")[0]
            try:
                if preferences.public_private_experiment:
                    self.p2p_dataset = DataLoader().load_splitted_dataset(
                        f"../data/{dataset_name}/federated_data/cluster_{cluster_id}_node_{node_id}_public_train.pt",
                    )
                    self.server_dataset = DataLoader().load_splitted_dataset(
                        f"../data/{dataset_name}/federated_data/cluster_{cluster_id}_node_{node_id}_private_train.pt",
                    )
                    self.training_set = None
                else:
                    self.training_set = DataLoader().load_splitted_dataset(
                        f"../data/{dataset_name}/federated_data/cluster_{cluster_id}_node_{node_id}_train.pt",
                    )
                    self.p2p_dataset = None
                    self.server_dataset = None
                self.test_set = DataLoader().load_splitted_dataset(
                    f"../data/{dataset_name}/federated_data/cluster_{cluster_id}_node_{node_id}_test.pt",
                )
            except Exception as error:
                raise InvalidDatasetNameError from error
        self.trainloader = None
        self.testloader = None
        self.preferences = preferences
        self.diff_privacy_initialized = False
        gpus = preferences.gpu_config
        if len(node_name) == 11:
            if gpus:
                gpu_name = gpus[int(node_name[-1]) % len(gpus)]
            self.device = torch.device(
                gpu_name if torch.cuda.is_available() and gpus else "cpu",
            )
        print(f"---> USING {self.device}")

        self.net = None

    def init_model(self, net: nn.Module) -> None:
        """Initialize the Federated Model before the use of it.

        Args:
            net (nn.Module): model we want to use
        """
        if not self.trainloader and not self.testloader and self.node_name != "server":
            self.load_data()

        self.add_model(net)
        if self.net:
            params_to_update = []
            for _, param in self.net.named_parameters():
                if param.requires_grad is True:
                    params_to_update.append(param)

            # self.optimizer = torch.optim.Adam(
            # self.optimizer = torch.optim.SGD(
            self.optimizer = torch.optim.RMSprop(
                params_to_update,
                lr=self.preferences.hyperparameters_config.lr,
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
        self.net = model.to(self.device)

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
            batch_size = self.preferences.hyperparameters_config.batch_size
            if self.preferences.public_private_experiment:
                self.trainloader_private = torch.utils.data.DataLoader(
                    self.server_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                )
                self.trainloader_public = torch.utils.data.DataLoader(
                    self.p2p_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                )
            else:
                self.trainloader = torch.utils.data.DataLoader(
                    self.training_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                )

            self.testloader = torch.utils.data.DataLoader(
                self.test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,

            )

            if self.preferences and self.preferences.debug:
                if self.trainloader:
                    self.print_data_stats(self.trainloader)
                else:
                    # One call is enough because we are summing
                    # the size of the two datasets
                    self.print_data_stats(self.trainloader_public)
        else:
            raise NotYetInitializedPreferencesError

    def print_data_stats(self, trainloader: torch.utils.data.DataLoader) -> None:
        """Debug function used to print stats about the loaded datasets.

        Args:
            trainloader (torch.utils.data.DataLoader): training set
        """
        if self.training_set:
            num_examples = {
                "trainset": len(self.training_set),
                "testset": len(self.test_set),
            }
        else:
            num_examples = {
                "trainset": len(self.server_dataset) + len(self.p2p_dataset),
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
            # check if the avg_tensor and the models have the same keys
            # in particular, if the avg_tensor has the keys with name "_module."
            # and the model has the keys without "_module." we need to remove
            # the "_module." from the avg_tensor keys. Instead if the avg_tensor
            # does not have the "_module." and the model has it, we need to add it
            if (
                "_module." in list(avg_tensors.keys())[0]
                and "_module." not in list(self.net.state_dict().keys())[0]
            ):
                new_weights = OrderedDict()
                for key, value in avg_tensors.items():
                    new_weights[key.replace("_module.", "")] = value
                avg_tensors = new_weights
                # avg_tensors = {
                #     k.replace("_module.", ""): v for k, v in avg_tensors.items()
                # }
            if (
                "_module." not in list(avg_tensors.keys())[0]
                and "_module." in list(self.net.state_dict().keys())[0]
            ):
                new_weights = OrderedDict()
                for key, value in avg_tensors.items():
                    new_weights["_module." + key] = value
                # avg_tensors = {"_module." + k: v for k, v in avg_tensors.items()}
                avg_tensors = new_weights

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

    
    def train_with_differential_privacy(
        self,
        phase: Phase,
    ) -> tuple[float, float, float, float]:
        """Train the network using differential privacy and computes loss and accuracy.

        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized

        Returns
        -------
            Tuple[float, float, float, float]: Loss, accuracy, epsilon and noise multiplier
        """
        if self.net:
            max_physical_batch_size = (
                self.preferences.hyperparameters_config.max_phisical_batch_size
            )
            # Train the network
            criterion = nn.CrossEntropyLoss()
            running_loss = 0.0
            total_correct = 0
            total = 0
            losses = []
            self.net.train()

            if self.trainloader:
                training_data = self.trainloader
            else:
                training_data = (
                    self.trainloader_public
                    if phase == Phase.P2P

                    else self.trainloader_private
                )
                print(
                    f"Node {self.node_name} - Phase {phase}, using a dataset of size {len(training_data.dataset)}"
                )
            
            with BatchMemoryManager(
                data_loader=training_data,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=self.optimizer,
            ) as memory_safe_data_loader:
                for _, (data, target) in enumerate(memory_safe_data_loader, 0):
                    self.optimizer.zero_grad()

                    if isinstance(data, list):
                        data = data[0]
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.net(data)
                    loss = criterion(outputs, target)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    _, predicted = torch.max(outputs.data, 1)

                    total_correct += (predicted == target).float().sum()
                    total += target.size(0)
                    losses.append(loss.item())

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            train_loss = np.mean(losses)
            accuracy = total_correct / total
            logger.info(f"Training loss: {train_loss}, accuracy: {accuracy} - total_correct = {total_correct}, total = {total}")

            if self.node_name != "server":
                epsilon = self.privacy_engine.accountant.get_epsilon(
                    delta=self.preferences.hyperparameters_config.delta,
                )
                noise_multiplier = self.optimizer.noise_multiplier
                logger.info(
                    f"noise_multiplier: {noise_multiplier} - epsilon: {epsilon}"
                )

            return train_loss, accuracy, epsilon, noise_multiplier

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
                        outputs = self.net(data)
                        total += target.size(0)
                        test_loss = criterion(outputs, target).item()
                        losses.append(test_loss)
                        predicted = outputs.argmax(dim=1, keepdim=True)
                        correct += predicted.eq(target.view_as(predicted)).sum().item()
                        y_pred.extend(predicted)
                        y_true.extend(target)

                test_loss = np.mean(losses)
                accuracy = correct / total

                y_true = [item.item() for item in y_true]
                y_pred = [item.item() for item in y_pred]

                f1score = f1_score(y_true, y_pred, average="macro")
                precision = precision_score(y_true, y_pred, average="macro")
                recall = recall_score(y_true, y_pred, average="macro")

                cm = confusion_matrix(y_true, y_pred)
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                accuracy_per_class = cm.diagonal()

                print(f"---> TEST SET: accuracy: {accuracy}")

                return (
                    test_loss,
                    accuracy,
                    f1score,
                    precision,
                    recall,
                    accuracy_per_class,
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
            max_grad_norm = self.preferences.hyperparameters_config.max_grad_norm
            delta = self.preferences.hyperparameters_config.delta
            differential_privacy_p2p = self.preferences.p2p_config.differential_privacy

            if self.preferences.hyperparameters.get("epsilon_per_round", None):
                epochs = (
                    self.preferences.server_config.local_training_epochs
                    if phase == Phase.SERVER
                    else self.preferences.p2p_config.local_training_epochs
                )
            else:
                epochs = (
                    self.preferences.server_config.fl_rounds
                    + self.preferences.p2p_config.fl_rounds
                    if differential_privacy_p2p
                    else self.preferences.server_config.fl_rounds
                )

            if not epochs:
                raise Exception("Epochs is not initialized")

            if self.trainloader:
                training_data = self.trainloader
            else:
                training_data = (
                    self.trainloader_public
                    if phase == Phase.P2P
                    else self.trainloader_private
                )

            if self.privacy_engine:
                (
                    self.net,
                    self.optimizer,
                    train_loader,
                ) = self.privacy_engine.make_private_with_epsilon(
                    module=self.net,
                    optimizer=self.optimizer,
                    data_loader=training_data,
                    epochs=epochs,
                    target_epsilon=epsilon,
                    target_delta=delta,
                    max_grad_norm=max_grad_norm,
                )

                # Qui con questa ci riprendiamo il modello originale a cui avevamo
                # aggiunto tutto quello che riguarda la DP.
                # Se noi lo riprendiamo in questo modo possiamo fare in modo di aggiungere di nuovo
                # la DP ogni volta che mi serve fare la singola iterazione.
                logger.debug(
                    f"Model extracted from GradSample: {type(self.net._module)}"
                )
                if self.trainloader:
                    self.trainloader = train_loader
                elif phase == Phase.P2P:
                    self.trainloader_public = train_loader
                else:
                    self.trainloader_private = train_loader

        else:
            raise NotYetInitializedPreferencesError

    def init_privacy_with_noise(self, phase: Phase, noise_multiplier: float, clipping: float = None) -> None:
        """Initialize differential privacy using the noise parameter
        without the epsilon parameter.
        Noise multiplier: the more is higher the more is the noise
        Max grad: the more is higher the less private is training.

        Args:
            phase (Phase): phase of the training
            noise_multiplier: float: noise that we want to add
                every time we touch the data
        Raises:
            Exception: Preference is not initialized
        """
        if self.trainloader:
            training_data = self.trainloader
        else:
            training_data = (
                self.trainloader_public
                if phase == Phase.P2P
                else self.trainloader_private
            )

        if self.preferences:
            max_grad_norm = self.preferences.hyperparameters_config.max_grad_norm
            if self.privacy_engine:
                self.net.train()
                (
                    self.net,
                    self.optimizer,
                    train_loader,
                ) = self.privacy_engine.make_private(
                    module=self.net,
                    optimizer=self.optimizer,
                    data_loader=training_data,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm if not clipping else clipping,
                )
                if self.trainloader:
                    self.trainloader = train_loader
                elif phase == Phase.P2P:
                    self.trainloader_public = train_loader
                else:
                    self.trainloader_private = train_loader

        else:
            raise NotYetInitializedPreferencesError

    def init_differential_privacy(
        self,
        phase: Phase,
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
            self.privacy_engine = PrivacyEngine(accountant="rdp")
            noise_multiplier = None
            epsilon = None
            # We can initialize the private model using noise multiplier or
            # using the epsilon. In the experiments we used the noise multiplier
            # because using this we can have a better control of the privacy
            # budget spent during the training of the model
            if phase == Phase.P2P and self.preferences.p2p_config.differential_privacy:
                epsilon = self.preferences.p2p_config.epsilon
                noise_multiplier = self.preferences.p2p_config.noise_multiplier
            elif phase == Phase.SERVER and self.preferences.server_config.differential_privacy:
                epsilon = self.preferences.server_config.epsilon
                noise_multiplier = self.preferences.server_config.noise_multiplier

            self.net.train()
            if epsilon:
                # If we specify an epsilon, we have to use it during the
                # iterations and the noise will depend on the epsilon
                self.init_privacy_with_epsilon(phase=phase, epsilon=epsilon)
            elif noise_multiplier: 
                self.init_privacy_with_noise(
                    phase=phase,
                    noise_multiplier=noise_multiplier,
                )
            else:
                self.init_privacy_with_noise(
                    phase=phase,
                    noise_multiplier=0,
                    clipping=1000000000
                )

            if self.trainloader:
                return self.net, self.optimizer, self.trainloader
            if phase.SERVER:
                return self.net, self.optimizer, self.trainloader_private
            if phase.P2P:
                return self.net, self.optimizer, self.trainloader_public
        raise NotYetInitializedPreferencesError
