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
from pistacchio_simulator.Exceptions.errors import (
    InvalidDatasetNameError,
    NotYetInitializedFederatedLearningError,
    NotYetInitializedPreferencesError,
)
from pistacchio_simulator.Utils.data_loader import DataLoader
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from torch import nn, optim

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
        model: nn.Module,
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

        self.preferences = preferences
        gpus = preferences.gpu_config
        if len(node_name) == 11:
            if gpus:
                gpu_name = gpus[int(node_name[-1]) % len(gpus)]
            self.device = torch.device(
                gpu_name if torch.cuda.is_available() and gpus else "cpu",
            )
        print(f"---> USING {self.device}")

        self.net = model

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

    # def init_differential_privacy(
    #     self,
    #     phase: Phase,
    # ) -> tuple[nn.Module, optim.Optimizer, torch.utils.data.DataLoader]:
    #     """Initialize the differential privacy.

    #     Args:
    #         phase (str): phase of the training

    #     Returns
    #     -------
    #         _type_:
    #     """
    #     if self.preferences:
    #         self.diff_privacy_initialized = True
    #         self.privacy_engine = PrivacyEngine(accountant="rdp")
    #         noise_multiplier = None
    #         epsilon = None
    #         # We can initialize the private model using noise multiplier or
    #         # using the epsilon. In the experiments we used the noise multiplier
    #         # because using this we can have a better control of the privacy
    #         # budget spent during the training of the model
    #         if phase == Phase.P2P and self.preferences.p2p_config.differential_privacy:
    #             epsilon = self.preferences.p2p_config.epsilon
    #             noise_multiplier = self.preferences.p2p_config.noise_multiplier
    #         elif (
    #             phase == Phase.SERVER
    #             and self.preferences.server_config.differential_privacy
    #         ):
    #             epsilon = self.preferences.server_config.epsilon
    #             noise_multiplier = self.preferences.server_config.noise_multiplier

    #         self.net.train()
    #         if epsilon:
    #             # If we specify an epsilon, we have to use it during the
    #             # iterations and the noise will depend on the epsilon
    #             self.init_privacy_with_epsilon(phase=phase, epsilon=epsilon)
    #         elif noise_multiplier:
    #             self.init_privacy_with_noise(
    #                 phase=phase,
    #                 noise_multiplier=noise_multiplier,
    #             )
    #         else:
    #             self.init_privacy_with_noise(
    #                 phase=phase, noise_multiplier=0, clipping=1000000000
    #             )

    #         if self.trainloader:
    #             return self.net, self.optimizer, self.trainloader
    #         if phase.SERVER:
    #             return (
    #                 self.net,
    #                 self.optimizer,
    #                 self.trainloader_private,
    #             )
    #         if phase.P2P:
    #             return self.net, self.optimizer, self.trainloader_public
    #     raise NotYetInitializedPreferencesError

    def init_privacy_with_epsilon(
        self,
        phase: Phase,
        epsilon: float,
        train_loader,
        privacy_engine,
        optimizer,
        epochs,
        delta,
        clipping: float = None,
    ) -> None:
        """Initialize differential privacy using the epsilon parameter.

        Args:
            phase (Phase): phase of the training
            EPSILON (float): epsilon parameter for differential privacy

        Raises
        ------
            Exception: Preference is not initialized
        """
        # if self.preferences:
        #     max_grad_norm = self.preferences.hyperparameters_config.max_grad_norm
        #     delta = self.preferences.hyperparameters_config.delta
        #     differential_privacy_p2p = self.preferences.p2p_config.differential_privacy

        #     if self.preferences.hyperparameters.get("epsilon_per_round", None):
        #         epochs = (
        #             self.preferences.server_config.local_training_epochs
        #             if phase == Phase.SERVER
        #             else self.preferences.p2p_config.local_training_epochs
        #         )
        #     else:
        #         epochs = (
        #             self.preferences.server_config.fl_rounds
        #             + self.preferences.p2p_config.fl_rounds
        #             if differential_privacy_p2p
        #             else self.preferences.server_config.fl_rounds
        #         )

        #     if not epochs:
        #         raise Exception("Epochs is not initialized")

        #     if self.trainloader:
        #         training_data = self.trainloader
        #     else:
        #         training_data = (
        #             self.trainloader_public
        #             if phase == Phase.P2P
        #             else self.trainloader_private
        #         )

        if privacy_engine:
            (
                self.net,
                optimizer,
                train_loader,
            ) = privacy_engine.make_private_with_epsilon(
                module=self.net,
                optimizer=self.optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=clipping,
            )

            # Qui con questa ci riprendiamo il modello originale a cui avevamo
            # aggiunto tutto quello che riguarda la DP.
            # Se noi lo riprendiamo in questo modo possiamo fare in modo di aggiungere di nuovo
            # la DP ogni volta che mi serve fare la singola iterazione.
            logger.debug(f"Model extracted from GradSample: {type(self.net._module)}")
            return optimizer, train_loader, privacy_engine

            # if self.trainloader:
            #     self.trainloader = train_loader
            # elif phase == Phase.P2P:
            #     self.trainloader_public = train_loader
            # else:
            #     self.trainloader_private = train_loader

        else:
            raise NotYetInitializedPreferencesError

    def init_privacy_with_noise(
        self,
        phase: Phase,
        noise_multiplier: float,
        train_loader,
        privacy_engine,
        optimizer,
        clipping: float = None,
    ) -> None:
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
        # if self.trainloader:
        #     training_data = self.trainloader
        # else:
        #     training_data = (
        #         self.trainloader_public
        #         if phase == Phase.P2P
        #         else self.trainloader_private
        #     )

        # if self.preferences:
        #     max_grad_norm = self.preferences.hyperparameters_config.max_grad_norm
        if privacy_engine:
            self.net.train()
            (
                self.net,
                optimizer,
                train_loader,
            ) = privacy_engine.make_private(
                module=self.net,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=clipping,
            )
            # if self.trainloader:
            #     self.trainloader = train_loader
            # elif phase == Phase.P2P:
            #     self.trainloader_public = train_loader
            # else:
            #     self.trainloader_private = train_loader
            return optimizer, train_loader, privacy_engine

        else:
            raise NotYetInitializedPreferencesError
