import gc
import os
import sys
from pathlib import Path
from typing import Mapping, TypeVar

import dill
import numpy as np
import torch
from loguru import logger
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import Tensor, nn

from pistacchio_simulator.Utils.data_loader import DataLoader
from pistacchio_simulator.Utils.learning import Learning
from pistacchio_simulator.Utils.performances import Performances
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.utils import Utils

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {level} | {message}",
)

TDestination = TypeVar("TDestination ", bound=Mapping[str, Tensor])
node_name_lenght = 16


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, epoch, device, privacy_engine):
    model.train()

    criterion = nn.CrossEntropyLoss()

    DELTA = 1e-5
    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=128,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        for _, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()

            images = images.to(device)
            target = target.long()
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

        epsilon = privacy_engine.get_epsilon(DELTA)
        return np.mean(losses), np.mean(top1_acc), epsilon


class FederatedNode:
    """FederatedNode is the component that can be used for the
    classic Federated Learning.
    It trains the model locally and then sends the weights to the server.
    """

    def __init__(
        self,
        node_id: str,
        cluster_id: str,
        node_name: str,
        preferences: Preferences,
        phase: Phase,
        weights,
    ) -> None:
        """Init the Federated Node.

        Args:
            node_id (str): id of the node
            preferences (Preferences): preferences object of the node that contains
                all the preferences for this node
            # logging_queue (CommunicationChannel): queue that is used to send back the
            #     performances of the node to the main thread.
        """

        self.node_id = node_id
        self.cluster_id = cluster_id
        self.preferences = preferences
        self.phase = phase
        self.mode = "federated"
        self.mixed = False
        self.message_counter = 0
        self.node_name = node_name
        self.node_folder_path = f"{self.preferences.data_split_config.store_path}/nodes_data/{self.node_name}/"
        self.load_data()
        gpus = preferences.gpu_config
        if gpus:
            gpu_name = gpus[int(self.node_id) % len(gpus)]
            self.device = torch.device(
                gpu_name if torch.cuda.is_available() and gpus else "cpu",
            )
        else:
            self.device = "cpu"
        self.dp = False
        self.weights = weights
        accountant = self.load_accountant()
        self.privacy_engine = PrivacyEngine(accountant="rdp")
        if accountant:
            logger.info(f"Loading the accountant on {self.node_name}")
            self.privacy_engine.accountant = accountant

        logger.info(f"Node {self.node_name} is using device {self.device}")

    def load_data(self):
        self.validation_set = None
        self.validation_loader = None
        if self.preferences.public_private_experiment and self.phase == Phase.P2P:
            if self.preferences.dataset_p2p:
                # When I have the P2P phase and during that phase I don't use differential privacy
                # then I have to use the public dataset. Instead, when I use DP, I'll use the
                # private training dataset even during the P2P phase
                self.train_set = DataLoader().load_splitted_dataset(
                    f"{self.preferences.data_split_config.store_path}/{self.node_name}_{self.preferences.dataset_p2p}_train.pt",
                )
            else:
                self.train_set = DataLoader().load_splitted_dataset(
                    f"{self.preferences.data_split_config.store_path}/{self.node_name}_public_train.pt",
                )
        elif self.preferences.public_private_experiment and self.phase == Phase.SERVER:
            if self.preferences.dataset_server:
                self.train_set = DataLoader().load_splitted_dataset(
                    f"{self.preferences.data_split_config.store_path}/{self.node_name}_{self.preferences.dataset_server}_train.pt",
                )
            else:
                self.train_set = DataLoader().load_splitted_dataset(
                    f"{self.preferences.data_split_config.store_path}/{self.node_name}_private_train.pt",
                )
        else:
            self.train_set = DataLoader().load_splitted_dataset(
                f"{self.preferences.data_split_config.store_path}/{self.node_name}_train.pt",
            )
        from collections import Counter

        print(
            f"NODE NAME {self.node_name} has {Counter([target.item() for target in self.train_set.targets])}"
        )
        # self.test_set = DataLoader().load_splitted_dataset(
        #     f"{self.preferences.data_split_config.store_path}/{self.node_name}_test.pt",
        # )
        if self.phase == Phase.P2P:
            batch_size = self.preferences.p2p_config.batch_size
        else:
            batch_size = self.preferences.server_config.batch_size
        print(f"Batch size {batch_size}")
        
        if self.preferences.data_split_config.validation_size > 0:
            if self.preferences.public_private_experiment and self.phase == Phase.P2P:
                if self.preferences.dataset_p2p:
                    # self.train_set = DataLoader().load_splitted_dataset(
                    #     f"{self.preferences.data_split_config.store_path}/{self.node_name}_{self.preferences.dataset_p2p}_train.pt",
                    # )
                    self.validation_set = DataLoader().load_splitted_dataset(
                        f"{self.preferences.data_split_config.store_path}/{self.node_name}_{self.preferences.dataset_p2p}_validation.pt",
                    )
            elif self.preferences.public_private_experiment and self.phase == Phase.SERVER:
                if self.preferences.dataset_server:
                    # self.train_set = DataLoader().load_splitted_dataset(
                    #     f"{self.preferences.data_split_config.store_path}/{self.node_name}_{self.preferences.dataset_server}_train.pt",
                    # )
                    self.validation_set = DataLoader().load_splitted_dataset(
                        f"{self.preferences.data_split_config.store_path}/{self.node_name}_{self.preferences.dataset_server}_validation.pt",
                    )
           
            
            self.validation_loader = torch.utils.data.DataLoader(
                self.validation_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        print("train ", len(self.train_loader.dataset))
        print("test ", len(self.validation_loader.dataset))


     

    def load_accountant(self):
        accountant = None
        # If we already used this client we need to load the state regarding
        # the private model
        if os.path.exists(f"{self.node_folder_path}privacy_engine.pkl"):
            with open(f"{self.node_folder_path}privacy_engine.pkl", "rb") as file:
                accountant = dill.load(file)
                logger.info(
                    f"Node {self.node_name} loaded privacy engine during {self.phase} phase"
                )

        return accountant

    def compute_performances(
        self,
        loss_list: list,
        accuracy_list: list,
        phase: str,
        message_counter: int,
        epsilon_list: list | None,
    ) -> dict:
        epochs = range(
            1,
            self.preferences.server_config.fl_rounds + 1,
        )

        performances = {}
        performances[phase] = Performances(
            node_name=self.node_id,
            epochs=list(epochs),
            loss_list=loss_list,
            accuracy_list=accuracy_list,
            loss=None,
            accuracy=None,
            message_counter=message_counter,
            epsilon_list=epsilon_list,
        )

        return performances

    def train_local_model(
        self,
        phase: Phase,
        # results: dict | None = None,
    ) -> tuple[list[float], list[float], list[float]]:
        model = Utils.get_model(preferences=self.preferences)
        model.to(self.device)
        Utils.set_params(model, self.weights)

        optimizer = Utils.get_optimizer(
            preferences=self.preferences, model=model, phase=phase
        )

        # model = model.to(self.device)

        (
            private_model,
            private_optimizer,
            private_train_loader,
        ) = self.init_differential_privacy(
            phase=phase,
            optimizer=optimizer,
            model=model,
        )
        private_model.to(self.device)
        print("TO DEVICE", self.device)

        local_epochs = (
            self.preferences.server_config.local_training_epochs
            if phase == Phase.SERVER
            else self.preferences.p2p_config.local_training_epochs
        )

        for local_epoch in range(local_epochs):
            train_loss, accuracy, epsilon = train(
                model=private_model,
                train_loader=private_train_loader,
                optimizer=private_optimizer,
                epoch=local_epoch,
                device=self.device,
                privacy_engine=self.privacy_engine,
            )
            metrics = {"loss": train_loss, "accuracy": accuracy, "epsilon": epsilon}

        # Evaluate the model on validation set
        if self.validation_loader is not None:
            (
                validation_loss,
                validation_accuracy,
                _,
                _,
                _,
            ) = Learning.evaluate_model(
                model=private_model,
                test_loader=self.validation_loader,
                device=self.device,
            )

        # Evaluate the model on test set
        # if self.test_set:
        #     (
        #         test_loss,
        #         test_accuracy,
        #         _,
        #         _,
        #         _,
        #     ) = Learning.evaluate_model(
        #         model=private_model,
        #         test_loader=self.test_loader,
        #         device=self.device,
        #     )

        metrics["validation_loss"] = torch.tensor(validation_loss)
        # metrics["test_loss"] = torch.tensor(test_loss)
        metrics["validation_accuracy"] = torch.tensor(validation_accuracy)
        # metrics["test_accuracy"] = torch.tensor(test_accuracy)

        # We need to store the state of the privacy engine and all the
        # details about the private training
        directory_path = Path(self.node_folder_path)
        directory_path.mkdir(parents=True, exist_ok=True)

        if self.noise_multiplier != 0:
            with open(f"{self.node_folder_path}privacy_engine.pkl", "wb") as f:
                dill.dump(self.privacy_engine.accountant, f)

        Utils.set_params(model, Utils.get_parameters(private_model))
        gc.collect()

        return (
            Utils.get_parameters(model),
            metrics,
            len(self.train_loader.dataset),
        )

    def init_differential_privacy(self, phase: Phase, optimizer, model):
        logger.info(f"Initializing differential privacy for Phase {phase}")

        epsilon = None
        noise_multiplier = 0
        clipping = (
            self.preferences.hyperparameters_config.max_grad_norm
            if self.preferences.hyperparameters_config.max_grad_norm
            else 100000000
        )
        if (
            phase == Phase.P2P
            and self.preferences.p2p_config
            and self.preferences.p2p_config.differential_privacy
        ):
            epsilon = self.preferences.p2p_config.epsilon
            noise_multiplier = self.preferences.p2p_config.noise_multiplier
        elif (
            phase == Phase.SERVER
            and self.preferences.server_config
            and self.preferences.server_config.differential_privacy
        ):
            epsilon = self.preferences.server_config.epsilon
            noise_multiplier = self.preferences.server_config.noise_multiplier

        if epsilon:
            logger.info(f"Initializing differential privacy with epsilon {epsilon}")
            # (
            #     self.net,
            #     optimizer,
            #     train_loader,
            # ) = self.privacy_engine.make_private_with_epsilon(
            #     module=self.net,
            #     optimizer=self.optimizer,
            #     data_loader=self.train_loader,
            #     epochs=epochs,
            #     target_epsilon=epsilon,
            #     target_delta=delta,
            #     max_grad_norm=clipping,
            # )
        else:
            self.noise_multiplier = noise_multiplier
            if self.noise_multiplier == 0:
                clipping = 100000000
            logger.info(
                f"Initializing differential privacy with noise {noise_multiplier} and clipping {clipping}"
            )

            (
                private_model,
                private_optimizer,
                private_train_loader,
            ) = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=clipping,
            )
        return private_model, private_optimizer, private_train_loader
