import numpy as np
import torch
from loguru import logger
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pistacchio_simulator.Exceptions.errors import (
    NotYetInitializedFederatedLearningError,
)
from pistacchio_simulator.Utils.phases import Phase
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn


class Learning:
    def train(
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
        if self.federated_model and self.federated_model.net:
            model = self.federated_model.net
            max_physical_batch_size = (
                self.preferences.hyperparameters_config.max_phisical_batch_size
            )
            # Train the network
            criterion = nn.CrossEntropyLoss()
            running_loss = 0.0
            total_correct = 0
            total = 0
            losses = []
            model.train()

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
                    outputs = model(data)
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
            logger.info(
                f"Training loss: {train_loss}, accuracy: {accuracy} - total_correct = {total_correct}, total = {total}"
            )

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
            if self.federated_model.net:
                self.federated_model.net.eval()
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
                        outputs = self.federated_model.net(data)
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
