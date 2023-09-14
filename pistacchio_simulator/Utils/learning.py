import numpy as np
import torch
from loguru import logger
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pistacchio_simulator.Exceptions.errors import (
    NotYetInitializedFederatedLearningError,
)
from pistacchio_simulator.Utils.phases import Phase
from pistacchio_simulator.Utils.preferences import Preferences
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn


class Learning:
    @staticmethod
    def train(
        model,
        preferences: Preferences,
        phase: Phase,
        node_name: str,
        optimizer: torch.optim.Optimizer,
        device: str,
        privacy_engine: torch.optim.Optimizer = None,
        train_loader: torch.utils.data.DataLoader = None,
    ) -> tuple[float, float, float, float]:
        """Train the network using differential privacy and computes loss and accuracy.

        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized

        Returns
        -------
            Tuple[float, float, float, float]: Loss, accuracy, epsilon and noise multiplier
        """
        if model:
            max_physical_batch_size = (
                preferences.hyperparameters_config.max_phisical_batch_size
            )
            # Train the network
            criterion = nn.CrossEntropyLoss()
            total_correct = 0
            total = 0
            losses = []
            model.train()

            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=optimizer,
            ) as memory_safe_data_loader:
                for _, (data, target) in enumerate(memory_safe_data_loader, 0):
                    optimizer.zero_grad()

                    if isinstance(data, list):
                        data = data[0]
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
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

            if node_name != "server":
                epsilon = privacy_engine.accountant.get_epsilon(
                    delta=preferences.hyperparameters_config.delta,
                )
                noise_multiplier = optimizer.noise_multiplier
                logger.info(
                    f"noise_multiplier: {noise_multiplier} - epsilon: {epsilon}"
                )

            return train_loss, accuracy, epsilon, noise_multiplier

        raise NotYetInitializedFederatedLearningError

    @staticmethod
    def evaluate_model(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str,
    ) -> tuple[float, float, float, float, float, list]:
        """Validate the network on the entire test set.

        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized

        Returns
        -------
            Tuple[float, float]: loss and accuracy on the test set.
        """
        with torch.no_grad():
            if model:
                model.eval()
                criterion = nn.CrossEntropyLoss()
                test_loss = 0
                correct = 0
                total = 0
                y_pred = []
                y_true = []
                losses = []
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        outputs = model(data)
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

                return (
                    test_loss,
                    accuracy,
                    f1score,
                    precision,
                    recall,
                    accuracy_per_class,
                )

            raise NotYetInitializedFederatedLearningError
