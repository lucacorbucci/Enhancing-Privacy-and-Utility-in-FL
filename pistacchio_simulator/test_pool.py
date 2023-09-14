import gc
import os
import random
from collections import OrderedDict

import dill
import multiprocess
import numpy as np
import torch
import torch.nn.functional as F
from multiprocess import set_start_method
from opacus import PrivacyEngine
from torch import Tensor, nn


def load_accountant():
    accountant = None
    # If we already used this client we need to load the state regarding
    # the private model
    if os.path.exists(f"../examples/data/privacy_engine.pkl"):
        with open(f"../examples/data/privacy_engine.pkl", "rb") as file:
            accountant = dill.load(file)
    return accountant


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


class MnistNet(nn.Module):
    """Mnist network definition."""

    def __init__(self) -> None:
        """Initialization of the network."""
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns
        -------
            Tensor: Output data
        """
        out = input_data.view(-1, 28 * 28)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


data_root = "../mnist"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
BATCH_SIZE = 128


if __name__ == "__main__":
    set_start_method("spawn")

    MAX_GRAD_NORM = 100000.0
    NOISE = 1.0
    DELTA = 1e-5
    EPOCHS = 10

    LR = 0.01
    MAX_PHYSICAL_BATCH_SIZE = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # privacy_engine = PrivacyEngine(accountant="rdp")
    def get_parameters(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_params(model: torch.nn.ModuleList, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def accuracy(preds, labels):
        return (preds == labels).mean()

    def train(weights, device, epoch):
        data = torch.load(
            "../examples/data/mnist/federated_data/cluster_0_node_0_private_train.pt"
        )

        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        accountant = load_accountant()
        privacy_engine = PrivacyEngine(accountant="rdp")
        if accountant:
            privacy_engine.accountant = accountant

        def accuracy(preds, labels):
            return (preds == labels).mean()

        def get_parameters(model):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_params(model: torch.nn.ModuleList, params):
            """Set model weights from a list of NumPy ndarrays."""
            params_dict = zip(model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        DELTA = 1e-5
        model = MnistNet().to(device)
        set_params(model, weights)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0)

        private_model, private_optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=2.0,
        )
        private_model.train()
        criterion = nn.CrossEntropyLoss()
        losses = []
        top1_acc = []
        for _batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            private_optimizer.zero_grad()
            output = private_model(data)
            loss = criterion(output, target)
            loss.backward()
            private_optimizer.step()
            private_optimizer.zero_grad()
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            losses.append(loss.item())
            top1_acc.append(acc)

            epsilon = privacy_engine.accountant.get_epsilon(delta=DELTA)
            if (_batch_idx + 1) % 20 == 0:
                print(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} LEN {len(losses)}"
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f}"
                )

        with open(f"../examples/data/privacy_engine.pkl", "wb") as f:
            dill.dump(privacy_engine.accountant, f)
        set_params(model, get_parameters(private_model))
        gc.collect()
        return True, get_parameters(model)

    data_test = torch.load(
        "../examples/data/mnist/federated_data/server_validation_set.pt"
    )

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = MnistNet().to(device)

    with multiprocess.Pool(10) as pool:
        for epoch in range(10):
            weigts = get_parameters(model)
            results = [
                pool.apply_async(train, (weigts, device, epoch)),
            ]
            for result in results:
                (_, weights) = result.get()
                set_params(model, weights)

            test(model, "cuda:0", test_loader)
