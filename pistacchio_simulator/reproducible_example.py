#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""
import argparse
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocess import Process, Queue, set_start_method
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
delta = 1e-5

epochs = 10
lr = 0.1
sigma = 2.0
max_per_sample_grad_norm = 2.0

from collections import OrderedDict

from multiprocess import Process, Queue


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
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

    def forward(self, input_data):
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


device = torch.device("cuda:0")
data_root = "../mnist"
test_batch_size = 128
batch_size = 128
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    # seed = 42
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ["PYTHONHASHSEED"] = str(seed)

    set_start_method("spawn")

    def train_process(queue, weights, device, privacy_engine, epoch):
        print(f"Training epoch: {epoch}")

        def get_parameters(model):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_params(model: torch.nn.ModuleList, params):
            """Set model weights from a list of NumPy ndarrays."""
            params_dict = zip(model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                    ]
                ),
            ),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
        )
        # data = torch.load(
        #     "../examples/data/mnist/federated_data/cluster_0_node_0_private_train.pt"
        # )

        # train_loader = torch.utils.data.DataLoader(
        #     data,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=0,
        # )

        model = MnistNet().to("cuda:0")

        set_params(model, weights)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)

        # privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=sigma,
            max_grad_norm=max_per_sample_grad_norm,
        )

        model.train()
        criterion = nn.CrossEntropyLoss()
        losses = []
        correct = 0
        print(f"Starting Train: {epoch}")
        for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {delta})"
            f"(Accuracy: {100.0 * correct / len(train_loader.dataset)})"
        )
        queue.put((get_parameters(model), privacy_engine))

    model = MnistNet().to(device)

    weights = get_parameters(model)
    privacy_engine = PrivacyEngine(accountant="rdp")

    q = Queue()
    p = Process(target=train_process, args=[q, weights, device, privacy_engine, 0])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 1])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 2])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 3])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 4])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 5])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 6])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 7])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 8])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 9])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    weigts = get_parameters(model)

    q = Queue()
    p = Process(target=train_process, args=[q, weigts, device, privacy_engine, 10])
    p.start()
    weights, privacy_engine = q.get()
    p.join()

    set_params(model, weights)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test(model, "cuda:0", test_loader)
