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
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from multiprocess import Process, Queue, set_start_method
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


class CelebaNet(nn.Module):
    """This class defines the CelebaNet."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        dropout_rate: float = 0,
    ) -> None:
        """Initializes the CelebaNet network.

        Args:
        ----
            in_channels (int, optional): Number of input channels . Defaults to 3.
            num_classes (int, optional): Number of classes . Defaults to 2.
            dropout_rate (float, optional): _description_. Defaults to 0.2.
        """
        super().__init__()
        self.cnn1 = nn.Conv2d(
            in_channels,
            8,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
        )
        self.cnn2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.cnn3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(2048, num_classes)
        self.gn_relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns
        -------
            Tensor: Output data
        """
        out = self.gn_relu(self.cnn1(input_data))
        out = self.gn_relu(self.cnn2(out))
        out = self.gn_relu(self.cnn3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out


class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms = None,
        debug: bool = True,
    ) -> None:
        """Initialization of the dataset.

        Args:
        ----
            csv_path (str): path of the csv file with all the information
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
        """
        dataframe = pd.read_csv(csv_path)

        self.targets = dataframe["Target"].tolist()
        self.classes = dataframe["Target"].tolist()

        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path
        self.debug = debug
        if not self.debug:
            self.images = [
                Image.open(os.path.join(self.image_path, sample)).convert(
                    "RGB",
                )
                for sample in self.samples
            ]

    def __getitem__(self, index: int):
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """
        if self.debug:
            img = Image.open(
                os.path.join(self.image_path, self.samples[index]),
            ).convert(
                "RGB",
            )
        else:
            img = self.images[index]

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.targets[index],
        )

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


device = torch.device("cuda:0")
data_root = "../data/celeba"
test_batch_size = 512
batch_size = 512
seed = 42
delta = 1e-5
epochs = 10
lr = 0.01
sigma = 1.0
max_per_sample_grad_norm = 5.0

import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, epoch, device, privacy_engine):
    model.train()
    criterion = nn.CrossEntropyLoss()

    DELTA = 1e-5
    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader, max_physical_batch_size=128, optimizer=optimizer
    ) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )


if __name__ == "__main__":
    set_start_method("spawn")

    def train_process(queue, weights, device, privacy_engine, epoch):
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )
        train_dataset = CelebaDataset(
            csv_path="../data/celeba/train_smiling.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
            debug=True,
        )

        # train_dataset = torch.load("../examples/celeba/data/celeba/federated_data/cluster_0_node_0_public_train.pt")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )

        model = CelebaNet().to("cuda:0")

        set_params(model, weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        (
            private_model,
            private_optimizer,
            private_train_loader,
        ) = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=sigma,
            max_grad_norm=max_per_sample_grad_norm,
        )

        private_model.train()
        for i in range(0, 10):
            train(
                model=private_model,
                train_loader=private_train_loader,
                optimizer=private_optimizer,
                epoch=i,  # epoch,
                device=device,
                privacy_engine=privacy_engine,
            )

        queue.put((get_parameters(model), privacy_engine))

    model = CelebaNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    privacy_engine = PrivacyEngine(accountant="rdp")

    weights = get_parameters(model)

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

    # set_params(model, weights)

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         data_root,
    #         train=False,
    #         transform=transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    #             ]
    #         ),
    #     ),
    #     batch_size=test_batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     pin_memory=True,
    # )

    # test(model, "cuda:0", test_loader)
