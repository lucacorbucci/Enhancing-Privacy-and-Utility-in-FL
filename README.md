[![Build and check code style](https://github.com/lucacorbucci/Hierarchical_FL/actions/workflows/main.yml/badge.svg)](https://github.com/lucacorbucci/Hierarchical_FL/actions/workflows/main.yml) [![Pytest](https://github.com/lucacorbucci/Hierarchical_FL/actions/workflows/pytest.yml/badge.svg)](https://github.com/lucacorbucci/Hierarchical_FL/actions/workflows/pytest.yml)


## How to use this repository

- Clone this repository
```
git clone https://github.com/lucacorbucci/Pistacchio.git
```

- In this project I used Poetry to manage the dependencies. If you don't have it installed, you can install it with:
```
curl -sSL https://install.python-poetry.org | python3 -
```

- Install the dependencies with:
```
poetry install
```

# Overview

## How do we split the data

The clas PrepareDataset defines all the methods necessary to use the datasets and to split them into smaller group of data that can be used by all the nodes.
The class defines some methods to download the datasets, and a method to split the data in equal parts and write them in a file.

Data are divided keeping the data distribution of the original dataset.
Example: if in the original dataset there are 100 images, and the number of nodes is 5, then each node will have 20 images. If the 80% of the images are classified with class A and the 20% with class B then each node will have 80% of images classified as A and 20% of images classified as B.

# Components

We defined several components that can be used in our experiments.
...

## P2PNode

P2PNode is the simplest node we can have in a cluster of nodes. This node does the following things for each epoch:

- Trains the model locally for one epoch
- Sends the weights to the other nodes of the cluster
- Receives the weights from the other nodes of the cluster
- Compute the average of all the weights
- Updates the weights of the model with the average and goes to the next epoch

## SemiP2PNode

SemiP2PNode is a node that does two different things. First of all it trains the model with the other nodes of the cluster. Then it start a training process with the server.
This is a summary of what this node does:

- Trains the model locally for one epoch
- Sends the weights to the other nodes of the cluster
- Receives the weights from the other nodes of the cluster
- Compute the average of all the weights
- Updates the weights of the model with the average and goes to the next epoch
- When the inter-cluster training is finished, it starts a training process with the server.
- For each epochs, It trains the model locally
- It sends the weights to the server
- It waits the updated weights from the server and then it starts again the local training

## FederatedNode

This is a classic Federated Learning node, it does the following things:

- For each epochs, It trains the model locally
- It sends the weights to the server
- It waits the updated weights from the server and then it starts again the local training

## P2PCluster

We organize the nodes in clusters. In each cluster we can have several nodes that aim to train the same model.
