from torch import nn
from pistacchio_simulator.Components.Orchestrator.orchestrator import Orchestrator
from pistacchio_simulator.Exceptions.errors import InvalidDatasetErrorNameError
from pistacchio_simulator.Models.celeba import CelebaGenderNet, CelebaNet
from pistacchio_simulator.Models.fashion_mnist import FashionMnistNet
from pistacchio_simulator.Models.mnist import MnistNet

def get_model(model_name: str) -> nn.Module:
    """This function is used to get the model.
    Args:
        model_name (str): string denoted the model name
    Returns
        nn.Module: the model
        """
    if model_name == "mnist":
        model = MnistNet()
    #elif preferences.dataset_name == "cifar10":
        #model = Experiment.get_model_to_fine_tune()
        #preferences.fine_tuning = True
    elif model_name == "celeba":
        model = CelebaNet()
    elif model_name == "celeba_gender":
        model = CelebaGenderNet()
    elif model_name == "fashion_mnist":
        model = FashionMnistNet()
    #elif preferences.dataset_name == "imaginette":
        #model = Experiment.get_model_to_fine_tune()
        #preferences.fine_tuning = True
    else:
        raise InvalidDatasetErrorNameError("Invalid dataset name")
    return model