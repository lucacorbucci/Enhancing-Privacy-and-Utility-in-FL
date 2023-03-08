from torch import nn
from torchvision import models


class FairFace:
    @staticmethod
    def get_model_to_fine_tune() -> nn.Module:
        """This function is used to get the model to fine tune.
        In this case we use a pre trained EfficientNet B0 pre trained
        on image net.

        Returns
        -------
            nn.Module: the model to fine tune
        """
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")

        for name, param in model.named_parameters(recurse=True):
            if not name.startswith("classifier"):
                param.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=2)

        return model
