import torch
from torch import nn as nn
from torchvision import models as models
from enum import Enum


class ResNetModelType(Enum):
    # refer to this post: https://stackoverflow.com/questions/31907060/python-3-enums-with-function-values/40774465
    # although, we could have directly defined the enum values to be function handles for the specific resnet,
    # that'd turn the individual enum instances into function handles, hence, turning this into a class
    # that includes a bunch of function handles as opposed to an enum in the traditional sense. So for eg.,
    # something like RESNET18.value would no longer work then since RESNET18 would be a function handle and not an enum
    RESNET18 = "RESNET18"
    RESNET34 = "RESNET34"
    RESNET50 = "RESNET50"
    RESNET101 = "RESNET101"
    RESNET152 = "RESNET152"


def get_resnet_model(resnet_model_type, pretrained):
    if resnet_model_type == ResNetModelType.RESNET18:
        return models.resnet18(pretrained=pretrained)
    elif resnet_model_type == ResNetModelType.RESNET34:
        return models.resnet34(pretrained=pretrained)
    elif resnet_model_type == ResNetModelType.RESNET50:
        return models.resnet50(pretrained=pretrained)
    elif resnet_model_type == ResNetModelType.RESNET101:
        return models.resnet101(pretrained=pretrained)
    elif resnet_model_type == ResNetModelType.RESNET152:
        return models.resnet152(pretrained=pretrained)

    raise ValueError(f"unknown resnet model type: {resnet_model_type.value}")


class ResNetWithLastLayerModified(nn.Module):
    def __init__(self, modified_last_layer_output_size: int, resnet_model_type: ResNetModelType,
                 pretrained: bool = True, freeze_pretrained_layers: bool = True):
        super(ResNetWithLastLayerModified, self).__init__()

        """Load the pretrained ResNet-X and replace top fc layer."""
        resnet = get_resnet_model(resnet_model_type=resnet_model_type, pretrained=pretrained)

        # delete the last fc layer and flatten the module list.
        self.resnet_with_last_layer_removed = nn.Sequential(*list(resnet.children())[:-1])
        self.linear_layer = nn.Linear(resnet.fc.in_features, modified_last_layer_output_size)
        self.batch_norm = nn.BatchNorm1d(modified_last_layer_output_size, momentum=0.01)
        self.pretrained = pretrained
        self.freeze_pretrained_layers = freeze_pretrained_layers

    def forward(self, input_images):
        """Extract feature vectors from input images."""

        if self.pretrained and self.freeze_pretrained_layers:
            with torch.no_grad():
                resnet_features = self.resnet_with_last_layer_removed(input_images)
        else:
            resnet_features = self.resnet_with_last_layer_removed(input_images)

        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        return self.batch_norm(self.linear_layer(resnet_features))
