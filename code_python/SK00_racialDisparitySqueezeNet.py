import torch
import torch.nn as nn

from torchvision import models
from shutup import please; please()

'''
@author Sakin Kirti
@date 6/20/22

class to define the SqueezeNet architecture used for the racial disparity project
'''

class RacialDisparity_SqueezeNet(nn.Module):
    '''
    Note that this uses a pre-built squeezenet1_0 from torchvision.models with...
    - a classifier addition
    - this is meant solely for model training
    '''

    def __init__(self, checkpoint_path) -> None:
        super(RacialDisparity_SqueezeNet).__init__()

        # define model architecture
        self.model = models.squeezenet1_0()
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.55),
            nn.Conv2d(512, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # load the model weights
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class ActivationMap_SqueezeNet(nn.Module):
    '''
    Note that this uses a pre-built squeezenet1_0 from torchvision.models with...
    - a classifier addition
    - some additional methods
    - this is meant solely for generating the activation maps
    '''

    # initialize
    def __init__(self, checkpoint_path=None):
        super(ActivationMap_SqueezeNet, self).__init__()

        # define the model architecture
        self.model = models.squeezenet1_0()
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # load the model weights
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        # define some other features
        self.classifier = self.model.classifier
        self.features_conv = self.model.features
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.gradients = None

    # forward pass
    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    # set the gradients to be the grad that is passed in
    def activations_hook(self, grad):
        self.gradients = grad

    # get the activations gradient
    def get_activations_gradient(self):
        return self.gradients

    # get the features
    def get_activations(self, x):
        return self.features_conv(x)