"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    def conv_block(in_channels, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ]

    layers = [
        nn.Conv2d(n_channels, 64, 3, padding=1, stride=1),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(3, 2),

        nn.Conv2d(64, 128, 3, padding=1, stride=1),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(3, 2),

        *conv_block(128, 256),
        *conv_block(256, 512),
        *conv_block(512, 512),

        nn.AvgPool2d(1, 1)
    ]

    self.blocks = nn.Sequential(*layers)

    self.linear = nn.Linear(512, n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = self.blocks(x)
    x = x.view(len(x), -1)
    out = self.linear(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
