################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device

        self.W_hx = nn.Parameter(0.01 * torch.randn(input_dim, num_hidden))
        self.W_hh = nn.Parameter(0.01 * torch.randn(num_hidden, num_hidden))
        self.b_h = nn.Parameter(0.01 * torch.randn(num_hidden))

        self.tanh = nn.Tanh()

        self.W_ph = nn.Parameter(0.01 * torch.randn(num_hidden, num_classes))
        self.b_p = nn.Parameter(0.01 * torch.randn(num_classes))

    def forward(self, x):
        # Implementation here ...
        h = torch.zeros(self.batch_size, len(self.W_hh)).to(self.device)

        for step in range(self.seq_length):
            h = self.tanh(
                x[:, step, :] @ self.W_hx   # Input-to-hidden
                + h @ self.W_hh             # Hidden-to-hidden
                + self.b_h                  # Hidden bias
            )
        
        p = h @ self.W_ph + self.b_p        # Hidden-to-output

        return p
