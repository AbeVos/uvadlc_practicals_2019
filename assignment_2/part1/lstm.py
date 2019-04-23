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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = device

        self.W_gx = nn.Parameter(0.01 * torch.randn(input_dim, num_hidden))
        self.W_gh = nn.Parameter(0.01 * torch.randn(num_hidden, num_hidden))
        self.b_g = nn.Parameter(0.01 * torch.randn(num_hidden))

        self.W_ix = nn.Parameter(0.01 * torch.randn(input_dim, num_hidden))
        self.W_ih = nn.Parameter(0.01 * torch.randn(num_hidden, num_hidden))
        self.b_i = nn.Parameter(0.01 * torch.randn(num_hidden))

        self.W_fx = nn.Parameter(0.01 * torch.randn(input_dim, num_hidden))
        self.W_fh = nn.Parameter(0.01 * torch.randn(num_hidden, num_hidden))
        self.b_f = nn.Parameter(0.01 * torch.randn(num_hidden))

        self.W_ox = nn.Parameter(0.01 * torch.randn(input_dim, num_hidden))
        self.W_oh = nn.Parameter(0.01 * torch.randn(num_hidden, num_hidden))
        self.b_o = nn.Parameter(0.01 * torch.randn(num_hidden))

        self.W_ph = nn.Parameter(0.01 * torch.randn(num_hidden, num_classes))
        self.b_p = nn.Parameter(0.01 * torch.randn(num_classes))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        c = torch.zeros(self.batch_size, self.num_hidden).to(self.device)

        for step in range(self.seq_length):
            x_ = x[:, step, :]
        
            g = self.tanh(x_ @ self.W_gx + h @ self.W_gh + self.b_g)
            i = self.sigmoid(x_ @ self.W_ix + h @ self.W_ih + self.b_i)
            f = self.sigmoid(x_ @ self.W_fx + h @ self.W_fh + self.b_f)
            o = self.sigmoid(x_ @ self.W_ox + h @ self.W_oh + self.b_o)

            c = g * i + c * f
            h = self.tanh(c) * o

        p = h @ self.W_ph + self.b_p        # Hidden-to-output

        return p
