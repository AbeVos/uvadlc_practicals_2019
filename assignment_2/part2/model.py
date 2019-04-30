# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0',
                 dropout_keep_prob=0):
        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers,
                            dropout=dropout_keep_prob)
        self.relu = nn.ReLU()

        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x):
        # Implementation here...
        out, _ = self.lstm(x)

        out = self.linear(out)

        return out

    def sample(self):
        """
        Generate a sample.
        """
        indices = torch.LongTensor(self.batch_size,1).random_(
            0, self.vocabulary_size).to(self.device)

        x = torch.zeros(self.batch_size, self.vocabulary_size).to(self.device)
        x.scatter_(1, indices, 1)
        x = x.unsqueeze(0)

        output = [x]

        x, (h, c) = self.lstm(x)
        x = self.relu(x)
        x = self.linear(x)

        output.append(x)

        for i in range(self.seq_length - 2):
            x, (h, c) = self.lstm(x, (h, c))
            x = self.relu(x)
            x = self.linear(x)
            
            output.append(x)

        out = torch.stack(output).squeeze()
        return out
