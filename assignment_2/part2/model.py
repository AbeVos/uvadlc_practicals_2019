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
import torch.nn.functional as F
import torch.distributions as dist


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0',
                 dropout_keep_prob=1):
        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers,
                            dropout=dropout_keep_prob)

        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x):
        # Implementation here...
        out, _ = self.lstm(x)

        out = self.linear(out)

        return out

    def sample(self, random=False, temperature=0.5):
        """
        Generate a sample.
        """
        self.eval()

        indices = torch.LongTensor(self.batch_size,1).random_(
            0, self.vocabulary_size).to(self.device)

        x = torch.zeros(self.batch_size, self.vocabulary_size).to(self.device)
        x.scatter_(1, indices, 1)
        x = x.unsqueeze(0)

        output = [x]

        x, (h, c) = self.lstm(x)
        x = self.linear(x)

        output.append(x)

        for i in range(self.seq_length - 2):
            x, (h, c) = self.lstm(x, (h, c))
            x = self.linear(x)

            if random:
                # Apply temperature.
                x = x / temperature

                x = F.softmax(x, -1)    

                distribution = dist.Categorical(x)
                indices = distribution.sample().t()
            else:
                indices = x.argmax(-1).t()

            # Reshape into one-hot vectors.
            x.zero_().squeeze_()
            x.scatter_(1, indices, 1).unsqueeze_(0)
            output.append(x)

        out = torch.stack(output).squeeze()

        self.train()

        return out
