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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def train(config):
    def compute_accuracy(outputs, targets):
        """
        Compute the accuracy of the predicitions.
        """
        outputs = torch.argmax(outputs, -1)

        return (outputs == targets).float().mean()

    def sample_text(outputs):
        # TODO: Sample from a single character.
        outputs = torch.argmax(outputs, -1)

        return [dataset.convert_to_string(x.cpu().numpy())
                for x in outputs.t()]

        return dataset.convert_to_string(outputs.t()[i].cpu().numpy())

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=4)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size, config.seq_length, dataset.vocab_size,
        config.lstm_num_hidden, config.lstm_num_layers, device,
        config.dropout_keep_prob).to(device)

    learning_rate = config.learning_rate

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce='sum')  # fixme
    optimizer = optim.Adam(model.parameters(), learning_rate)  # fixme

    x_onehot = torch.FloatTensor(config.seq_length, config.batch_size,
                                 dataset.vocab_size).to(device)
    y_onehot = torch.FloatTensor(config.seq_length, config.batch_size,
                                 dataset.vocab_size).to(device)

    # HACK: config.train_steps seems to be of type 'float' instead of 'int'.
    config.train_steps = int(config.train_steps)

    step = 0
    
    loss_list = []
    accuracy_list = []

    while step < config.train_steps:
        for batch_inputs, batch_targets in data_loader:

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            optimizer.zero_grad()

            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).to(device)
            # print(dataset.convert_to_string(batch_inputs.t()[0].cpu().numpy()))

            try:
                x_onehot.zero_()
                x_onehot.scatter_(2, batch_inputs.unsqueeze(-1), 1)
            except RuntimeError:
                continue

            y = model(x_onehot)

            loss = criterion(y.view(-1, dataset.vocab_size), batch_targets.view(-1))

            loss.backward()
            optimizer.step()

            loss = loss.item()   # fixme
            accuracy = compute_accuracy(y, batch_targets)  # fixme

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            loss_list.append(loss)
            accuracy_list.append(accuracy)

            plt.figure()
            plt.plot(loss_list)
            plt.plot(accuracy_list)
            plt.savefig('loss.png')
            plt.close()

            if step % config.learning_rate_step == 0:
                learning_rate = config.learning_rate_decay * learning_rate
                print(learning_rate)
                optimizer = optim.Adam(model.parameters(), learning_rate)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                      "Examples/Sec = {:.2f}, Accuracy = {:.2f}, "
                      "Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                inputs = sample_text(x_onehot)
                output = sample_text(y)
                sample = sample_text(model.sample())

                for idx in range(5):
                    print(f"{inputs[idx]} | {output[idx]} | {sample[idx]}")


            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this
                # bug report: https://github.com/pytorch/pytorch/pull/9655
                break
            else:
                step += 1

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()

    # Train the model
    train(config)
