"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import cifar10_utils
from collections import defaultdict
from convnet_pytorch import ConvNet

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
    Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    accuracy = (predictions.argmax(1) == targets.argmax(1)).sum().item()
    accuracy = accuracy / len(predictions)
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy

def train():
    """
    Performs training and evaluation of ConvNet model. 

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    model = ConvNet(3, 10).cuda()
    print(model)

    cv_size = 10000
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py',
                                        validation_size=cv_size)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    log = defaultdict(list)

    for step in range(FLAGS.max_steps):
        optimizer.zero_grad()

        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()

        h = model.forward(x)

        loss = criterion(h, y.argmax(1))
        loss.backward()
        
        optimizer.step()

        if step % FLAGS.eval_freq == 0:
            log['train_loss'].append(loss.item())
            log['train_acc'].append(accuracy(h, y))

            model.eval()

            x, y = cifar10['validation'].next_batch(cv_size)
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()

            h = model.forward(x)

            loss = criterion(h, y.argmax(1))

            log['cv_loss'].append(loss.item())
            log['cv_acc'].append(accuracy(h, y))

            model.train()

            print(
                f"Step {step} | "
                f"Training loss: {log['train_loss'][-1]:.5f}, "
                f"accuracy: {100 * log['train_acc'][-1]:.1f}% | "
                f"CV loss: {log['cv_loss'][-1]:.5f}, "
                f"accuracy: {100 * log['cv_acc'][-1]:.1f}%")

    model.eval()
    x, y = cifar10['test'].next_batch(cifar10['test'].num_examples)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    h = model.forward(x)

    loss = criterion(h, y.argmax(1))

    print(f"Test loss: {loss.item()}, accuracy: {100 * accuracy(h, y):.1f}%")

    # Plot loss and accuracy.
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(log['train_loss'], label="Training")
    plt.plot(log['cv_loss'], label="Cross Validation")
    plt.xlabel("Step")
    plt.legend()

    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(log['train_acc'], label="Training")
    plt.plot(log['cv_acc'], label="Cross Validation")
    plt.xlabel("Step")
    plt.legend()

    plt.legend()
    plt.show()
    ########################
    # END OF YOUR CODE    #
    #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
