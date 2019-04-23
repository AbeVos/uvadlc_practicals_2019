"""
Plot the values from 'results.csv', a file containing the final accuracy of the
vanilla RNN after training on the Palindrome dataset using palindromes of a
particular length.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

if __name__ == "__main__":
    model_type = 'LSTM'
    accuracies = defaultdict(list)

    with open('results.csv') as file:
        reader = csv.reader(file, delimiter=';')

        for row in reader:
            if len(row) is 3:
                length, accuracy, model = row

                if model != model_type:
                    continue
            else:
                if model_type == 'LSTM':
                    continue

                length, accuracy = row
            length = int(length)
            accuracy = float(accuracy)
            accuracies[length].append(accuracy)

    means = []
    medians = []
    vars = []
    maxs = []
    mins = []
    
    for length in accuracies:
        maxs.append(np.max(accuracies[length]))
        mins.append(np.min(accuracies[length]))
        vars.append(np.var(accuracies[length]))
        means.append(np.mean(accuracies[length]))
        medians.append(np.median(accuracies[length]))

    length = list(accuracies.keys())

    plt.fill_between(length, mins, maxs, alpha=0.5)
    plt.plot(length, means)
    plt.xlabel("Palindrome length")
    plt.ylabel("Accuracy")
    plt.show()
