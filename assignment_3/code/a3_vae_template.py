import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear1 = nn.Linear(28 ** 2, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, z_dim)
        self.linear_logvar = nn.Linear(hidden_dim, z_dim)

        self.activation = nn.ReLU()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        input = self.activation(self.linear1(input))

        mean = self.linear_mu(input)
        std = self.activation(self.linear_logvar(input))

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 28 ** 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.relu(self.linear1(input))
        mean = self.sigmoid(self.linear2(mean))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

        self.recon_loss = nn.BCELoss()

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        def kl_divergence(mean, logvar):
            """
            Compute the KL-divergence between the predicted mean and standard
            deviation and the standard normal distribution.
            """
            var = torch.exp(logvar)
            std = torch.pow(var, 2)

            return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        mean, logvar = self.encoder(input)
        std = torch.pow(torch.exp(logvar), 2)

        noise = torch.randn(std.size())
        z = noise * std + mean

        output = self.decoder(z)

        D_kl = kl_divergence(mean, logvar).mean(-1)
        recon_loss = self.recon_loss(output, input)

        average_negative_elbo = (recon_loss + D_kl).mean(0)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn(n_samples, self.z_dim)
        im_means = self.decoder(z)

        sampled_ims = torch.rand(*im_means.shape) < im_means
        sampled_ims = sampled_ims.view(-1, 28, 28)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    total_avg_neg_elbo = []

    for idx, batch in enumerate(data):
        batch = batch.view(len(batch), -1)
        avg_neg_elbo = model(batch)

        optimizer.zero_grad()
        avg_neg_elbo.backward()
        optimizer.step()

        total_avg_neg_elbo.append(avg_neg_elbo.item())

    average_epoch_elbo = sum(total_avg_neg_elbo) / len(total_avg_neg_elbo)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def save_sample_plot(samples, filename):
    n = len(samples)
    samples = samples.view(n, 1, 28, 28)
    plt.figure()
    grid = make_grid(samples, nrow=n)
    plt.imshow(grid.permute(1, 2, 0) * 255)
    plt.axis('off')
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(hidden_dim=500, z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    samples, means = model.sample(5)
    save_sample_plot(samples, f"samples_noise.png")

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------

        samples, means = model.sample(5)
        save_sample_plot(samples, f"samples_{epoch}.png")

        if ARGS.zdim is 2:
            save_manifold_plot(means, f"manifold_{epoch}.png")

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
