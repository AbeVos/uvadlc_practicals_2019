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
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

        self.activation = nn.ReLU()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        input = self.linear1(input)
        input = self.activation(input)

        mean = self.mean(input)
        logvar = self.logvar(input)

        return mean, logvar


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
        mean = self.linear2(mean)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device='cpu'):
        super().__init__()

        self.device = device

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

        self.recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

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
            return 0.5 * torch.sum(
                logvar.exp() + mean.pow(2) - 1 - logvar, -1)

        mean, logvar = self.encoder(input)

        noise = torch.randn(mean.size()).to(self.device)
        z = noise * torch.exp(0.5 * logvar) + mean

        output = self.decoder(z)

        D_kl = kl_divergence(mean, logvar).mean()
        recon_loss = self.recon_loss(output, input) / len(input)

        average_negative_elbo = recon_loss + D_kl

        return average_negative_elbo

    def sample(self, n_samples, z=None):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        if z is None:
            z = torch.randn(n_samples, self.z_dim).to(self.device)

        im_means = torch.sigmoid(self.decoder(z))

        sampled_ims = torch.rand(*im_means.shape).to(self.device) < im_means
        sampled_ims = sampled_ims.view(-1, 28, 28).float()

        im_means = im_means.view(-1, 28, 28).detach()

        return sampled_ims, im_means, z


def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    total_avg_neg_elbo = []

    for idx, batch in enumerate(data):
        batch = batch.view(len(batch), -1).to(device)

        avg_neg_elbo = model(batch)

        if model.training:
            optimizer.zero_grad()
            avg_neg_elbo.backward()
            optimizer.step()

        total_avg_neg_elbo.append(avg_neg_elbo.item())

    average_epoch_elbo = sum(total_avg_neg_elbo) / len(total_avg_neg_elbo)

    return average_epoch_elbo


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device)

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


def save_sample_plot(samples, filename, nrow=5):
    n = len(samples)
    samples = samples.view(n, 1, 28, 28)
    plt.figure()
    grid = make_grid(samples, nrow=nrow).cpu()
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


def plot_manifold(model, filename, nrow=10):
    """
    Plot the manifold of the first two latent dimensions.
    """
    x = torch.linspace(-1, 1, nrow)
    xv, yv = torch.meshgrid(x, x)
    z = torch.stack([xv, yv], 0)
    z = z.view(2, -1).t().to(model.device)

    samples, means, _ = model.sample(1, z)

    save_sample_plot(means, filename, nrow)


def main():
    device = torch.device(ARGS.device)

    data = bmnist()[:2]  # ignore test split
    model = VAE(hidden_dim=500, z_dim=ARGS.zdim, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # samples, means, z = model.sample(25)
    # save_sample_plot(samples, f"samples_noise.png")

    z = None

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer, device)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------

        samples, means, z = model.sample(25, z)
        save_sample_plot(means, f"samples_{epoch:03d}.png")

        if ARGS.zdim is 2:
            plot_manifold(model, "manifold.png")

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
    parser.add_argument('--device', default='cuda:0', type=str)

    ARGS = parser.parse_args()

    main()
