import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.layers = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 28 ** 2),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(28 ** 2, 512),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.layers(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
          device):
    sample_z = torch.randn(args.batch_size, args.latent_dim).to(device)

    loss_g_plot = []
    loss_d_plot = []
    loss_g_mean = []
    loss_d_mean = []

    batch_plot = []

    criterion = nn.BCELoss().to(device)

    for epoch in range(args.n_epochs):
        for i, (images, _) in enumerate(dataloader):
            generator.train()

            ones = torch.ones((len(images), 1)).to(device)
            zeros = torch.zeros((len(images), 1)).to(device)

            # imgs.cuda()
            images = images.view(len(images), -1).to(device)

            z = torch.randn(len(images), args.latent_dim).to(device)

            # Train Generator
            # ---------------
            samples = generator(z)

            pred_samples = discriminator(samples)

            loss_g = criterion(pred_samples, ones)

            loss_g.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            # Train Discriminator
            # -------------------
            samples = generator(z).detach()

            pred_images = discriminator(images)
            pred_samples = discriminator(samples)

            loss_d = criterion(pred_images, ones) \
                    + criterion(pred_samples, zeros)

            loss_d.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()

            loss_g_mean.append(loss_g.item())
            loss_d_mean.append(loss_d.item())

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                generator.eval()

                batch_plot.append(batches_done)

                loss_g_mean = sum(loss_g_mean) / args.save_interval
                loss_d_mean = sum(loss_d_mean) / args.save_interval

                loss_g_plot.append(loss_g_mean)
                loss_d_plot.append(loss_d_mean)

                print(f"Epoch {epoch:03d}, batch {batches_done:06d} | "
                      f"J(G) = {loss_g_mean:.5f}, "
                      f"J(D) = {loss_d_mean:.5f}")

                samples = generator(sample_z)
                samples = samples.view(len(samples), 1, 28, 28)
                save_image(samples[:25], f'gan_samples/sample_{batches_done:06d}.png',
                           nrow=5, normalize=True)

                loss_g_mean = []
                loss_d_mean = []

                plt.figure()
                plt.plot(batch_plot[1:], loss_g_plot[1:], label=r"$J^{(G)}$")
                plt.plot(batch_plot[1:], loss_d_plot[1:], label=r"$J^{(D)}$")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig("gan_loss.pdf")
                plt.close()

        # You can save your generator here to re-use it to generate images for your
        # report, e.g.:
        torch.save(generator.state_dict(), "gan_generator.pth")


def main():
    # Create output image directory
    os.makedirs('gan_samples', exist_ok=True)

    device = torch.device(args.device)

    # load data.
    dataset = datasets.MNIST(
        './data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    try:
        train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
              device)
    except KeyboardInterrupt:
        # Allow for manual early stopping.
        pass


def interpolate(model, n):
    """
    Randomly select two random points in the latent space and generate
    images corresponding to their interpolated coordinates.
    """
    def lerp(start, end, t):
        return (1 - t) * start + t * end

    start, end = torch.randn(2, args.latent_dim)

    samples = []

    for t in torch.linspace(0, 1, n):
        z = lerp(start, end, t).unsqueeze(0).to(args.device)
        samples.append(model(z))

    samples = torch.stack(samples)
    print(samples.shape)

    samples = samples.view(len(samples), 1, 28, 28)
    save_image(samples, "interpolation.png", nrow=n, normalize=True)

    grid = make_grid(samples, nrow=n, normalize=True).permute((1, 2, 0))
    plt.imshow(grid.cpu().detach().numpy())
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="")
    parser.add_argument('--interpolation', type=str, default=None,
                        help="Load an existing model and Interpolate between "
                             "two random points in the latent space.")
    args = parser.parse_args()

    if args.interpolation is None:
        main()
    else:
        model = Generator().to(args.device)
        model.load_state_dict(torch.load(args.interpolation))
        model.eval()

        interpolate(model, 9)
