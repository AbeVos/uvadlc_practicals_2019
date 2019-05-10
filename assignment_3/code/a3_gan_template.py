import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
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
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
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
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.LeakyReLU(0.2),
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

    for epoch in range(args.n_epochs):
        for i, (images, _) in enumerate(dataloader):
            generator.train()

            # imgs.cuda()
            images = images.view(len(images), -1).to(device)

            z = torch.randn(args.batch_size, args.latent_dim).to(device)

            # Train Generator
            # ---------------
            samples = generator(z)

            pred_samples = discriminator(samples)

            loss_g = - torch.log(pred_samples).mean(0)

            loss_g.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            # Train Discriminator
            # -------------------
            samples = generator(z).detach()

            pred_images = discriminator(images)
            pred_samples = discriminator(samples)

            loss_d = - torch.log(pred_images).mean(0) \
                - torch.log(1 - pred_samples).mean(0)

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

                loss_g_mean = sum(loss_g_mean) / args.save_interval
                loss_d_mean = sum(loss_d_mean) / args.save_interval

                loss_g_plot.append(loss_g_mean)
                loss_d_plot.append(loss_d_mean)

                print(f"Epoch {epoch:03d}, batch {batches_done:06d} | "
                      f"J(G) = {loss_g_mean:.5f}, "
                      f"J(D) = {loss_d_mean:.5f}")

                samples = generator(sample_z)
                samples = samples.view(len(samples), 1, 28, 28)
                save_image(samples[:25], f'images/{batches_done:05d}.png',
                           nrow=5, normalize=True)

                loss_g_mean = []
                loss_d_mean = []

                plt.figure()
                plt.plot(loss_g_plot[1:], label=r"$J^{(G)}$")
                plt.plot(loss_d_plot[1:], label=r"$J^{(D)}$")
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig("gan_loss.png")
                plt.close()


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

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

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "gan_generator.pth")


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
    args = parser.parse_args()

    main()
