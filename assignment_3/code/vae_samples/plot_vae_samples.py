import matplotlib.pyplot as plt
import matplotlib.image as mpimg

indices = [0, 19, 39]

samples = [mpimg.imread(f'sample_{idx:03d}.png') for idx in indices]
means = [mpimg.imread(f'mean_{idx:03d}.png') for idx in indices]

for idx, sample in enumerate(samples):
    plt.subplot(2, 3, idx+1)
    plt.imshow(sample, cmap='gray')
    plt.axis('off')
    plt.title(f"Epoch {indices[idx]+1}")

for idx, sample in enumerate(means):
    plt.subplot(2, 3, idx+4)
    plt.imshow(sample, cmap='gray')
    plt.axis('off')
    plt.title(f"Epoch {indices[idx]+1}")

plt.savefig('vae_samples.pdf')
