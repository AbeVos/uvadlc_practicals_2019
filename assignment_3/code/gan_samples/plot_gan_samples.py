import matplotlib.pyplot as plt
import matplotlib.image as mpimg

indices = [0, 93500, 187500]

samples = [mpimg.imread(f'sample_{idx:06d}.png') for idx in indices]

plt.figure(figsize=(6,3))
plt.tight_layout()

for idx, sample in enumerate(samples):
    plt.subplot(1, 3, idx+1)
    plt.imshow(sample, cmap='gray')
    plt.axis('off')
    plt.title(f"Epoch {indices[idx]+1}")

plt.savefig('gan_samples.pdf', bbox_inches='tight')
