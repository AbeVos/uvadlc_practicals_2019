import argparse
import torch

from train import sample_text
from dataset import TextDataset
from model import TextGenerationModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', type=str,
                        help="Path to the trained model.")
    parser.add_argument('-d', dest='data', type=str,
                        default='assets/book_EN_grimms_fairy_tails.txt',
                        help="Path to the dataset.")
    parser.add_argument('-t', dest='temperature', type=float, default=1,
                        help="Sampling temperature.")
    args = parser.parse_args()

    checkpoint = torch.load(args.path)

    dataset = TextDataset(args.data, 30)  # fixme

    model = TextGenerationModel(512, 30, 87, lstm_num_hidden=128).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Randomly sample sequences from the model.
    sample = model.sample(True, args.temperature)
    sample = sample_text(dataset, sample)

    for s in sample:
        print(s)
