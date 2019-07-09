import argparse
from check_dataset import mytrainset

import torch
import torch.utils.data

if __name__ == "__main__":
    # hyper_parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_loader = torch.utils.data.DataLoader(mytrainset(train=True),batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(mytrainset(train=False))

    model = torch.hub.load('pytorch/vision','alexnet',pretrained=True)
    model.to("cuda")

    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
        print(label)
        break