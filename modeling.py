import os
import argparse
from check_dataset import mytrainset

import torch
import torch.nn as nn
import torch.utils.data

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.fc1 = nn.Linear(6*6*64, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)
        self.switch1 = None
        self.switch2 = None

    def forward(self, x):
        conv1_output = self.conv1(x)
        relu1_output = self.relu1(conv1_output)
        pool1_output, self.switch1 = self.pool1(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = self.relu2(conv2_output)
        pool2_output, self.switch2 = self.pool2(relu2_output)
        fc1_output = self.fc1(pool2_output.view(-1, 6*6*64))
        relu3_output = self.relu3(fc1_output)
        fc2_output = self.fc2(relu3_output)
        return fc2_output

    def visualization(self, x):
        conv1_output = self.conv1(x)
        relu1_output = self.relu1(conv1_output)
        pool1_output, self.switch1 = self.pool1(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = self.relu2(conv2_output)
        pool2_output, self.switch2 = self.pool2(relu2_output)

        return

if __name__ == "__main__":
    # hyper_parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_loader = torch.utils.data.DataLoader(mytrainset(train=True),batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(mytrainset(train=False))

    # model
    model = ConvNet()
    model.to(device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = torch.nn.CrossEntropyLoss().to(device=args.device)

    # training
    for epoch in range(args.epochs):
        loss_list = []
        for image, label in train_loader:
            image = image.to(torch.float32).to(device=args.device)
            label = label.to(device=args.device)

            pred = model(image)
            loss = loss_function(pred, label)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch :", epoch, "\tloss :", sum(loss_list)/len([loss_list]))

    try:
        os.mkdir(os.path.join(os.getcwd(), "saved_model"))
    except FileExistsError as e:
        pass
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "saved_model/convnet"))