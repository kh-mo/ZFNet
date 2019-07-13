import os
import argparse
from check_dataset import mytrainset

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as f

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.fc1 = nn.Linear(6*6*64, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        layer1_output, switch1 = self.layer1(x)
        layer2_output, switch2 = self.layer2(layer1_output)
        fc1_output = self.fc1(layer2_output.view(-1, 6*6*64))
        relu3_output = self.relu3(fc1_output)
        fc2_output = self.fc2(relu3_output)
        return fc2_output

    def visualization(self, x, layer="layer1", max=True):
        if layer == "layer1":
            layer1_output, switch1 = self.layer1(x)
            if max == True:
                mat_matrix = get_max_mat(layer1_output)
                layer1_output = torch.mul(layer1_output, mat_matrix)
                deconv1 = f.conv_transpose2d(self.unpool(layer1_output, switch1), self.layer1[0].weight)
            else:
                deconv1 = f.conv_transpose2d(self.unpool(layer1_output, switch1), self.layer1[0].weight)
        else:
            layer1_output, switch1 = self.layer1(x)
            layer2_output, switch2 = self.layer2(layer1_output)
            if max == True:
                mat_matrix = get_max_mat(layer2_output)
                layer2_output = torch.mul(layer2_output, mat_matrix)
                deconv2 = f.conv_transpose2d(self.unpool(layer2_output, switch2), self.layer2[0].weight)
            else:
                deconv2 = f.conv_transpose2d(self.unpool(layer2_output, switch2), self.layer2[0].weight)
            deconv1 = f.conv_transpose2d(self.unpool(deconv2, switch1), self.layer1[0].weight)
        return deconv1

# get accuracy score function
def get_acc(pred, label):
    pred_idx = torch.argmax(pred, dim=1)
    acc = 0
    for idx, _ in enumerate(pred_idx):
        if pred_idx[idx].item() == label[idx].item():
            acc += 1
    acc /= len(pred_idx)
    return acc

# get max matrix for visualization
def get_max_mat(output):
    indice = (output==torch.max(output)).nonzero()
    max_mat = torch.zeros_like(output)
    max_mat[indice[0][0],indice[0][1],indice[0][2],indice[0][3]] = 1
    return max_mat

if __name__ == "__main__":
    # hyper_parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_loader = torch.utils.data.DataLoader(mytrainset(train=True),batch_size=args.batch_size)

    # model
    model = ConvNet()
    model.to(device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = torch.nn.CrossEntropyLoss().to(device=args.device)

    # training
    for epoch in range(args.epochs):
        loss_list = []
        acc_list = []
        for image, label in train_loader:
            image = image.to(torch.float32).to(device=args.device)
            label = label.to(device=args.device)

            pred = model(image)
            loss = loss_function(pred, label)
            loss_list.append(loss.item())
            acc_list.append(get_acc(pred, label))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch : {}, \tloss : {}, \tacc : {}".format(epoch, sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)))

    ## save model
    try:
        os.mkdir(os.path.join(os.getcwd(), "saved_model"))
    except FileExistsError as e:
        pass
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "saved_model/convnet"))

