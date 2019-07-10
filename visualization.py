import os
import torch
from modeling import ConvNet
from check_dataset import mytrainset, imshow
import torch.utils.data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device=device)
    model.state_dict(torch.load(os.path.join(os.getcwd(),"saved_model/convnet")))

    test_loader = torch.utils.data.DataLoader(mytrainset(train=False), batch_size=1)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    imshow(images[0])
    imshow(model.visualization(images.to(torch.float32).to(device=device))[0].to("cpu").detach())