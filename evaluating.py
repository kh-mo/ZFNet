import os
import torch
from modeling import ConvNet
from check_dataset import mytrainset, imshow
import torch.utils.data

if __name__ == "__main__":
    # hyper_parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model & pretrain parameter load
    model = ConvNet().to(device=device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_model/convnet")))

    # test data load
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    test_loader = torch.utils.data.DataLoader(mytrainset(train=False), batch_size=1)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # final output
    # imshow(images[0]) ## torch.uint8, min 0, max 255, shape [3, 32, 32]
    print(classes[torch.argmax(model(images.to(torch.float32).to(device=device)))])
