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
    model.state_dict(torch.load(os.path.join(os.getcwd(), "saved_model/convnet")))

    # test data load
    test_loader = torch.utils.data.DataLoader(mytrainset(train=False), batch_size=1)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # visualization
    imshow(images[0])
    imshow(torch.mul(images[0].to(device), model.visualization(images.to(torch.float32).to(device=device),layer="layer2")[0].to(device).detach()))
    imshow(model.visualization(images.to(torch.float32).to(device=device),layer="layer2")[0].detach())
    imshow(model.visualization(images.to(torch.float32).to(device=device))[0].to(device).detach())

    imshow(torch.mul(images[0].to(torch.float32),model.visualization(images.to(torch.float32).to(device=device), layer="layer2")[0].to(device).detach()))

    images[0]
    vis_img = model.visualization(images.to(torch.float32).to(device=device), layer="layer2")[0].detach()

    from torchvision.utils import save_image
    save_image(vis_img, os.path.join(os.getcwd(),"t.jpg"))

    from PIL import Image
    import matplotlib.pyplot as plt

    plt.imshow(vis_img.mul_(255).add_(0.5).clamp_(0,255).permute(1,2,0).to("cpu", torch.uint8).numpy())
    plt.show()

    plt.imshow(vis_img.mul_(255).to("cpu",torch.uint8).permute(1,2,0))
    plt.show()