import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch.utils.data

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class mytrainset:
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    ]
    def __init__(self):
        self.data = []

        for file_name in self.train_list:
            file_path = os.path.join(os.getcwd(), "raw_data", file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

    def __getitem__(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    md = mytrainset()
    mdloader = torch.utils.data.DataLoader(md, batch_size=4, shuffle=False)
    dataiter = iter(mdloader)
    images = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
