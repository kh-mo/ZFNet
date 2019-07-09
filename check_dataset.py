import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch.utils.data

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class mytrainset:
    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']

    def __init__(self, train=True):
        self.data = []
        self.labels = []
        self.downloaded_list = None

        if train == True:
            self.downloaded_list = self.train_list
        else:
            self.downloaded_list = self.test_list

        for file_name in self.downloaded_list:
            file_path = os.path.join(os.getcwd(), "raw_data", file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.labels += entry['labels']

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

    # __getitem__, __len__은 python의 규약(protocol) 일종
    # duck typing : if it walks like a duck and it quacks like a duck, then it must be a duck.
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    md = mytrainset(train=True)
    mdloader = torch.utils.data.DataLoader(md, batch_size=4, shuffle=False)
    dataiter = iter(mdloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
