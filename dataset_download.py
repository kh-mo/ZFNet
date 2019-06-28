""" Download CIFAR10 dataset from <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz> """

import os
import tarfile
import urllib.request

def download_CIFAR10():
    # dataset url
    urls = ["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"]
    raw_folder = 'raw_data'
    empty_folder = 'cifar-10-batches-py'

    # folder for download
    try:
        os.makedirs(os.path.join(raw_folder))
    except FileExistsError as e:
        pass

    for url in urls:
        print('Downloading ' + url)
        # download file
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        # unzip
        with tarfile.open(file_path) as tar:
            for member in tar.getmembers():
                member.name = os.path.basename(member.name)
                tar.extract(member, path=raw_folder)

        # file remove
        os.unlink(file_path)
        os.rmdir(os.path.join(raw_folder, empty_folder))

    print("download done.")

if __name__ == "__main__":
    download_CIFAR10()
