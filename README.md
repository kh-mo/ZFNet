# ZFNet

This repository is implemented by Pytorch.

**ZFNet**: [Paper](https://arxiv.org/abs/1311.2901) | accepted ECCV 2014

## Getting Start
### Download data
- Download in current space
- Input : CIFAR10 URL
- Output : raw_data folder, 8 files
```shell
python dataset_download.py
```

### Check dataset
- Make data instance for dataloader
- Input : raw_data folder, data_batch files
- Output : show image
```shell
python check_dataset.py
```

### modeling
- Input : train files
- Output : convnet model
```shell
python modeling.py \
    --batch_size = 128
    --epochs = 50
```

### evaluating
- Input : convnet model, test files
- Output : predict category
```shell
python evaluating.py
```
