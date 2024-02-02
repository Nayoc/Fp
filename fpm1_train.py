import torch
from torch.utils import data
from torch import nn
from fp.dataset import DatasetM1
from fp.net.fpcnn import NormalCnn

# 仅仅使用rssi训练

dataset_url = "/data_csv/train_data.csv"


def load_dataset(batch_size):
    ds = DatasetM1(dataset_url)
    return data.DataLoader(ds, batch_size, shuffle=True)


num_channels = 64
b1 = NormalCnn(1, 64)
net = nn.Sequential(b1,
                    nn.Flatten(),
                    nn.Linear(64, 2))
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD()
