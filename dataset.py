import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 仅使用信号强度的数据集对象,后两位为特征坐标x,y
class DatasetM1(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-2].values  # 前面的列作为特征
        self.labels = self.data.iloc[:, -2:].values  # 最后两列作为标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

