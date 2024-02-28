
import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))

print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.__version__)