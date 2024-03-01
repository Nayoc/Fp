import torch
from torch import nn

a = torch.Tensor([0.1233112])

print(torch.round(a, decimals=1))
print(round(a.item(), 3))
