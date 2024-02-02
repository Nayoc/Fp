import torch
from torch import nn
from torch.nn import functional as F


class NormalCnn(nn.Module):
    def __init__(self,in_channels,out_channels):
        return 1