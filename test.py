import numpy as np
import struct
import torch

a = torch.arange(36)
a = a.reshape((4, 1, 3, 3))
b = torch.arange(8)
b = b.reshape((4, 2))
print(a)


def norm(data):
    min = data.min()
    max = data.max()
    # norm_data = 2 * (data - min) / (max - min) - 1
    norm_data = (data - min) / (max - min)
    return norm_data

# def denorm(data,):

# def denorm(tensor,)

norm(a)
