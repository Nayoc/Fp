import numpy as np
import struct
import torch

path = 'data/dataset/wi/rsrp.npy'
pos = 'data/dataset/wi/pos.npy'

data = np.load(pos)
print(type(data))