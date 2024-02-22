import numpy as np
import struct

num = 12.3
i = 0
while num > 1 or num < 0.1:
    if num >= 1:
        num /= 10
        i += 1
        continue
    if num < 0.1:
        num *= 10
        i -= 1
        continue
    break
print(num)
print(i)
