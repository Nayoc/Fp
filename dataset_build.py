from env.simulate_indoor import simulate,batch_simulate
import gzip
import numpy as np
import sys
import os
import json

simulate_root_path = os.path.dirname(__file__) + "/data/dataset/simulate/"


def simulate_dataset_build():
    # train_rssi, train_label, test_rssi, test_label = simulate()
    train_rssi, train_label, test_rssi, test_label = batch_simulate(30)

    # 共记录5个文件：训练数据集，训练标签，测试数据集，测试标签，数据尺寸json
    size = {}
    size['train_rssi.gz'] = train_rssi.shape
    size['train_label.gz'] = train_label.shape
    size['test_rssi.gz'] = test_rssi.shape
    size['test_label.gz'] = test_label.shape

    with open(simulate_root_path + 'size.json', "w") as f:
        json.dump(size, f)

    train_rssi = train_rssi.reshape(-1)
    with gzip.open(simulate_root_path + 'train_rssi.gz', 'wb') as f:
        f.write(train_rssi)

    train_label = train_label.reshape(-1)
    with gzip.open(simulate_root_path + 'train_label.gz', 'wb') as f:
        f.write(train_label)

    test_rssi = test_rssi.reshape(-1)
    with gzip.open(simulate_root_path + 'test_rssi.gz', 'wb') as f:
        f.write(test_rssi)

    test_label = test_label.reshape(-1)
    with gzip.open(simulate_root_path + 'test_label.gz', 'wb') as f:
        f.write(test_label)


if __name__ == '__main__':
    simulate_dataset_build()
    # test_read()
