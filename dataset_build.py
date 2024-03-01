# from env.simulate_indoor import simulate, batch_simulate
import gzip
import numpy as np
import sys
import os
import json

# pylayers simulate data
simulate_root_path = os.path.dirname(__file__) + "/data/dataset/simulate/"
# wireless intelligence data
wi_root_path = os.path.dirname(__file__) + "/data/dataset/wi/"


# def simulate_build():
#     # train_rssi, train_label, test_rssi, test_label = simulate()
#     train_rssi, train_label, test_rssi, test_label = batch_simulate(30)
#     return train_rssi, train_label, test_rssi, test_label


def wi_build():
    # 80000 * (18+2)
    rsrp = np.load(wi_root_path + 'rsrp.npy')
    pos = np.load(wi_root_path + 'pos.npy')

    # 分为60000训练集，20000测试集
    train_num = 60000

    train_rssi = rsrp[0:train_num, :]
    train_rssi = train_rssi.reshape(-1, 1, 18)
    test_rssi = rsrp[train_num:, :]
    test_rssi = test_rssi.reshape(-1, 1, 18)

    train_label = pos[0:train_num, :]
    test_label = pos[train_num:, :]

    return train_rssi, train_label, test_rssi, test_label


def dataset_build(source='simulate'):
    # if source == 'simulate':
    #     root_path = simulate_root_path
    #     train_data, train_label, test_data, test_label = simulate_build()
    if source == 'wi':
        root_path = wi_root_path
        train_data, train_label, test_data, test_label = wi_build()

    # 共记录5个文件：训练数据集，训练标签，测试数据集，测试标签，数据尺寸json
    size = {}
    size['train_rssi.gz'] = train_data.shape
    size['train_label.gz'] = train_label.shape
    size['test_rssi.gz'] = test_data.shape
    size['test_label.gz'] = test_label.shape

    with open(root_path + 'size.json', "w") as f:
        json.dump(size, f)

    train_data = train_data.reshape(-1)
    with gzip.open(root_path + 'train_rssi.gz', 'wb') as f:
        f.write(train_data)

    train_label = train_label.reshape(-1)
    with gzip.open(root_path + 'train_label.gz', 'wb') as f:
        f.write(train_label)

    test_data = test_data.reshape(-1)
    with gzip.open(root_path + 'test_rssi.gz', 'wb') as f:
        f.write(test_data)

    test_label = test_label.reshape(-1)
    with gzip.open(root_path + 'test_label.gz', 'wb') as f:
        f.write(test_label)


if __name__ == '__main__':
    dataset_build('wi')
