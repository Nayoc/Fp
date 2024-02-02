import torch
import pandas as pd
import os
import shutil

'''
    wifi_rssi文件下存放rssi信号数据
    分别提取bssid和rssi，映射后将rssi转置
'''

# 新rssi数据文件夹，名称为坐标，"_"分割坐标
wifi_rssi_path = "wifi_rssi"

# 坐标文件每行一个
coords_path = "wifi_rssi/coords.txt"

# 训练数据数据集
train_data_path = "data_csv/train_data.csv"

# 对于接收不到信号的ap的默认信号强度值
none_rssi = -110


def data_build():
    new_data = rssi_origin_read()

    add_dict_to_csv(new_data, train_data_path)
    a = data_read()
    print(a)


def add_dict_to_csv(data_dict, csv_file):
    # 尝试读取 CSV 文件，如果文件不存在则创建一个空的 DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame()

    # 将字典转换为 DataFrame，并添加到现有 DataFrame 或作为新的 DataFrame
    new_data = pd.DataFrame(data_dict, index=[0])
    df = pd.concat([df, new_data], axis=0, ignore_index=True, sort=False)

    # 将更新后的 DataFrame 写入到 CSV 文件中
    df.to_csv(csv_file, index=False)


def data_read():
    if os.path.isfile(train_data_path):
        return pd.read_csv(train_data_path)
    else:
        data = pd.DataFrame()
        data.to_csv(train_data_path, index=False)
        return data
    # for data in datas:


def rssi_origin_read():
    rssi_data = []
    files = os.listdir(wifi_rssi_path)
    files = [file for file in files if file.endswith(".csv")]

    coords_array = read_coords()

    for i in range(len(files)):
        if not files[i].endswith('.csv'):
            break
        coords = coords_array[i]
        file_path = os.path.join(wifi_rssi_path, files[i])

        rssi_origin_csv = pd.read_csv(file_path)

        bssid_list = rssi_origin_csv['BSSID'].tolist()
        rssi_list = rssi_origin_csv["RSSI"].tolist()

        dict = {}
        for i in range(len(bssid_list)):
            dict[bssid_list[i]] = rssi_list[i]

        dict['x'] = coords[0]
        dict['y'] = coords[1]
        rssi_data.append(dict)

    return rssi_data


def read_coords():
    coords = []

    with open(coords_path, 'r') as file:
        for line in file:
            # 去除换行符并按逗号分割坐标
            x, y = map(float, line.strip().split(','))
            coords.append((x, y))
    return coords


data_build()
