import torch
from torch import nn
from torch.utils import data
from mplt import Animator
import os
import gzip
import numpy as np
import json

# 模拟数据集地址
simulate_data_path = os.path.dirname(__file__) + '/data/dataset/simulate/'

model_params_path = os.path.dirname(__file__) + '/model/'

# 坐标误差范围表示准确率
error_scale = 1.5

# 标签归一化参数，用于还原标签值
norm_label_params = ()


def load_data(device, type='simulate', batch_size=20):


    train_rssi = torch.tensor(data_read(filename='train_rssi.gz', type=type), device=try_gpu())
    train_label = torch.tensor(data_read(filename='train_label.gz', type=type), device=try_gpu())
    test_rssi = torch.tensor(data_read(filename='test_rssi.gz', type=type), device=try_gpu())
    test_label = torch.tensor(data_read(filename='test_label.gz', type=type), device=try_gpu())


    # 扩展加入通道维度
    train_rssi = extand(train_rssi)
    test_rssi = extand(test_rssi)

    norm = nn.BatchNorm2d(num_features=1)
    norm_train_rssi = norm(train_rssi)
    norm_test_rssi = norm(test_rssi)

    norm_train_label = norm_label(train_label)
    norm_test_label = norm_label(test_label)

    train_data = data.TensorDataset(norm_train_rssi, norm_train_label)
    test_data = data.TensorDataset(norm_test_rssi, norm_test_label)

    return (data.DataLoader(train_data, batch_size, shuffle=True),
            data.DataLoader(test_data, batch_size, shuffle=True))


def norm_label(labels):
    # 计算 x 和 y 维度的绝对值的最大值
    abs_max_x = torch.abs(labels)[:, 0].max()
    abs_max_y = torch.abs(labels)[:, 1].max()

    global norm_label_params
    norm_label_params = torch.tensor((abs_max_x, abs_max_y))

    # 归一化标签
    normalized_labels = labels / norm_label_params

    return normalized_labels


def denormalize_labels(normalized_labels, abs_max):
    # 逆归一化标签
    denormalized_labels = normalized_labels * abs_max

    return denormalized_labels


def extand(data, dim=1):
    # 将数据的通道维度扩展为 1
    return torch.unsqueeze(data, dim=dim)


def data_read(filename='train_rssi.gz', type='simulate'):
    if type == 'simulate':
        data_url = simulate_data_path

    with open(data_url + 'size.json', 'r') as f:
        size = json.load(f)

    with gzip.open(data_url + filename, 'rb') as f:
        train_label = np.frombuffer(f.read(), dtype=np.double, offset=0).astype(np.float32)

    return train_label.reshape(size[filename])


def size_read(type='simulate'):
    if type == 'simulate':
        data_url = simulate_data_path

    with open(data_url + 'size.json', 'r') as f:
        size = json.load(f)

    # rssi 取从第二维开始的维度信息
    single_rssi_shape = size['train_rssi.gz'][1:]

    return single_rssi_shape


# 坐标差距在一定阈值内
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    y_hat_origin = denormalize_labels(y_hat, norm_label_params)
    y_origin = denormalize_labels(y, norm_label_params)

    distance = torch.norm(y_hat_origin - y_origin, dim=1)
    # print(f'y_hat:{y_hat},y:{y}')

    return (distance < error_scale).sum().item()


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward(retain_graph=True)
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward(retain_graph=True)
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def judge_loss_weight(num):
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

    return num, i


def train(net, train_iter, test_iter, loss, num_epochs, updater,
          model_file='fpcnn.params', record_term=100):
    # 加载历史训练模型
    net = load(net, model_file)

    # 权重系数，放大loss观察值
    global_loss_weight = 1
    term_loss_weight = 1

    # 预训练一次，确定损失数量级
    train_loss, train_acc = train_epoch(net, train_iter, loss, updater)

    animator_global = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 0.9],
                               legend=['train loss ', 'train acc', 'test acc'])
    animator_term = Animator(xlabel='epoch', xlim=[1, record_term], ylim=[0, 0.9],
                             legend=['train loss', 'train acc', 'test acc'])

    """训练模型"""
    for epoch in range(num_epochs):

        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        if epoch == 0:
            # 调整损失值到0.1-1区间方便观察
            global_weight_train_loss, _e = judge_loss_weight(train_loss)
            term_weight_train_loss = global_weight_train_loss
            global_loss_weight = 10 ** _e
            term_loss_weight = 10 ** _e
            _e = str(_e) if _e > 0 else '(' + str(_e) + ')'
            animator_global.rename_legend('train loss' + ' 10^' + _e, _index=0)
            animator_term.rename_legend('train loss' + ' 10^' + _e, _index=0)
        else:
            global_weight_train_loss = train_loss / global_loss_weight
            # 周期迭代记录一次文件并绘图——100次后记录并绘图
            if epoch % record_term == 0:
                term_weight_train_loss, _e = judge_loss_weight(train_loss)
                term_loss_weight = 10 ** _e
                animator_term.draw()
                animator_term.clear()
                _e = str(_e) if _e > 0 else '(' + str(_e) + ')'
                animator_term.rename_legend('train loss' + ' 10^' + _e, _index=0)
                save(net, model_file)

            else:
                term_weight_train_loss = train_loss / term_loss_weight

        animator_global.update(epoch + 1, (global_weight_train_loss, train_acc) + (test_acc,))
        animator_term.update(epoch % record_term + 1, (term_weight_train_loss, train_acc) + (test_acc,))
        print(f'epoch:{epoch},loss:{global_weight_train_loss},train_acc:{train_acc},test_acc:{test_acc}')

    animator_global.draw()

    # assert train_loss < 1, train_loss
    assert train_acc <= 1 and train_acc > 0, train_acc
    assert test_acc <= 1 and test_acc > 0, test_acc

    save(net, model_file)

    return train_loss, train_acc


def save(net, params_file='fpcnn.params'):
    torch.save(net.state_dict(), model_params_path + params_file)


def load(net, filename):
    try:
        net.load_state_dict(torch.load(model_params_path + filename))
        net.eval()
        print(filename + ' is loaded')
        return net
    except Exception as e:
        print('no model is loaded')
        return net

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
