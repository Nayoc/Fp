import torch
from torch.utils import data
from view.mplt import Animator
import os
import gzip
import numpy as np
import json

# 模拟数据集地址
simulate_data_path = os.path.dirname(__file__) + '/data/dataset/simulate/'

wi_data_path = os.path.dirname(__file__) + '/data/dataset/wi/'

model_params_path = os.path.dirname(__file__) + '/model/'

# 坐标误差范围表示准确率
error_scale = 2

# 标签归一化参数，用于还原标签值
norm_label_params = (-10, 50)


def load_data(source='simulate', batch_size=20):
    train_rssi = torch.tensor(data_read(filename='train_rssi.gz', source=source), device=try_gpu())
    train_label = torch.tensor(data_read(filename='train_label.gz', source=source), device=try_gpu())
    test_rssi = torch.tensor(data_read(filename='test_rssi.gz', source=source), device=try_gpu())
    test_label = torch.tensor(data_read(filename='test_label.gz', source=source), device=try_gpu())

    # 扩展加入通道维度
    train_rssi = extand(train_rssi)
    test_rssi = extand(test_rssi)

    norm_train_rssi = norm(train_rssi)
    norm_test_rssi = norm(test_rssi)

    norm_train_label = norm(train_label)
    norm_test_label = norm(test_label)

    train_data = data.TensorDataset(norm_train_rssi, norm_train_label)
    test_data = data.TensorDataset(norm_test_rssi, norm_test_label)

    return (data.DataLoader(train_data, batch_size, shuffle=True),
            data.DataLoader(test_data, batch_size, shuffle=True))


def norm_label(labels):
    # 归一化标签
    normalized_labels = labels / norm_label_params

    return normalized_labels


def norm(data, norm_params=None):
    if norm_params is None:
        min = data.min()
        max = data.max()
    else:
        min = norm_params[0]
        max = norm_params[1]
    # norm_data = 2 * (data - min) / (max - min) - 1
    norm_data = (data - min) / (max - min)
    return norm_data


def denorm(data, norm_params):
    min = norm_params[0]
    max = norm_params[1]
    origin = data * (max - min) + min
    # origin = (data + 1) * (max - min) / 2 + min
    return origin


def denormalize_labels(normalized_labels, abs_max):
    # 逆归一化标签
    denormalized_labels = normalized_labels * abs_max

    return denormalized_labels


def extand(data, dim=1):
    # 将数据的通道维度扩展为 1
    return torch.unsqueeze(data, dim=dim)


def data_read(filename='train_rssi.gz', source='simulate'):
    if source == 'simulate':
        data_url = simulate_data_path

    if source == 'wi':
        data_url = wi_data_path

    with open(data_url + 'size.json', 'r') as f:
        size = json.load(f)

    with gzip.open(data_url + filename, 'rb') as f:
        train_label = np.frombuffer(f.read(), dtype=np.double, offset=0).astype(np.float32)

    return train_label.reshape(size[filename])


def size_read(source='simulate'):
    if source == 'simulate':
        data_url = simulate_data_path

    with open(data_url + 'size.json', 'r') as f:
        size = json.load(f)

    # rssi 取从第二维开始的维度信息
    single_rssi_shape = size['train_rssi.gz'][1:]

    return single_rssi_shape


# 返回四个维度的数据，规定误差内的准确数量，平均误差距离，最小误差距离，最大误差距离
def count_distance(y_hat, y):
    """计算预测正确的数量"""
    y_hat_origin = denorm(y_hat, norm_label_params)
    y_origin = denorm(y, norm_label_params)

    distance = torch.norm(y_hat_origin - y_origin, dim=len(y_origin.shape) - 1)
    accuracy = (distance < error_scale).sum().item()

    return accuracy, distance.mean(), distance.min(), distance.max()


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


def evaluate_result(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            distance = count_distance(net(X), y)
            metric.add(distance[0], y.numel() / 2)
    return metric[0] / metric[1], distance[1], distance[2], distance[3]


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数、平均误差距离、最小误差距离、最大误差距离
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

        distance = count_distance(y_hat, y)
        # y.numel()/2是因为最后距离是是坐标聚合出来的，所以总数只有一半
        metric.add(float(l.sum()), distance[0], y.numel() / 2)
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2], distance[1], distance[2], distance[3]


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
    train_loss, train_acc, mean_error, min_error, max_error = train_epoch(net, train_iter, loss, updater)

    animator_global = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                               legend=['train loss ', 'train acc', 'test acc'])
    animator_term = Animator(xlabel='epoch', xlim=[1, record_term], ylim=[0, 1],
                             legend=['train loss', 'train acc', 'test acc'])

    # 创建学习率调度器
    # scheduler = ExponentialLR(updater, gamma=0.9)

    # 打印当前学习率
    # current_lr = updater.state_dict()['param_groups'][0]['lr']

    """训练模型"""
    for epoch in range(num_epochs):

        train_loss, train_acc, train_mean_error, train_min_error, train_max_error = train_epoch(net, train_iter, loss,
                                                                                                updater)
        test_acc, test_mean_error, test_min_error, test_max_error = evaluate_result(net, test_iter)

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
                print('---------current learning rate:' + str(updater.param_groups[0]['lr']))

            else:
                term_weight_train_loss = train_loss / term_loss_weight

        animator_global.update(epoch + 1, (global_weight_train_loss, train_acc) + (test_acc,))
        animator_term.update(epoch % record_term + 1, (term_weight_train_loss, train_acc) + (test_acc,))
        torch.set_printoptions(sci_mode=False, precision=8)
        print(
            f'epoch:{epoch},loss:{round(train_loss, 8)},train_acc:{round(train_acc, 5)},test_acc:{round(test_acc, 5)} | '
            f'mean_error:{round(train_mean_error.item(), 5)}/{round(test_mean_error.item(), 5)},'
            f'min_error:{round(train_min_error.item(), 5)}/{round(test_min_error.item(), 5)},'
            f'max_error:{round(train_max_error.item(), 5)}/{round(test_max_error.item(), 5)}')

        adjust_learning_rate(updater, epoch)

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


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']

    if lr < 0.0001:
        return

    if epoch % 100 == 0 and epoch != 0:
        lr = lr * 0.9  # 学习率没100个epoch乘以0.9

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f'---learning rate change---,current lr is:{lr}')
