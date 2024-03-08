from torch import nn
import torch
import torchvision.models as models
from torch.nn import functional as F


class NormalCnn(nn.Module):
    def __init__(self, _out):
        super().__init__()
        self.net = nn.Sequential()
        return 1

    def forward(self, x):
        x = self.net(x)
        return x


class mMultilayer(nn.Module):
    def __init__(self, _in, _out=2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(_in, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, _out))

    def forward(self, x):
        x = self.net(x)
        return x


class mLeNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, num_classes))

    def forward(self, x):
        x = self.net(x)
        return x


class mAlexNet(nn.Module):
    def __init__(self, _out: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, _out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimuCnn1(nn.Module):
    def __init__(self, num_classes=2):
        super(SimuCnn1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 修改全连接层的输入尺寸
        self.fc2 = nn.Linear(128, num_classes)  # 输出2个标签，坐标 x 和 y

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 7 * 7)  # 将特征图展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimuCnn2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 修改全连接层的输入尺寸
        self.fc2 = nn.Linear(128, num_classes)  # 输出2个标签，坐标 x 和 y

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 4 * 4)  # 将特征图展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class WiCnn2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DNN(nn.Module):
    def __init__(self, in_shape):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(in_shape, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 32)  # 隐藏层到隐藏层
        self.fc4 = nn.Linear(32, 2)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一个隐藏层，激活函数使用ReLU
        x = F.relu(self.fc2(x))  # 第二个隐藏层，激活函数使用ReLU
        x = F.relu(self.fc3(x))  # 第三个隐藏层，激活函数使用ReLU
        x = self.fc4(x)  # 输出层，不使用激活函数
        return x


class WiCnn1(nn.Module):
    def __init__(self, num_classes=40):
        super(WiCnn1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 4, 128)  # 修改全连接层的输入尺寸
        self.fc2 = nn.Linear(128, num_classes)  # 输出2个标签，坐标 x 和 y

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 5 * 4)  # 将特征图展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 20, 2)


class Conv1xnV3(nn.Module):
    def __init__(self, drop_out=0.5):
        super().__init__()
        # 样本较小，所以不用大stride降低尺寸，而是用maxpool缓降
        # 第一、二层层平滑特征图，通过池化减小尺寸
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),  # 8
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),  # 3
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(128, 64, kernel_size=(1, 3), padding=(0, 1)),
            # nn.BatchNorm2d(64),
            nn.Dropout(drop_out)
        )
        self.fc1 = nn.Linear(192, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 64 * 1 * 3)
        x = self.fc1(x)
        return x


class Conv1xnV2(nn.Module):
    def __init__(self, m, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, int(m / 3)), stride=int(m / 6))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, int(m / 6)), stride=int(m / 9))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, int(m / 9)), stride=int(m / 12))
        self.bn1 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 1 * 1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = x.view(-1, 128 * 1 * 1)
        x = self.fc1(x)
        return x


class Conv1xnV1(nn.Module):
    def __init__(self, m, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1, int(m / 3)), stride=int(m / 6))
        self.bn1 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 1 * 5, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = x.view(-1, 128 * 1 * 5)
        x = self.fc1(x)
        return x
