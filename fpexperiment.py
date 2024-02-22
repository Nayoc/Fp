import torch
from torch.utils import data
from torch import nn
import mtrain
import torchvision.models as models
import net.fpcnn as fpcnn


def fp_train(device='cpu'):
    train_iter, test_iter = mtrain.load_data(batch_size=200)

    net = fpcnn.mAlexNet(_out=2, dropout=0.1)
    # net = fpcnn.mLeNet()
    # net = fpcnn.mMultilayer(_in=900)
    # net = fpcnn.mResNet(input_channels=1, num_channels=2)

    net = net.to(device=mtrain.try_gpu())
    # 展示数据所在设备
    print(net[0].weight.data.device)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.3)

    epochs = 5000

    '''
        fp_AlexNet.params
        fp_ResNet.params
    '''
    mtrain.train(net, train_iter, test_iter, loss, epochs, trainer,
                 model_file='fp_AlexNet.params', record_term=100)


if __name__ == '__main__':
    fp_train()
