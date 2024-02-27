import torch
from torch import nn
import mtrain
import net.fpcnn as fpcnn

'''
    指定内容：
    坐标区间
    误差距离
'''


def fp_train(net, lr, batch_size, epochs=100, record_term=100, source='simulate'):
    # train_iter, test_iter = mtrain.load_data(batch_size=batch_size)
    train_iter, test_iter = mtrain.load_data(source=source, batch_size=batch_size)

    net = net.to(device=mtrain.try_gpu())
    # 展示数据所在设备
    print('设备:' + str(next(net.parameters()).device))
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    '''
        fp_AlexNet.params
        fp_ResNet.params
    '''
    filename = 'fp_' + source + '_' + type(net).__qualname__ + '.params'

    mtrain.train(net, train_iter, test_iter, loss, epochs, trainer,
                 model_file=filename, record_term=record_term)


if __name__ == '__main__':
    # net = fpcnn.mLeNet()
    # net = fpcnn.mMultilayer(_in=900)
    # net = fpcnn.mResNet(input_channels=1, num_channels=2)
    net = fpcnn.WiGPTNet()
    fp_train(net, 0.3, 32, epochs=5000, record_term=100, source='wi')
