from pylayers.antprop.coverage import *
from matplotlib.pyplot import *
import random

nx = 0
ny = 0
lx = 1
ly = 1
xmin = 0
xmax = 0
ymin = 0
ymax = 0
ap_num = 0
rp_step = 1.5

error_bottom = -0.5
error_top = 0.5


# 模拟数据的每隔两米获取一个rp点
# 有a个ap点，范围是长nx个点，高ny个点，每个点采集20遍，需要得结果矩阵是(nx*ny,20,a+2)，即每个点，20组结果，每个结果7个元素
# def batch_simulate(batch=20):
#     # 先模拟一次，得到shape
#     rssi_once, label_once = simulate(is_show=False)
#     point_num = rssi_once.shape[0]
#
#     simulate_data = np.zeros((point_num, batch, ap_num + 2))
#
#     # 多次模拟获得批次数据
#     for i in range(batch):
#         show_pic = True if i == 0 else False
#         # dataset的shape为(nx*ny,a+2)
#         rssi, label = simulate(is_show=show_pic)
#
#         for j in range(point_num):
#             simulate_data[j, i, :] = np.concatenate((rssi, label), axis=1)[j]
#
#     # 分离为数据和标签
#     sample_rssi = np.around(np.array(simulate_data)[:, :, 0:-2], decimals=2)
#     sample_label = np.around(np.mean(np.array(simulate_data)[:, :, -2:], axis=1), decimals=2)
#
#     return sample_rssi, sample_label

# batch_simulate_copy_and_rand
def batch_simulate(batch=20):
    train_rssi, train_label, test_rssi, test_label = simulate(is_show=True)
    train_point_num = train_rssi.shape[0]
    test_point_num = test_rssi.shape[0]

    # 复制扩展rssi
    train_rssi = np.tile(train_rssi, (1, batch)).reshape(train_point_num, batch, ap_num)
    test_rssi = np.tile(test_rssi, (1, batch)).reshape(test_point_num, batch, ap_num)

    # 为每组数据产生随机误差
    error = (error_top - error_bottom) * np.random.random(size=(train_point_num, batch, ap_num)) + error_bottom

    train_rssi = train_rssi + error

    return train_rssi, train_label, test_rssi, test_label


# f:frequency index
def buil_data(C: Coverage, f=0):
    print("--------------------------------rssi evaluation--------------------------------")

    # --------- 暂时只使用水平数据Vp ----------
    # Vo = C.CmWo
    Vp = C.CmWp

    # 选择指定频率的V
    Vp = Vp[f, :, :]

    # Uo = Vo.reshape((C.nx, C.ny)).T
    # Uo = 10 * np.log10(Uo)
    Up = Vp.reshape((C.nx, C.ny, len(C.dap))).T
    Up = 10 * np.log10(Up)
    # U.shape=(a.num,ny,nx)
    U = Up
    # print(U)

    coord = np.empty(U[0].shape, dtype=tuple)
    for j in range(len(coord)):
        for i in range(len(coord[j])):
            coord[j][i] = (xmin + (i + 0.5) * lx / nx, ymin + (j + 0.5) * ly / ny)

    # print(coord)

    # 重组U和pos构成数据集
    dataset = rebuild_signal_coord(U, coord)
    np.set_printoptions(precision=4, suppress=True)

    return dataset


def rebuild_signal_coord(signal, coord):
    total_points = ny * nx
    # 数据集 ny*nx为所有点的信号值 ap_num+2为ap数+2位坐标值
    dataset = np.empty((total_points, ap_num + 2))

    for j in range(ny):
        for i in range(nx):
            v = signal[:, j, i]
            x, y = coord[j, i]
            dataset[j * nx + i] = np.concatenate((v, [x, y]))

    return dataset


def simulate(is_show=True):
    '''
        lib-version:
        numpy==1.23
        pandas==1.5
        networkx==1.7
        matplotlib==2.0.0
        seaborn==0.9.0
        simpy
        numba
        pip install descarteslabs triangle meshpy  osmapi simplejson pdbpp
        mayavi
        basemap

        ini/coverag.ini
        stru/lay/TA-office.lay 修改最后floor和ceil为zfloor和zceil
    '''

    # boundary = [20,0,30,20] (xmin,ymin,xmax,ymax)
    """
        ap参数:
        wstd:无线协议，默认ieee80211b
        p:ap位置坐标,(x,y,z)
        PtdBm:传输功率，默认0，表示标准功率
        chan:传输通道
        on:正常开启
        ant:Gauss高斯天线，Omni全向天线
        phideg:天线角度

        13维的原因：13种频率，[2.412 2.417 2.422 2.427 2.432 2.437 2.442 2.447 2.452 2.457 2.462 2.467, 2.472]

    """
    '''
        C对象下属性：
        self.ptdbm 发射功率
        self.freeScope 自由路径传播
        self.Lwo 路径损耗

        CmW : Received Power coverage in mW
        TODO : tgain in o and p polarization
        self.CmWo = 10**(self.ptdbm[np.newaxis,...]/10.)*self.Lwo*self.freespace*self.tgain
        self.CmWp = 10**(self.ptdbm[np.newaxis,...]/10.)*self.Lwp*self.freespace*self.tgain
    '''
    C = Coverage('coverage.ini')
    # C.dap[1]['on'] = False
    # C.dap[2]['on'] = False
    # C.dap[3]['on'] = False
    # C.dap[4]['on'] = False

    # 从左下角开始 向上为第一行的数据，逐渐向右
    C.cover()
    global nx, ny, ap_num
    nx, ny, ap_num = C.nx, C.ny, len(C.dap)
    global xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = C.L.ax
    global lx, ly
    lx = xmax - xmin
    ly = ymax - ymin

    if is_show:
        # f=频率索引，db=True使用dB，polar='p'信号轴水平
        f_all, a_all = C.show(fig=figure(figsize=(10, 5)), polar='p')
        a_all.axis('on')
        f_all.show()

    data = np.around(buil_data(C), decimals=2)

    # 获取训练数据：从模拟点采集部分位置，采集间隔(米) * 每米的grip点数
    step = max(1, int(round(rp_step * np.min((ny / ly, nx / lx)), 0)))

    sample_data = []
    for j in range(0, ny, step):
        for i in range(0, nx, step):
            sample_data.append(data[j * nx + i])

    sample_data = np.array(sample_data)

    train_rssi = sample_data[:, 0:-2]
    train_label = sample_data[:, -2:]

    # 获取测试数据：随机从data中选择，数量为训练数据的1/4
    test_num = train_rssi.shape[0] // 4
    test_index = np.random.choice(train_rssi.shape[0], size=test_num, replace=False)
    test_data = data[test_index, :]
    test_rssi = test_data[:, 0:-2]
    test_label = test_data[:, -2:]

    return train_rssi, train_label, test_rssi, test_label


def random_generate_ap(ap_num, xmin=-10, ymin=-5, xmax=50, ymax=20, high=1.2):
    left_bottom = (xmin, ymin)
    right_top = (xmax, ymax)

    # 生成30个随机坐标点
    random_points = []
    for _ in range(ap_num):
        # 在指定的x和y范围内生成包含一位小数的随机数
        x = round(random.uniform(left_bottom[0], right_top[0]), 1)
        y = round(random.uniform(left_bottom[1], right_top[1]), 1)
        z = round(random.uniform(0.5, 1.4), 1)
        # 将生成的坐标添加到列表中
        random_points.append((x, y, z))

    # 初始字典
    base_dict = {
        'name': 'a1',
        'wstd': 'ieee80211b',
        'p': (1, 12, 1.2),
        'PtdBm': 0,
        'chan': [11],
        'on': True
    }

    for i in range(len(random_points)):
        point = random_points[i]
        base_dict['p'] = point
        base_dict['name'] = 'a' + str(i)
        print(str(i) + ' = ' + str(base_dict))


if __name__ == '__main__':
    # L = Layout("TA-Office.lay")
    # f, a = L.showG()
    # f.show()
    # print(L.ax)
    # cover_simulate()
    # batch_simulate()
    # batch_simulate()
    # batch_simulate()
    # simulate()
    random_generate_ap(30)
