#!coding=utf-8
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 表示对继承自父类属性进行初始化
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        tensor.contiguous()会返回有连续内存的相同张量
        有些tensor并不是占用一整块内存，而是由不同的数据块组成
        tensor的view()操作依赖于内存是整块的，这时只需要执行
        contiguous()函数，就是把tensor变成在内存中连续分布的形式
        本函数主要是增加padding方式对卷积后的张量做切边而实现因果卷积
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数,输入层通道为1，隐含层是25。
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)  # *作用是将输入迭代器拆成一个个元素

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

class TCN1(nn.Module):
    def __init__(self, num_hiddens=[32,8], s2h = True):  # 这里的隐层最后一层必须是8
        super().__init__() 
        # 每个点的流量值只被自己影响
        # self.tp1 = [[0],[1],[2],[3],[4],[5],[6],[7]]    
        # 根据拓扑，s1到所有终端的流量被s1，s3，s8所影响，因此s1hx的预测结果由s1hx,s3hx,s8hx 这24个点的流量值计算得到, 所以num_inputs[0]是24。其他的同理。
        self.tp2 = [[0,2,7],[1,3,4,6],[0,2,3,5,7],[1,2,3,4,6],[1,3,4,5,6],[2,4,5,7],[1,3,4,6,7],[0,2,5,6,7]]
        # num_inputs1 = [len(item)*8 for item in self.tp1]    # [8, 8, 8, 8, 8, 8, 8, 8]   
        num_inputs2 = [len(item)*8 for item in self.tp2]    # [24, 32, 40, 40, 40, 32, 40, 40] 
        # 创建8个tcn模型，8个模型的输入都是不一样的， 
        self.tcn_models = [TemporalConvNet(num_inputs=num_inputs2[i], num_channels=num_hiddens, kernel_size=2, dropout=0.2) for i in range(8)]
       
        

    def forward(self, x, adj):  # 直接把输入拉直进行计算
        # 默认输入数据 （bs, seq_len, h, w）即（bs, 32, 8, 8）需要修改为 (bs, h*w, seq_len)
        bs, seq_len, h, w = x.shape
        output = [] 
        for i in range(8):
            selected_data = x[:, :, self.tp2[i], :].view(bs, seq_len, len(self.tp2[i]) * w).permute(0, 2, 1)    # selected_data:  torch.Size([32, 8*邻接数, 10])
            output.append(self.tcn_models[i](selected_data)[:, :, -1])  # output的尺寸 （bs, num_hiddens[-1], seq_len）， 即 （bs, 8, 10），直接把最后一维取-1，得到（bs, 8）
        y = torch.stack(output, -1)  # 把8个output的输出堆叠到最后一维，torch.Size([32, 8, 8])
        return y 
 