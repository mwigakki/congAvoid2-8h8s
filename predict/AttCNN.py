
import torch
import torch.nn as nn

# 注意力层  用于计算注意力权重并应用于输入特征图。
'''
这里给出的是一个使用PyTorch实现的带注意力机制的CNN模型，其中包含了一个SelfAttention类和一个AttCNN类。
注意力机制可以帮助模型更好地关注输入图像中的重要部分
在SelfAttention类中，通过三个卷积层分别计算输入特征图的query、key和value，
然后将query、key和value通过矩阵乘法计算得到注意力权重，最后将权重应用到value上得到输出。
具体来说，首先对query、key和value进行形状的变换，然后通过矩阵乘法计算得到注意力权重，最后将权重与value相乘得到输出。

在AttCNN类中，借助SelfAttention类实现了注意力机制，具体来说，在前向传播过程中，
首先将输入特征图通过一个卷积层得到中间特征图，然后使用SelfAttention类计算注意力权重并将其应用到中间特征图上得到输出特征图，
最后通过一个卷积层将输出特征图转换为最终输出。同时，可以看到这里还使用了Batch Normalization和ReLU激活函数。
'''
class SelfAttention(nn.Module):
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Input shape: (batch_size, in_channels, 8, 8)
        batch_size, _, h, w = x.size()

        # Compute query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for matrix multiplication
        q = q.view(batch_size, -1, h * w).permute(0, 2, 1)  # (batch_size, h * w, in_channels)
        k = k.view(batch_size, -1, h * w)  # (batch_size, in_channels, h * w)
        v = v.view(batch_size, -1, h * w)  # (batch_size, in_channels, h * w)

        # Compute attention weights
        attn_weights = torch.matmul(q, k)  # (batch_size, h * w, h * w)
        attn_weights = self.softmax(attn_weights)

        # Compute output
        out = torch.matmul(v, attn_weights.permute(0, 2, 1))  # (batch_size, in_channels, h * w)
        out = out.view(batch_size, -1, h, w)  # (batch_size, in_channels, 8, 8)

        out = self.softmax(out + x)
        return out


class AttCNN(nn.Module):
    def __init__(self, seq_length=10, out_channels=1, num_filters=64, s2h = True):
        super(AttCNN, self).__init__()
        self.seq_length = seq_length
        self.conv1 = nn.Conv2d(seq_length, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.attention = SelfAttention(num_filters)
        self.conv2 = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1)
        topo = [[0,0,1,0,0,0,0,1],
                [0,0,0,1,1,0,1,0],
                [1,0,0,1,0,1,0,1],
                [0,1,1,0,1,0,1,0],
                [0,1,0,1,0,1,1,0],
                [0,0,1,0,1,0,0,1],
                [0,1,0,1,1,0,0,1],
                [1,0,1,0,0,1,1,0]
                ]
        self.topo_tensor = torch.tensor(topo).float()
        self.s2h = s2h
    # 学习率设为 lr = 0.00001 为佳        
        
    def forward(self, x, adj):
        # Input shape: (batch_size, seq_length, 8, 8)

        x = self.conv1(x)  # (batch_size, num_filters, 8, 8)
        x = self.bn1(x)     # 这里使用bn比不适用bn效果好很多！
        x = self.relu(x)

        x = self.attention(x)  # (batch_size, num_filters, 8, 8)

        x = self.conv2(x)  # (batch_size, out_channels, 8, 8)
        
        if self.s2h:
            return x.squeeze(dim=1)
        else:
            return x.squeeze(dim=1) * self.topo_tensor    # 乘以拓扑，让不该出现值的位置不要出现值 
