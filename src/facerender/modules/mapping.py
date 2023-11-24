import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
这段代码定义了一个名为 MappingNet 的神经网络模型，
该模型用于将3D Morphable Model (3DMM) 的系数映射到头部姿态和表情的预测中。
"""


class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer, num_kp, num_bins):
        super( MappingNet, self).__init__()
        """
        coeff_nc：3DMM 系数的数量。
        descriptor_nc：描述符的数量，也是输出的通道数。
        layer：神经网络中卷积层的数量。
        num_kp：关键点的数量。
        num_bins：用于头部姿态预测的方向角的数量。
        """

        # 该模型的第一层是一个包含一个卷积层的序列，该卷积层将输入的 3DMM 系数映射到描述符空间。
        # LeakyReLU 是非线性激活函数，用于引入一些非线性特性。
        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        # 该循环创建了一系列卷积层。每一层都包含 LeakyReLU 激活函数和一个带有 3 持续膨胀的卷积层。
        # 这些层用于逐步提取输入的更高级特征。
        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        # 自适应平均池化层，将特征图的长度降为1。然后定义了 output_nc，表示输出的通道数，即描述符的数量。
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc
        # 接下来是一组全连接层，将提取的特征映射到头部姿态（yaw、pitch、roll）、平移（t）和表情（exp）的预测。
        self.fc_roll = nn.Linear(descriptor_nc, num_bins)
        self.fc_pitch = nn.Linear(descriptor_nc, num_bins)
        self.fc_yaw = nn.Linear(descriptor_nc, num_bins)
        self.fc_t = nn.Linear(descriptor_nc, 3)
        self.fc_exp = nn.Linear(descriptor_nc, 3*num_kp)

    # 最后定义了前向传播的方法。在该方法中，输入通过第一层卷积层，然后通过循环中的一系列卷积层，
    # 最终通过全连接层得到头部姿态和表情的预测。函数返回一个包含这些预测的字典。
    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        #print('out:', out.shape)

        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp} 