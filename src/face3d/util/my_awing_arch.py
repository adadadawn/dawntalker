# 这段代码实现了一个人脸关键点检测模型，使用了改进的 Hourglass 网络结构。
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_points(heatmaps):
    # change heatmaps to landmarks
    # 用于从关键点热图中计算出关键点的坐标。
    B, N, H, W = heatmaps.shape
    HW = H * W
    BN_range = np.arange(B * N)
    # 对每个热图，找到最大值对应的索引，即热图中最可能是关键点的位置。
    heatline = heatmaps.reshape(B, N, HW)
    indexes = np.argmax(heatline, axis=2)
    #将索引转换为二维坐标，得到每个关键点在热图上的位置。
    preds = np.stack((indexes % W, indexes // W), axis=2)
    preds = preds.astype(np.float, copy=False)
    
    # 对于每个关键点，考虑其周围像素的值，通过插值的方式微调关键点的位置，以更精确地定位关键点在图像中的位置。
    """
    具体地说，对于每个关键点，分别考虑其水平方向和垂直方向上相邻像素的值（x_up, x_down, y_up, y_down）。
    然后，通过计算这些相邻像素值的差异，并进行符号化和缩放，得到了一个微调的偏移向量 think_diff。
    最后，将这个微调的偏移向量应用到关键点坐标上，以更精确地定位关键点在图像中的位置。
    最后两行代码是为了将微调后的坐标值映射到图像范围内。
    """
    inr = indexes.ravel()

    heatline = heatline.reshape(B * N, HW)
    x_up = heatline[BN_range, inr + 1]
    x_down = heatline[BN_range, inr - 1]
    # y_up = heatline[BN_range, inr + W]

    if any((inr + W) >= 4096):
        y_up = heatline[BN_range, 4095]
    else:
        y_up = heatline[BN_range, inr + W]
    if any((inr - W) <= 0):
        y_down = heatline[BN_range, 0]
    else:
        y_down = heatline[BN_range, inr - W]

    think_diff = np.sign(np.stack((x_up - x_down, y_up - y_down), axis=1))
    think_diff *= .25

    preds += think_diff.reshape(B, N, 2)
    preds += .5
    return preds

"""
在输入张量中添加坐标信息。它的主要目的是为了在训练过程中向模型提供关于像素在图像中位置的额外信息，以增强模型对位置敏感的任务的性能。
"""
class AddCoordsTh(nn.Module):

    def __init__(self, x_dim=64, y_dim=64, with_r=False, with_boundary=False):
        # 图像的宽度（x_dim）、高度（y_dim）、是否包含径向坐标（with_r）、是否包含边界信息（with_boundary）
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]
        # 创建一个形状为 [1, self.y_dim, 1] 的张量 xx_ones，其中所有元素均为 1。
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32, device=input_tensor.device)
        xx_ones = xx_ones.unsqueeze(-1)
        # 代码使用 torch.arange 和矩阵乘法创建x和y坐标的二维网格。这些网格覆盖范围为[0, x_dim-1]和[0, y_dim-1]，分别表示x和y的坐标。
        xx_range = torch.arange(self.x_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)
        # 
        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32, device=input_tensor.device)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())  # 建两个表示 x 和 y 坐标的通道（xx_channel 和 yy_channel），并对它们进行一系列的处理，包括矩阵乘法、维度调整、归一化等。
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)   

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        # 如果 with_boundary 为真且存在热图（heatmap），则处理边界信息。
        # 通过阈值处理热图，创建仅在边界处为正的新坐标通道。这些通道分别为 xx_boundary_channel 和 yy_boundary_channel。
        # 将这两个通道转移到与输入张量相同的设备。
        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)
        if self.with_boundary and heatmap is not None:
            xx_boundary_channel = xx_boundary_channel.to(input_tensor.device)
            yy_boundary_channel = yy_boundary_channel.to(input_tensor.device)
    # 拼接坐标信息到输入张量：将坐标信息（xx_channel 和 yy_channel）和可能的径向坐标（rr）拼接到输入张量上。
    # 如果启用了边界信息，还将边界坐标信息拼接到输入张量上。
        
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and heatmap is not None:
            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)
        return ret


# 实现了坐标卷积（CoordConv），该方法是在输入张量中添加坐标信息，然后进行卷积操作。
class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper.
    "CoordConv layer as in the paper" 指的是在论文中描述的 CoordConv（Coordinate Convolution）层的实现。
    CoordConv 的主要思想是在卷积神经网络（CNN）的输入中引入额外的坐标信息，以帮助模型更好地理解输入数据的空间结构和位置关系。
    这是通过在输入数据中添加表示位置信息的坐标通道来实现的。这种做法有助于解决标准 CNN 在处理位置信息时的一些困难，尤其是对于不规则的输入。
    """

    def __init__(self, x_dim, y_dim, with_r, with_boundary, in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r, with_boundary=with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:  #first_one表示这是不是模型中的第一个 CoordConv 层
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        # 首先通过调用 AddCoordsTh 实例的 forward 方法，将坐标信息添加到输入张量中，得到增强后的张量 ret。
        
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False, dilation=1):
    '3x3 convolution with padding'
    # 创建一个 3x3 的卷积层
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        __init__ 方法用于初始化 BasicBlock 类的实例。
        inplanes: 输入通道数，即输入张量的通道数量。
        planes: 输出通道数，即卷积操作后得到的特征图的通道数量。
        stride: 步长，表示卷积核在输入张量上滑动的步幅，默认为 1。
        downsample: 下采样，用于匹配输入和输出的维度，如果需要降低空间维度，可以提供一个下采样操作，默认为 None。
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 一个由卷积操作组成的块
class ConvBlock(nn.Module):
   
    def __init__(self, in_planes, out_planes):
        
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), padding=1, dilation=1)
# 如果输入通道数 in_planes 不等于输出通道数 out_planes，则定义一个下采样分支。
# 下采样分支使用 nn.Sequential 定义，包括批标准化层、ReLU 激活函数以及 1x1 卷积层，将输入通道数降至 out_planes。
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        """
        forward 方法定义了 ConvBlock 的前向传播逻辑。
        首先将输入张量 x 赋值给 residual，作为残差连接的基准。
        out1: 对输入进行批标准化、ReLU 激活函数和第一个卷积操作。
        out2: 对 out1 进行批标准化、ReLU 激活函数和第二个卷积操作。
        out3: 对 out2 进行批标准化、ReLU 激活函数和第三个卷积操作。
        将 out1、out2 和 out3 沿着通道维度进行拼接，使用 torch.cat 函数。
        如果存在下采样分支，对 residual 进行下采样。
        将下采样结果与拼接后的特征图相加，实现了残差连接。
        返回最终的输出结果。
        
        """
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

# 这段代码定义了一个名为 HourGlass 的 PyTorch 模块，
# 表示一个由多个堆叠的 Hourglass 模块组成的结构。Hourglass 模块通常用于图像处理任务，如关键点检测或姿态估计。
class HourGlass(nn.Module):

    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        """
        num_modules: Hourglass 模块的数量，表示网络中有多少个堆叠的 Hourglass 模块。
        depth: Hourglass 模块的深度，表示每个 Hourglass 模块内部有多少层。
        num_features: 特征通道数，表示每个 Hourglass 模块的特征通道数量。
        first_one: 一个布尔值，表示是否是网络中的第一个 Hourglass 模块。
        self.coordconv: 创建了一个名为 coordconv 的 CoordConvTh 模块，用于在 Hourglass 模块中添加坐标信息。这是一个在前面提到的 CoordConvTh 类的实例化。
        self._generate_network(self.depth): 调用 _generate_network 方法来生成 Hourglass 模块内部的网络结构。
        """
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(
            x_dim=64,
            y_dim=64,
            with_r=True,
            with_boundary=True,
            in_channels=256,
            first_one=first_one,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        """
        _generate_network 方法用于在 Hourglass 模块内部生成指定深度的网络结构。
        创建了多个名为 'b1_' + str(level)、'b2_' + str(level)、'b2_plus_' + str(level) 和 'b3_' + str(level) 的 ConvBlock 模块，
        构建了 Hourglass 模块的内部结构。这些模块可能包含卷积层、批标准化层等操作。
        递归地调用 _generate_network 方法，构建 Hourglass 模块的金字塔状结构。
        """
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        """
        _forward 方法实现了 Hourglass 模块的前向传播逻辑。
        """
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        """
        接受输入张量 x 和热图 heatmap。
        调用 coordconv 模块，将坐标信息添加到输入张量中。
        调用 _forward 方法进行 Hourglass 模块的前向传播，返回 Hourglass 模块的输出和热图中的最后两个通道。
        """
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


# 它是一个具有多个 Hourglass 模块的网络，用于进行人脸关键点检测。
class FAN(nn.Module):

    def __init__(self, num_modules=1, end_relu=False, gray_scale=False, num_landmarks=68, device='cuda'):
        """
        初始化模型的参数，如 num_modules 表示 Hourglass 模块的数量，end_relu 表示最终输出是否应用 ReLU 激活，gray_scale 表示输入是否为灰度图像，
        num_landmarks 表示要检测的关键点数量，device 表示运行模型的设备。
        创建了模型的基础部分，包括一些卷积块 (CoordConvTh, ConvBlock) 和 Hourglass 模块。
        通过 add_module 方法将这些子模块添加到模型中。
        """
        super(FAN, self).__init__()
        self.device = device
        self.num_modules = num_modules
        self.gray_scale = gray_scale
        self.end_relu = end_relu
        self.num_landmarks = num_landmarks

        # Base part
        if self.gray_scale:
            self.conv1 = CoordConvTh(
                x_dim=256,
                y_dim=256,
                with_r=True,
                with_boundary=False,
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3)
        else:
            self.conv1 = CoordConvTh(
                x_dim=256,
                y_dim=256,
                with_r=True,
                with_boundary=False,
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            if hg_module == 0:
                first_one = True
            else:
                first_one = False
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256, first_one))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, num_landmarks + 1, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module),
                                nn.Conv2d(num_landmarks + 1, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """
        定义了模型的前向传播逻辑。
        输入首先通过卷积块和池化层进行处理。
        接着通过多个 Hourglass 模块进行处理，每个模块都生成一组关键点的预测（tmp_out）。
        输出包含了每个 Hourglass 模块生成的关键点预测，以及用于边界信息的通道（boundary_channels）。
        边界信息通常指示关键点位置的边缘或边界。
        """
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        # x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        boundary_channels = []
        tmp_out = None
        for i in range(self.num_modules):
            hg, boundary_channel = self._modules['m' + str(i)](previous, tmp_out)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out)  # HACK: Added relu
            outputs.append(tmp_out)
            boundary_channels.append(boundary_channel)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, boundary_channels

    def get_landmarks(self, img):
        """
        该方法接受一张图像，将其预处理成模型可以接受的格式，并调用模型的前向传播方法获取关键点的预测。
        预测的关键点坐标通过 calculate_points 函数计算，然后根据图像的原始尺寸进行调整。
        返回关键点的预测坐标。
        """
        H, W, _ = img.shape
        offset = W / 64, H / 64, 0, 0

        img = cv2.resize(img, (256, 256))
        inp = img[..., ::-1]
        inp = torch.from_numpy(np.ascontiguousarray(inp.transpose((2, 0, 1)))).float()
        inp = inp.to(self.device)
        inp.div_(255.0).unsqueeze_(0)

        outputs, _ = self.forward(inp)
        out = outputs[-1][:, :-1, :, :]
        heatmaps = out.detach().cpu().numpy()

        pred = calculate_points(heatmaps).reshape(-1, 2)

        pred *= offset[:2]
        pred += offset[-2:]

        return pred
