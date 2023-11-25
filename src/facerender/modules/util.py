from torch import nn

import torch.nn.functional as F
import torch

from src.facerender.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from src.facerender.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

import torch.nn.utils.spectral_norm as spectral_norm

"""
这段代码定义了一个用于音频到图像合成的神经网络架构。
"""


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    该函数将关键点转换为高斯样式的表示，其中高斯分布的均值由关键点位置确定。
    使用坐标网格生成关键点的位置，然后计算高斯分布。
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


# make_coordinate_grid_2d 和 make_coordinate_grid 函数:
# 创建2D和3D坐标的网格，用于生成坐标信息。
def make_coordinate_grid_2d(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def make_coordinate_grid(spatial_size, type):
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
   
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed

# ResBottleneck、ResBlock2d、ResBlock3d：分别是残差块的不同变体，用于处理2D和3D数据。
class ResBottleneck(nn.Module):
    """
    该类表示瓶颈结构的残差块，通常用于处理维度较大的特征图。
    包含三个卷积层，其中第一个是1x1卷积，降低输入特征图的维度（减小通道数），第二个是3x3卷积，最后一个是1x1卷积，升高输出特征图的维度。
    使用批归一化和ReLU激活函数。
    """
    def __init__(self, in_features, stride):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features, kernel_size=1)
        self.norm1 = BatchNorm2d(in_features//4, affine=True)
        self.norm2 = BatchNorm2d(in_features//4, affine=True)
        self.norm3 = BatchNorm2d(in_features, affine=True)

        self.stride = stride
        if self.stride != 1:
            self.skip = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=stride)
            self.norm4 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.stride != 1:
            x = self.skip(x)
            x = self.norm4(x)
        out += x
        out = F.relu(out)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    Res块，保留空间分辨率。
    通过两个3x3卷积层、批归一化和激活函数的处理，最后的输出是与输入相同空间分辨率的特征图
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    似于 ResBlock2d，通过两个3D卷积层、批归一化和激活函数的处理，最后的输出是与输入相同空间分辨率的3D特征图。
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
     用于解码器（decoder）的3D上采样块。这个块的目的是通过上采样操作增加输入的体积（或深度）。
     UpBlock3d 通过上采样操作实现了特征图的扩张，这对于解码器的任务（例如生成图像等）是非常重要的。
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        # out = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')
        """
        forward 方法：在正向传播中，首先通过使用F.interpolate函数进行上采样，
        这里的scale_factor=(1, 2, 2)表示在深度方向（第一个维度）上进行两倍的上采样。
        然后，上采样后的结果经过3D卷积、批归一化和ReLU激活函数的处理。最后的输出是经过这一系列操作后的3D特征图。
        """
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out



class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    DownBlock2d 的作用是通过卷积和池化操作减小输入特征图的空间分辨率，这有助于在编码器中提取和学习更高级别的特征。
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=(1, 2, 2))
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    SameBlock2d 主要用于在神经网络中执行一些基本的卷积、归一化和激活操作，同时保留输入特征图的空间分辨率。
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        """
        in_features：输入特征图的通道数。
        out_features：输出特征图的通道数。
        groups：卷积操作的组数，控制卷积的连接方式。
        kernel_size：卷积核的大小。
        padding：卷积操作的填充大小。
        lrelu：一个布尔值，指定是否使用 LeakyReLU 激活函数。如果为 True，
        则使用 LeakyReLU；如果为 False，则使用 ReLU。
        """
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


"""
Encoder、Decoder、Hourglass、KPHourglass 类:
实现了图像合成的Hourglass架构的各个组件。
KPHourglass 类是Hourglass架构的变体，其中包括关键点变换。
"""

class Encoder(nn.Module):
    """
    Hourglass Encoder
    该模型是一个 Hourglass 编码器。
    编码器负责对输入特征图进行下采样，以提取高层次的特征表示。
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()
        """
        block_expansion: 块的扩展系数，指定了网络中特征图通道数的增长率。即，每个下采样块中特征通道的增长倍数。
        in_features：输入特征图的通道数。
        num_blocks：编码器中下采样块的数量，默认为 3。
        max_features：最大特征通道数，默认为 256。
        
        创建一个包含多个DownBlock3d模块的列表，这些模块用于下采样输入特征图。
        下采样块的数量由num_blocks确定，每个块都根据其相对位置决定输入通道数和输出通道数。
        将这些下采样块存储在down_blocks中。
        """
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        """
        参数：
        x:输入特征图
        接受输入特征图x。
        通过迭代调用DownBlock3d模块，依次进行下采样操作，并将每一步的输出存储在列表outs中。
        返回包含所有下采样层输出的列表。
        """
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()
        """
        block_expansion：块的扩展系数，指定了网络中特征图通道数的增长率。即，每个下采样块中特征通道的增长倍数。
        in_features：输入特征图的通道数。
        num_blocks：解码器中上采样块的数量，默认为3。每个块通过上采样操作增加特征图的空间分辨率。
        max_features：最大特征通道数，默认为256。用于限制特征通道数的上限。
        
        步骤:
        """
        # 创建一个包含多个UpBlock3d模块的列表，这些模块用于上采样输入特征图.
        up_blocks = []
        # 上采样块的数量由num_blocks确定，每个块都根据其相对位置决定输入通道数和输出通道数。这里使用了[::-1]来反转块的顺序，
        # 以便从最后一层开始上采样。 将这些上采样块存储在up_blocks中。
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)

        # self.out_filters = block_expansion
        #  定义self.out_filters，表示解码器的输出特征通道数，计算方式为block_expansion + in_features。
        self.out_filters = block_expansion + in_features
        # 创建一个3D卷积层self.conv，用于整合解码器的输出。
        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        # 创建一个3D批量归一化层self.norm，用于规范化卷积层的输出。
        self.norm = BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        """
        接受一个包含编码器各层输出的列表x。
        从x中弹出最后一层的输出，作为初始输入out。(解码器的最后一层)
        通过迭代调用UpBlock3d模块，逐层进行上采样操作。在每一步上采样后，将来自编码器相应层的输出（skip）与当前解码器层的输出（out）连接在一起。
        最后一层上采样完成后，将输出传递给3D卷积层、批量归一化层和ReLU激活函数。
        返回最终的解码器输出。

        """
        out = x.pop()
        # for up_block in self.up_blocks[:-1]:
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        # out = self.up_blocks[-1](out)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    表示一个Hourglass架构的神经网络
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        """
        Encoder和Decoder是两个子模块，分别表示Hourglass模型的编码器和解码器。
        编码器负责将输入特征映射到一个高层次的表示，而解码器则通过上采样操作将这个表示还原为输入的维度。
        block_expansion是块的扩展系数，特征通道的增长倍数
        in_features是输入特征的通道数，
        num_blocks是Hourglass中的块数量，
        max_features是最大特征通道数。
        """
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class KPHourglass(nn.Module):
    """
    Hourglass architecture.
    表示一个带有自定义结构的Hourglass架构的神经网络。与标准Hourglass不同，这里包含了一些特定的结构，
    例如自定义的DownBlock2d和UpBlock3d，以及一个包含卷积层和维度调整的过程。
    """ 

    def __init__(self, block_expansion, in_features, reshape_features, reshape_depth, num_blocks=3, max_features=256):
        """
        :param block_expansion:这是一个用于 Hourglass 模型的块扩展系数，它指定了网络中特征图通道数的增长率。
        :param in_features:这是输入特征图的通道数。
        :param reshape_features:这是一个用于重塑的特征通道数。
        :param reshape_depth:这是重塑的深度。
        :param num_blocks:这是 Hourglass 模型中下采样块的数量，默认为 3。
        :param max_features:这是最大特征通道数，默认为 256。
        """
        super(KPHourglass, self).__init__()
        # 定义下采样（DownBlock2d）阶段
        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.down_blocks.add_module('down'+ str(i), DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                                                   min(max_features, block_expansion * (2 ** (i + 1))),
                                                                   kernel_size=3, padding=1))
        # 定义一个卷积层，用于将下采样阶段的特征映射到指定的通道数
        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=reshape_features, kernel_size=1)
        # 定义上采样（UpBlock3d）阶段
        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_filters = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module('up'+ str(i), UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))
        # 定义一些额外的属性，如维度重塑深度和最终输出特征通道数
        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def forward(self, x):
        # 下采样阶段
        out = self.down_blocks(x)
        # 卷积层，将特征映射到指定通道数
        out = self.conv(out)
        # 调整维度，将通道数分为多个部分
        bs, c, h, w = out.shape
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        # 上采样阶段
        out = self.up_blocks(out)

        return out
        


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    这段代码实现了二维的抗锯齿插值（Anti-Aliasing Interpolation）操作，主要用于下采样时更好地保留输入信号的细节。
    """
    def __init__(self, channels, scale):
        """
        :param channels:表示输入张量的通道数。
        :param scale:表示下采样的比例，即输出相对于输入的尺寸缩小的倍数。
        """
        super(AntiAliasInterpolation2d, self).__init__()
        # 计算高斯核的大小和标准差
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        # 构建高斯核
        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        # 确保高斯核中的值的总和等于1
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        # 将高斯核转换为深度卷积的权重
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        # 将高斯核作为模块的参数进行注册,"这里，高斯核被注册为缓冲区，表明它在模型训练的过程中不会被更新。"
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        """
        如果 scale 等于 1.0，说明不进行下采样，直接返回输入 input。
        否则，对输入进行了边缘填充，然后使用深度卷积（depthwise convolution）进行卷积操作。最后，对输出进行了下采样，每隔一定间隔取样。
        """
        if self.scale == 1.0:
            return input
        # 对输入进行边缘填充
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        # 使用深度卷积进行卷积操作
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        # 下采样，每隔一定间隔取样
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class SPADE(nn.Module):
    """
    SPADE 模块允许根据语义分割图（segmap）中的信息对输入进行归一化。
    """
    def __init__(self, norm_nc, label_nc):
        """

        :param norm_nc: 输入通道的数量，用于实例归一化层。
        :param label_nc:语义分割图的通道数，用于生成中间表示的多层感知机（MLP）。
        """
        super().__init__()
        # param_free_norm: 这是一个无参数的实例归一化层（Instance Normalization），
        # 对输入 x 进行归一化，affine=False 表示不使用可学习的缩放和平移参数。
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        # mlp_shared: 这是一个共享的多层感知机（MLP），通过卷积和激活函数处理，以生成用于调整归一化的参数的中间表示。
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        # mlp_gamma 和 mlp_beta: 这两个分支分别使用卷积层来生成缩放参数 gamma 和平移参数 beta。
        # 这两个参数是根据语义分割图调整输入 x 的归一化参数的关键。
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        """
        该方法接收输入 x 和语义分割图 segmap。首先，对输入 x 进行无参数的实例归一化，
        然后将语义分割图进行插值以与输入 x 相匹配。接下来，通过 MLP 处理插值后的语义分割图，得到中间表示。
        最后，使用中间表示计算缩放参数 gamma 和平移参数 beta，然后根据这两个参数对输入 x 进行条件归一化。
        :param x:归一化的输入图像或特征图
        :param segmap:语义分割图
        :return:out 输出
        """
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    

class SPADEResnetBlock(nn.Module):
    """
    这个模块的目的是在 ResNet 结构中引入 SPADE 归一化，以充分利用语义分割信息来调整生成图像或特征图的分布。
    """
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        """
        :param fin:输入特征的通道数。
        :param fout:输出特征的通道数。
        :param norm_G:用于归一化的标准化方法
        :param label_nc:语义分割标签的通道数。
        :param use_se: 是否使用 Squeeze-and-Excitation (SE) 模块，默认为 False。
        :param dilation:卷积层的膨胀率，默认为 1。
        """
        super().__init__()
        # Attributes
        # 表示是否使用了学习的快捷连接（学习的快捷连接用于调整输入和输出通道数不一致的情况）。
        self.learned_shortcut = (fin != fout)
        #  输入和输出通道数的较小值，用于构建卷积层。
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        # 如果存在学习的快捷连接，这是一个用于调整输入和输出通道数的 1x1 卷积层。
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        # 如果 norm_G 中包含 'spectral'，则对卷积层应用谱范数。
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        """
        接受输入 x 和语义分割图 seg1。
        计算快捷连接 x_s。
        通过卷积和 SPADE 归一化操作处理输入 x，并通过两个卷积层 self.conv_0 和 self.conv_1。
        将快捷连接和处理后的输入相加，得到输出 out。

        :param x: 输入 x
        :param seg1:语义分割图 seg1
        :return:输出 out

        """

        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        """
        如果存在学习的快捷连接，则对输入 x 进行卷积和 SPADE 归一化操作。
        :param x: 输入 x
        :param seg1: 语义分割图 seg1
        :return: 输出 out
        """
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        """
        使用 leaky ReLU 作为激活函数。
        :param x:
        :return:
        """
        return F.leaky_relu(x, 2e-1)

class audio2image(nn.Module):
    """
    该模型的目标是将源图像和目标音频的信息结合，通过头部姿态估计和关键点变换，生成一个合成的图像。
    """
    def __init__(self, generator, kp_extractor, he_estimator_video, he_estimator_audio, train_params):
        """
        :param generator:生成器模型，用于将源图像和目标音频生成合成图像。
        :param kp_extractor:关键点提取器模型，用于从源图像中提取关键点。
        :param he_estimator_video:源图像头部姿态估计器模型
        :param he_estimator_audio:目标音频头部姿态估计器模型
        :param train_params: 训练参数，可能包括学习率、优化器等。
        """
        super().__init__()
        # Attributes
        self.generator = generator
        self.kp_extractor = kp_extractor
        self.he_estimator_video = he_estimator_video
        self.he_estimator_audio = he_estimator_audio
        self.train_params = train_params

    def headpose_pred_to_degree(self, pred):
        """
        将头部姿态预测值转换为角度表示
        :param pred:
        :return:
        """
        device = pred.device
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred)
        degree = torch.sum(pred*idx_tensor, 1) * 3 - 99

        return degree
    
    def get_rotation_matrix(self, yaw, pitch, roll):
        """
        根据给定的偏航 (yaw)、俯仰 (pitch) 和横滚 (roll) 角度，计算旋转矩阵。
        :param yaw:
        :param pitch:
        :param roll:
        :return:
        """
        yaw = yaw / 180 * 3.14
        pitch = pitch / 180 * 3.14
        roll = roll / 180 * 3.14

        roll = roll.unsqueeze(1)
        pitch = pitch.unsqueeze(1)
        yaw = yaw.unsqueeze(1)

        roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
        roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

        pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
        pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

        yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
        yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

        rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

        return rot_mat

    def keypoint_transformation(self, kp_canonical, he):
        """
        将规范化的关键点 (kp_canonical) 根据头部姿态参数进行变换，包括旋转、平移和表情偏差的添加。
        :param kp_canonical:
        :param he:
        :return:
        """
        kp = kp_canonical['value']    # (bs, k, 3)
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        t, exp = he['t'], he['exp']
    
        yaw = self.headpose_pred_to_degree(yaw)
        pitch = self.headpose_pred_to_degree(pitch)
        roll = self.headpose_pred_to_degree(roll)

        rot_mat = self.get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
        # keypoint rotation
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    

        # keypoint translation
        t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
        kp_t = kp_rotated + t

        # add expression deviation 
        exp = exp.view(exp.shape[0], -1, 3)
        kp_transformed = kp_t + exp

        return {'value': kp_transformed}

    def forward(self, source_image, target_audio):
        """
        通过头部姿态估计器获取源图像和目标音频的头部姿态参数。
        使用关键点提取器提取源图像的规范化关键点 (kp_canonical)。
        将源图像的规范化关键点根据源图像的头部姿态参数进行变换 (kp_source)。
        将源图像的规范化关键点根据目标音频的头部姿态参数进行变换 (kp_transformed_generated)。
        使用生成器将源图像和变换后的关键点生成合成图像 (generated)。
        :param source_image: 源图像
        :param target_audio: 目标音频
        :return: 合成图像
        """
        pose_source = self.he_estimator_video(source_image)
        pose_generated = self.he_estimator_audio(target_audio)
        kp_canonical = self.kp_extractor(source_image)
        kp_source = self.keypoint_transformation(kp_canonical, pose_source)
        kp_transformed_generated = self.keypoint_transformation(kp_canonical, pose_generated)
        generated = self.generator(source_image, kp_source=kp_source, kp_driving=kp_transformed_generated)
        return generated