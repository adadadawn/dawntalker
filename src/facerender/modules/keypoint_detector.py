from torch import nn
import torch
import torch.nn.functional as F

from src.facerender.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from src.facerender.modules.util import KPHourglass, make_coordinate_grid, AntiAliasInterpolation2d, ResBottleneck


class KPDetector(nn.Module):
    """
    Detecting canonical keypoints. Return keypoint position and jacobian near each keypoint.
    KPDetector 是关键点检测器模型，用于检测图像中的关键点位置及其雅克比矩阵。
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, reshape_channel, reshape_depth,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False):
        super(KPDetector, self).__init__()
        """
        参数：
        block_expansion: 沙漏网络中每个块的特征通道扩张因子。
        feature_channel: 特征通道的维度。
        num_kp: 要检测的关键点数量。
        image_channel: 输入图像的通道数。
        max_features: 沙漏网络中特征通道的最大数量。
        reshape_channel: 沙漏网络中用于形状重塑的特征通道数。
        reshape_depth: 沙漏网络中形状重塑的深度。
        num_blocks: 沙漏网络中沙漏块的数量。
        temperature: softmax 操作中的温度参数。
        estimate_jacobian: 是否估计雅克比矩阵。
        scale_factor: 输入图像的降采样因子。
        single_jacobian_map: 是否只使用一个雅克比矩阵（对所有关键点共享一个）。
        """

        self.predictor = KPHourglass(block_expansion, in_features=image_channel,
                                     max_features=max_features,  reshape_features=reshape_channel, reshape_depth=reshape_depth, num_blocks=num_blocks)

        # self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=7, padding=3)
        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=3, padding=1)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            # self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=7, padding=3)
            self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=3, padding=1)
            '''
            initial as:
            [[1 0 0]
             [0 1 0]
             [0 0 1]]
            '''
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean from a heatmap
        从关键点热力图中提取均值的方法。返回一个字典，包含关键点均值的张量。
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'value': value}

        return kp

    def forward(self, x):
        """
        接收输入图像 x。
        如果 scale_factor 不等于 1，对输入图像进行降采样。
        通过沙漏网络 self.predictor 提取特征。
        通过 self.kp 卷积层生成关键点热力图，并进行 softmax 操作。
        调用 gaussian2kp 方法从热力图中提取关键点均值。
        如果估计雅克比矩阵，通过 self.jacobian 卷积层生成雅克比矩阵，并与热力图相乘，最终得到雅克比矩阵。
        返回一个包含关键点均值和雅克比矩阵（如果估计的话）的字典。
        (关键点的位置和雅克比矩阵对于生成逼真的渲染效果至关重要。)
        """
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 9, final_shape[2],
                                                final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 3, 3)
            out['jacobian'] = jacobian

        return out


class HEEstimator(nn.Module):
    """
    Estimating head pose and expression.
    HEEstimator 是用于估计头部姿态和表情的模型。
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, num_bins=66, estimate_jacobian=True):
        super(HEEstimator, self).__init__()
        """
        block_expansion: 一个整数，表示残差块的扩张系数。
        feature_channel: 一个整数，表示输入图像的通道数。
        num_kp: 一个整数，表示关键点的数量。
        image_channel: 一个整数，表示输入图像的通道数。
        max_features: 一个整数，表示最大的特征数。
        num_bins: 一个整数，表示角度估计中分组的数量。
        estimate_jacobian: 一个布尔值，表示是否要估计雅可比矩阵。
        """

        """
        定义了一系列卷积层 (nn.Conv2d)、批归一化层 (BatchNorm2d) 以及残差块 (ResBottleneck)。
        最后定义了几个全连接层 (nn.Linear) 用于输出头部姿态和表情的估计结果。
        """
        self.conv1 = nn.Conv2d(in_channels=image_channel, out_channels=block_expansion, kernel_size=7, padding=3, stride=2)
        self.norm1 = BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=block_expansion, out_channels=256, kernel_size=1)
        self.norm2 = BatchNorm2d(256, affine=True)

        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module('b1_'+ str(i), ResBottleneck(in_features=256, stride=1))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.norm3 = BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)

        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module('b3_'+ str(i), ResBottleneck(in_features=512, stride=1))

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.norm4 = BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)

        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module('b5_'+ str(i), ResBottleneck(in_features=1024, stride=1))

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.norm5 = BatchNorm2d(2048, affine=True)
        self.block6 = ResBottleneck(in_features=2048, stride=2)

        self.block7 = nn.Sequential()
        for i in range(2):
            self.block7.add_module('b7_'+ str(i), ResBottleneck(in_features=2048, stride=1))

        self.fc_roll = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)
        self.fc_yaw = nn.Linear(2048, num_bins)

        self.fc_t = nn.Linear(2048, 3)

        self.fc_exp = nn.Linear(2048, 3*num_kp)

    def forward(self, x):
        """
        参数:
        x: 输入图像。
        操作:
        对输入图像进行一系列的卷积、批归一化和残差块操作，形成特征表示。
        对特征进行自适应平均池化，将其形状调整为(batch_size, 2048, 1, 1)。
        将平坦的特征输入到全连接层中，得到头部姿态 (yaw, pitch, roll)、平移 (t) 和表情 (exp) 的估计结果。
        返回一个字典，包含估计的头部姿态和表情。

        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.block2(out)

        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.block4(out)

        out = self.block5(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = F.relu(out)
        out = self.block6(out)

        out = self.block7(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        yaw = self.fc_roll(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_yaw(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}

