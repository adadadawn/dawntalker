from torch import nn
import torch.nn.functional as F
import torch
from src.facerender.modules.util import Hourglass, make_coordinate_grid, kp2gaussian

from src.facerender.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    这段代码实现了一个称为 Dense Motion Network 的神经网络模块，用于从稀疏的运动表示（由 kp_source 和 kp_driving 提供）中预测密集的运动。
    主要用于人脸渲染或视频动作迁移等任务中。
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        """
        参数：
        block_expansion：指定 Hourglass 模块中每个残差块（residual block）的通道扩展系数。
        num_blocks：指定 Hourglass 模块中堆叠的残差块的数量。
        max_features：指定 Hourglass 模块中最大的通道数。
        num_kp：表示关键点（keypoints）的数量。这是模型中要处理的关键点的个数
        feature_channel：表示输入特征的通道数
        reshape_depth：用于计算遮挡图的通道深度。
        estimate_occlusion_map:一个标志，表示是否估计遮挡图。如果设置为 True，则会添加用于估计遮挡图的卷积层，否则为 None
        """
        """
        初始化模块：
        self.hourglass：堆叠的 Hourglass 网络模块，用于处理输入特征。
        self.mask：卷积层，输出一个用于加权运动的遮罩。
        self.compress：卷积层，用于压缩输入特征。
        self.norm：批归一化层，对压缩后的特征进行归一化。
        self.occlusion：如果 estimate_occlusion_map 为 True，则为用于估计遮挡图的卷积层，否则为 None。
        self.num_kp：记录了关键点的数量，以备后续使用。
        """
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(feature_channel+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp

# 利用稀疏运动表示，对输入特征进行变形。通过使用 F.grid_sample 函数，实现了在特征上的空间变形。
    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)
        
        # if 'jacobian' in kp_driving:
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)                  


        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)                #bs num_kp+1 d h w 3
        
        # sparse_motions = driving_to_source

        return sparse_motions

# 创建用于网络输入的热图表示。通过计算 kp_driving 和 kp_source 的高斯热图之差。
    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3) !!!!
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed


# 基于给定的关键点生成热图表示
    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        """
        feature：这是一个输入张量，表示某种特征。在代码中，使用了这个特征来获取其空间尺寸。
        kp_driving：这是表示运动驱动关键点的张量。关键点是在图像中标识出的特殊点，用于描述运动或形状变化。
        kp_source：这是表示源关键点的张量，通常是在静止图像中标识的关键点。
        """
        spatial_size = feature.shape[3:]
        # kp2gaussian 该函数的作用是生成高斯热图，其中关键点的影响通过kp_variance进行调节。
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        # 这行生成了另一个高斯热图，基于kp_source关键点。
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
        # 这行计算两个高斯热图的差异，生成一个新的热图。
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        # 创建一个全零的张量，作为背景特征。
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        # 在第一个维度上连接零张量和之前计算的热图，将背景特征添加到热图中。
        heatmap = torch.cat([zeros, heatmap], dim=1)
        # 在第三个维度上添加一个维度，使得最终的形状为(bs, num_kp+1, 1, d, h, w)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        """
        将输入特征进行压缩、正则化，并构建了热图表示。
        使用 Hourglass 模块处理这些输入，输出一个遮罩 mask。
        遮罩 mask 通过 softmax 操作，用于加权运动。
        如果需要，通过卷积层 occlusion 计算遮挡图。
        """
        bs, _, d, h, w = feature.shape
        # 通过 self.compress 方法对输入特征进行卷积压缩，
        # 然后通过 self.norm 方法进行批归一化，最后应用了 ReLU 激活函数。
        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)


        #生成稀疏运动表示和变形特征。
        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        # 生成了热图表示，描述了运动驱动关键点和源关键点之间的关系。
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        # 这几行将热图表示和变形特征连接在一起，然后通过 Hourglass 模块处理，得到预测结果 prediction。
        input_ = torch.cat([heatmap, deformed_feature], dim=2)
        input_ = input_.view(bs, -1, d, h, w)
        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)
        prediction = self.hourglass(input_)

        # 这两行通过卷积层 self.mask 处理 Hourglass 模块的输出，然后通过 softmax 操作生成遮罩 mask。
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)

        # 到if之前的部分这对遮罩进行调整，然后利用遮罩对稀疏运动进行加权，计算最终的形变。
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        
        zeros_mask = torch.zeros_like(mask)   
        mask = torch.where(mask < 1e-3, zeros_mask, mask) 

        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation
        # 如果模型需要估计遮挡图（self.occlusion 不为 None），
        # 则对 Hourglass 模块的输出进行处理，计算遮挡图，并存储在输出字典中。
        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
