import torch
from torch import nn
import torch.nn.functional as F
from src.facerender.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
from src.facerender.modules.dense_motion import DenseMotionNetwork


"""
OcclusionAwareGenerator: 这个生成器类似于NVIDIA架构，是一个比较传统的生成器，使用卷积和残差块来处理图像生成任务。
它主要关注密集运动估计和特征的空间变换。
OcclusionAwareSPADEGenerator: 这个生成器在架构上引入了SPADE（Spatially-Adaptive Denormalization）解码器，
用于更灵活地处理条件信息。SPADE是一种用于图像到图像的翻译任务的归一化方法，可以根据条件信息调整生成器中的每个层。
"""
class OcclusionAwareGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
     这个生成器遵循 NVIDIA 架构，该架构通常用于图像生成任务。
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()
        """
        参数：
        image_channel: 输入图像的通道数。
        feature_channel: 特征通道数，用于表示编码器和解码器中的特征图。
        num_kp: 关键点的数量。
        block_expansion: 每个块（Block）中特征通道的扩展系数。
        max_features: 特征通道的最大数量。
        num_down_blocks: 下采样块（DownBlock2d）的数量，用于编码器。
        reshape_channel: 3D特征表示中的通道数量。
        reshape_depth: 3D特征表示中的深度（即第三维的大小）。
        num_resblocks: 残差块的数量，用于编码器、3D特征表示和解码器。
        estimate_occlusion_map: 一个布尔值，指示是否估计遮挡图。
        dense_motion_params: 一个字典，包含 Dense Motion 网络的参数。如果为 None，则不使用 Dense Motion 网络。
        estimate_jacobian: 一个布尔值，指示是否估计雅克比矩阵。
        """
        """
        Dense Motion 网络初始化： 如果提供了 dense_motion_params，则初始化 DenseMotionNetwork 用于密集运动估计
        """
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.resblocks_2d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_2d.add_module('2dr' + str(i), ResBlock2d(out_features, kernel_size=3, padding=1))

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, image_channel, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel


# 这个函数接受一个输入张量 inp 和一个变形场 deformation。它使用网格采样 (F.grid_sample) 根据变形场调整输入张量的形状。
# 如果输入张量和变形场的尺寸不匹配，它会先进行插值。
    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)


    def forward(self, source_image, kp_driving, kp_source):
        """
        # 这是模型的前向传播函数，它执行以下主要步骤：
        # 编码（Downsampling）： 将源图像通过编码器的下采样部分
        # 3D 特征表示： 将下采样得到的特征表示为3D形式，并应用3D残差块。
        # 密集运动估计（如果启用）： 如果启用了密集运动估计，通过 Dense Motion 网络计算变形、遮挡和遮挡图。
        # 特征变换： 根据密集运动估计结果对3D特征进行变换，包括遮挡处理。
        # 解码（Upsampling）： 使用解码器的上采样部分逐步还原图像。
        # 最终输出： 最终的生成图像，经过 sigmoid 函数处理。
        # 输出字典： 返回一个包含生成器输出的字典，包括生成的图像和其他可能的信息，如遮挡图和遮挡处理。
        """
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            # output_dict["deformed"] = self.deform_input(source_image, deformation)  # 3d deformation cannot deform 2d image

        # Decoding part
        out = self.resblocks_2d(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


# 这段代码定义了一个 SPADE 解码器模块 (SPADEDecoder)，它是生成器模型中的一部分。
# 这个 SPADE 解码器的作用是将输入的特征图和条件信息结合，生成最终的生成图像。
class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, feature):
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        x = self.up(x)                
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)                
        x = self.up_1(x, seg)         # 64, 256, 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        x = F.sigmoid(x)
        
        return x


class OcclusionAwareSPADEGenerator(nn.Module):
    # 这个生成器结合了密集运动估计（如果提供了 dense_motion_params）和 SPADE 结构，用于生成包含运动和语义信息的图像。
    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareSPADEGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.decoder = SPADEDecoder()

    def deform_input(self, inp, deformation):
        """
         这个方法用于将输入特征 inp 根据给定的变形 (deformation) 进行变形。它使用了 PyTorch 中的 F.grid_sample 函数。
        """
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        """
        在前向传播中，首先对源图像进行编码过程（downsampling）。
        然后，对编码后的特征进行 3D 表示和变形，根据密集运动估计网络的输出进行处理。
        处理后的特征通过 SPADE 解码器进行解码，生成最终的预测图像。

        """
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            # import pdb; pdb.set_trace()

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            # occlusion_map = torch.where(occlusion_map < 0.95, 0, occlusion_map)
            
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        # Decoding part
        out = self.decoder(out)

        output_dict["prediction"] = out
        # 返回一个字典，其中包含生成图像的预测 (prediction) 和其他相关信息，如遮挡图 (occlusion_map)。
        return output_dict