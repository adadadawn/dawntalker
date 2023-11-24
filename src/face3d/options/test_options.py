"""This script contains the test options for Deep3DFaceRecon_pytorch
Deep3DFaceRecon_pytorch 项目中的测试选项配置脚本。与训练选项相似，
测试选项定义了在测试模型时需要使用的各种参数，包括测试数据集的加载方式、测试图像的文件夹路径等。
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--img_folder', type=str, default='examples', help='folder for test images.')

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
