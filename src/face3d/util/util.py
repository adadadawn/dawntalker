"""This script contains basic utilities for Deep3DFaceRecon_pytorch
这段脚本包含了用于处理基于 PyTorch 的 3D 人脸重建模型的基本实用工具。
"""
from __future__ import print_function
import numpy as np
import torch
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision


# 将字符串转换为布尔值。如果字符串是 ('yes', 'true', 't', 'y', '1') 中的一个，返回 True，否则返回 False。
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 复制 default_opt 的属性到一个新的 Namespace 对象，用 kwargs 中的值更新任何指定的属性。
def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


# 通过复制 train_opt 的属性生成验证配置。它还处理与 train_opt 中验证相关的属性。
def genvalconf(train_opt, **kwargs):
    conf = Namespace(**vars(train_opt))
    attr_dict = train_opt.__dict__
    for key, value in attr_dict.items():
        if 'val' in key and key.split('_')[0] in attr_dict:
            setattr(conf, key.split('_')[0], value)

    for key in kwargs:
        setattr(conf, key, kwargs[key])

    return conf

# 在给定模块中查找指定名称的类。
def find_class_in_module(target_cls_name, module):
    # 将 target_cls_name 中的下划线去除，并将字符串转换为小写。
    # 这是因为在查找过程中，函数期望类的名称是小写且没有下划线的形式。
    target_cls_name = target_cls_name.replace('_', '').lower()
    # 遍历模块的字典（clslib.__dict__），该字典包含模块中定义的所有对象（包括类）。
    # 对于每个对象，检查其名称是否与经过处理的 target_cls_name 匹配，如果匹配，则将该类对象赋值给 cls 变量。
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj
    # 最后，使用断言（assert）确保找到了匹配的类。
    # 如果没有找到，将触发 AssertionError，其中包含一条错误消息，指示在给定模块中应该存在一个符合规范的类。
    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):

    """"Converts a Tensor array into a numpy image array.
    将 PyTorch 张量图像转换为 NumPy 数组。通常用于可视化图像。
    Parameters:
        input_image (tensor) --  the input image tensor array, range(0, 1)
        函数假定这个张量的值范围在 [0, 1] 之间，即它是归一化的。通常，PyTorch中的图像数据会被归一化到这个范围，以便更好地与深度学习模型一起工作。
        imtype (type)        --  the desired type of the converted numpy array
        这是转换后的NumPy数组的数据类型。默认是 np.uint8，表示8位无符号整数，范围在 [0, 255]。这是常见的图像表示范围。你可以根据需要选择其他数据类型，比如 np.float32。
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # 将张量中的值限制在指定范围内。在这里，它将张量中的值限制在 [0.0, 1.0] 的范围内。
        # clamp(0.0, 1.0) 任何小于0.0的值都将被设置为0.0，任何大于1.0的值都将被设置为1.0。
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            # np.tile(image_numpy, (3, 1, 1))这个函数的目的是在不同维度上复制数组的内容，以扩展数组的尺寸。
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    计算并打印 PyTorch 网络的平均绝对梯度的平均值。
    这个函数的主要目的是帮助调试神经网络。通过计算平均绝对梯度的平均值，可以了解网络训练时参数的梯度情况。
    这对于判断训练是否正常、梯度是否爆炸或消失等问题是有帮助的。
    Parameters:
        net（代表 PyTorch 的神经网络）和可选的 name（网络的名称，默认为 'network'）。
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    将 NumPy 图像数组保存到磁盘。
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    print_numpy(x, val=True, shp=False):
    打印 NumPy 数组的统计信息（平均值、最小值、最大值、中位数、标准差）。
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


# mkdirs(paths) 和 mkdir(path):
# 用于创建目录的实用函数。
def mkdirs(paths):
    """create empty directories if they don't exist
    mkdirs 函数的主要作用是根据提供的路径列表，在文件系统中创建不存在的目录。如果目录已经存在，则不执行任何操作。
    可以创建多个文件夹
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    只能创建一个文件夹
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


# correct_resize_label(t, size) 和 correct_resize(t, size, mode=Image.B:
# 处理标签和图像大小的调整的实用函数。
def correct_resize_label(t, size):
    """
    "调整大小" 指的是调整标签（label）的空间尺寸（shape），
    t: 输入的 PyTorch 张量，表示标签，形状为 (B, C, H, W)，其中C通常是标签的类别数。
    size: 调整大小的目标尺寸，可以是一个整数，也可以是一个包含两个整数的元组 (height, width)。
    """
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


# 其主要功能是对输入的张量 t 进行正确的调整大小操作。
def correct_resize(t, size, mode=Image.BICUBIC):
    """
    t: 输入的 PyTorch 张量，它是一个包含多个图像的批量张量，形状为 (B, C, H, W)。
    size: 调整大小的目标尺寸，可以是一个整数，也可以是一个包含两个整数的元组 (height, width)。
    mode: 调整大小的插值模式，默认为 Image.BICUBIC。
    """
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        # 调整尺寸
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        # 将调整大小后的PIL图像转换为PyTorch张量，并进行归一化操作。
        # to_tensor 的作用是将 PIL 图像转换为具有特定规范的 PyTorch 张量。 将像素值范围从 [0, 255] 缩放到 [0.0, 1.0]。
        # 然后进行了额外的归一化操作（* 2 - 1.0），将像素值范围调整到 [-1, 1]。
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


# 这段代码定义了一个函数 draw_landmarks，其目的是在图像上绘制关键点（landmarks）
def draw_landmarks(img, landmark, color='r', step=2):
    """
    Return:
        img              -- numpy.array, (B, H, W, 3) img with landmark, RGB order, range (0, 255)
        

    Parameters:
        img              -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
        landmark         -- numpy.array, (B, 68, 2), y direction is opposite to v direction
        color            -- str, 'r' or 'b' (red or blue)
    """
    if color =='r':
        c = np.array([255., 0, 0])
    else:
        c = np.array([0, 0, 255.])

    _, H, W, _ = img.shape
    img, landmark = img.copy(), landmark.copy()
    landmark[..., 1] = H - 1 - landmark[..., 1]
    landmark = np.round(landmark).astype(np.int32)
    for i in range(landmark.shape[1]):
        x, y = landmark[:, i, 0], landmark[:, i, 1]
        for j in range(-step, step):
            for k in range(-step, step):
                u = np.clip(x + j, 0, W - 1)  # 用于将数组 x + j 的值限制在指定的范围内。 ，这个表达式的含义是将 x + j 中的每个元素，
                # 如果小于 0，则将其设置为 0，如果大于 W - 1，则将其设置为 W - 1。这样可以确保数组中的元素不会超出指定的范围。
                v = np.clip(y + k, 0, H - 1)
                for m in range(landmark.shape[0]):
                    img[m, v[m], u[m]] = c
    return img
