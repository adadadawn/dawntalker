"""This script contains the image preprocessing code for Deep3DFaceRecon_pytorch
这段代码是Deep3DFaceRecon_pytorch项目中的图像预处理代码。
整个流程的目标是将输入的人脸图像对齐到一个标准的3D人脸形状，以便后续的人脸重建任务。
"""

import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2
import os
from skimage import transform as trans
import torch
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


# calculating least square problem for image alignment
# POS 函数：这个函数计算了通过最小二乘法解决图像对齐问题的平移和缩放参数。
# 具体而言，它使用了人脸关键点的标准3D坐标和检测到的2D坐标，通过线性方程求解得到平移向量 t 和缩放因子 s。
def POS(xp, x):
    """
    xp: 一个形状为 (2, N) 的 NumPy 数组，其中 N 是人脸关键点的数量。这个数组包含了检测到的人脸关键点的 2D 坐标，通常是 (x, y) 的形式。
    x: 一个形状为 (3, N) 的 NumPy 数组，表示对应的标准3D人脸关键点的坐标。这个数组包含了每个关键点的标准3D坐标，通常是 (x, y, z) 的形式。
    """
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s
    
# resize and crop images for face reconstruction
# resize_n_crop_img 函数：该函数用于将图像进行调整和裁剪，以用于人脸重建。
# 它接受原始图像、关键点、平移向量 t、缩放因子 s 和目标大小作为输入，并返回调整后的图像、关键点和（可选的）掩码。
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

# utils for face reconstruction
# extract_5p 函数：这个函数从所有的人脸关键点中提取出5个标志性的点，以便用于进一步的图像对齐操作。
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# utils for face reconstruction
# align_img 函数：这是主要的图像对齐函数，它接受原始图像、关键点、标准3D人脸关键点、（可选的）掩码以及其他一些参数。
# 它首先调用 POS 函数计算平移向量 t 和缩放因子 s，
# 然后使用这些参数调用 resize_n_crop_img 函数对图像进行裁剪和调整。最后，它返回调整后的图像、关键点、掩码以及转换参数。
def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty) tx,ty平移向量，裁剪前的
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)
    
    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)

    transparams（numpy.array）： 包含以下五个值的 NumPy 数组：原始图像的宽度 raw_W、原始图像的高度 raw_H、缩放因子 scale、水平平移量 tx、垂直平移量 ty。这些参数描述了图像的缩放和平移信息。
    img_new（PIL.Image）： 调整和裁剪后的图像，是一个 PIL 图像对象。其尺寸为 (target_size, target_size, 3)。
    lm_new（numpy.array）： 平移缩放后包含 68 个关键点的二维坐标的 NumPy 数组。lm_new 的形状为 (68, 2)，其中每一行包含一个关键点的 x 和 y 坐标。
    mask_new（PIL.Image）： 调整和裁剪后的掩码图像，是一个 PIL 图像对象。其尺寸为 (target_size, target_size)。

    img（PIL.Image）： 原始的人脸图像，是一个 PIL 图像对象。其尺寸为 (raw_H, raw_W, 3)。
    lm（numpy.array）： 包含 68 个关键点的二维坐标的 NumPy 数组。lm 的形状为 (68, 2)，其中每一行包含一个关键点的 x 和 y 坐标。在这里，“y 方向是与 v 方向相反”表示 y 坐标的方向可能是以图像底部为原点的，与常见的坐标系统相反。
    lm3D（numpy.array）： 包含 5 个标准 3D 关键点的三维坐标的 NumPy 数组。lm3D 的形状为 (5, 3)，其中每一行包含一个关键点的 x、y 和 z 坐标。
    mask（PIL.Image）： 原始的人脸掩码图像，是一个 PIL 图像对象。其尺寸为 (raw_H, raw_W, 3)。
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    # 使用 5 个人脸关键点（facial landmarks）和标准 3D 人脸关键点（standard landmarks of a 3D face）来计算平移和缩放因子
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new
