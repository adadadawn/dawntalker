# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import unittest

import numpy as np
from torch.autograd import Variable
"""
这份代码是一个用于进行 PyTorch 单元测试的辅助工具，主要包含以下内容：
as_numpy 函数：将 PyTorch 的张量（Tensor）或变量（Variable）转换为 NumPy 数组。
    如果输入是 Variable，则提取其数据部分。
TorchTestCase 类：继承自 unittest.TestCase，用于编写 PyTorch 单元测试的基类。
    包含了一个用于比较两个张量是否在一定误差范围内接近的辅助函数 assertTensorClose。
assertTensorClose 方法：用于断言两个张量在给定的绝对误差 (atol) 和相对误差 (rtol) 范围内是相似的。
    如果两个张量的差异超过了指定的误差范围，将引发断言错误，并显示相关信息，包括差异、相对差异等。


"""

def as_numpy(v):
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, a, b, atol=1e-3, rtol=1e-3):
        npa, npb = as_numpy(a), as_numpy(b)
        self.assertTrue(
                np.allclose(npa, npb, atol=atol),
                'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())
        )
