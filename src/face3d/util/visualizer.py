"""This script defines the visualizer for Deep3DFaceRecon_pytorch
这段代码定义了两个类：Visualizer 和 MyVisualizer，用于可视化训练过程中的结果和损失。
"""

import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from torch.utils.tensorboard import SummaryWriter
# 将图像保存到磁盘，并在 HTML 页面中展示这些图像
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.
    webpage: 一个 HTML 页面类的实例，用于存储这些图像。这个参数应该是来自 html.py 模块的 HTML 类的一个实例。
    visuals: 一个有序字典（OrderedDict），存储图像的名称和对应的图像数据。字典的键是图像的名称，值是包含图像数据的张量（tensor）或者 NumPy 数组。
    image_path: 字符串，用于创建图像路径的基础字符串。
    aspect_ratio: 保存图像时的宽高比。
    width: 图像将被调整到的宽度。
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    此功能将存储在“visuals”中的图像保存到“webpage”指定的HTML文件中。
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library tensprboardX for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    这个函数的作用是将训练过程中生成的图像保存到磁盘，并在 HTML 页面中展示这些图像，方便用户查看和分析。
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
            opt：存储所有实验标志的对象，需要是 BaseOptions 的子类。
        Step 1: Cache the training/test options
        Step 2: create a tensorboard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        步骤：
        缓存训练/测试选项。
        创建一个 Tensorboard 写入器（writer）。
        如果是训练阶段且不禁用 HTML（opt.isTrain 为真且 opt.no_html 为假），则创建一个 HTML 对象，用于保存 HTML 文件和图像。
        创建一个日志文件，用于存储训练损失。
        """
        self.opt = opt  # cache the option
        self.use_html = opt.isTrain and not opt.no_html
        self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'logs', opt.name))
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status
             重置 self.saved 状态。
        """
        self.saved = False


    def display_current_results(self, visuals, total_iters, epoch, save_result):
        """Display current results on tensorboad; save current results to an HTML file.
            在 Tensorboard 上显示当前结果；将当前结果保存到 HTML 文件中。
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        参数：
            visuals：一个有序字典，包含要显示或保存的图像。
            total_iters：总迭代次数。
            epoch：当前的训练轮次。
            save_result：布尔值，表示是否将当前结果保存到 HTML 文件中。
        """
        for label, image in visuals.items():
            self.writer.add_image(label, util.tensor2im(image), total_iters, dataformats='HWC')

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, total_iters, losses):
        # G_loss_collection = {}
        # D_loss_collection = {}
        # for name, value in losses.items():
        #     if 'G' in name or 'NCE' in name or 'idt' in name:
        #         G_loss_collection[name] = value
        #     else:
        #         D_loss_collection[name] = value
        # self.writer.add_scalars('G_collec', G_loss_collection, total_iters)
        # self.writer.add_scalars('D_collec', D_loss_collection, total_iters)
        for name, value in losses.items():
            self.writer.add_scalar(name, value, total_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


class MyVisualizer:
    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
            opt：存储所有实验标志的对象，需要是 BaseOptions 的子类。
        Step 1: Cache the training/test options
        Step 2: create a tensorboard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        步骤：
            缓存选项。
            如果不是测试阶段，创建一个 Tensorboard 写入器（writer）。
            如果不是测试阶段，创建一个日志文件，用于存储训练损失。
        """
        self.opt = opt  # cache the optio
        self.name = opt.name
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results')
        
        if opt.phase != 'test':
            self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'logs'))
            # create a logging file to store training losses
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)



    def display_current_results(self, visuals, total_iters, epoch, dataset='train', save_results=False, count=0, name=None,
            add_image=True):
        """Display current results on tensorboad; save current results to an HTML file.
            在 Tensorboard 上显示当前结果；将当前结果保存到 HTML 文件中。
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
            参数：
                visuals：一个有序字典，包含要显示或保存的图像。
                total_iters：总迭代次数。
                epoch：当前的训练轮次。
                dataset：数据集的标识，可以是 'train'、'val' 或 'test'。
                save_results：布尔值，表示是否将当前结果保存到 HTML 文件中。
                count：保存结果时的计数，用于生成文件名。
                name：保存结果时的文件名，可选。
                add_image：布尔值，表示是否将图像添加到 Tensorboard。
        """
        # if (not add_image) and (not save_results): return
        
        for label, image in visuals.items():
            for i in range(image.shape[0]):
                image_numpy = util.tensor2im(image[i])
                if add_image:
                    self.writer.add_image(label + '%s_%02d'%(dataset, i + count),
                            image_numpy, total_iters, dataformats='HWC')

                if save_results:
                    save_path = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d'%(epoch, total_iters))
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    if name is not None:
                        img_path = os.path.join(save_path, '%s.png' % name)
                    else:
                        img_path = os.path.join(save_path, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)


    def plot_current_losses(self, total_iters, losses, dataset='train'):
        """
        total_iters：总迭代次数。
        losses：包含训练损失的字典。
        dataset：数据集的标识，可以是 'train'、'val' 或 'test'。
        """
        for name, value in losses.items():
            self.writer.add_scalar(name + '/%s'%dataset, value, total_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, dataset='train'):
        """print current losses on console; also save the losses to the disk
            在控制台上打印当前损失；同时将损失保存到日志文件中。
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)

        参数：
            epoch：当前的训练轮次。
            iters：当前训练轮次中的迭代次数。
            losses：包含训练损失的字典。
            t_comp：每个数据点的计算时间（归一化为批大小）。
            t_data：每个数据点的数据加载时间（归一化为批大小）。
            dataset：数据集的标识，可以是 'train'、'val' 或 'test'。

        """
        message = '(dataset: %s, epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
            dataset, epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
