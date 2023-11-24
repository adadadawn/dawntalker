"""This script is to generate training list files for Deep3DFaceRecon_pytorch
这段代码是为 Deep3DFaceRecon_pytorch 生成训练列表文件。
"""

import os

# save path to training data
def write_list(lms_list, imgs_list, msks_list, mode='train',save_folder='datalist', save_name=''):
    """接受三个列表，分别是 lms_list（包含关键点文件的路径列表）、
    imgs_list（包含图像文件的路径列表）
    和 msks_list（包含掩码文件的路径列表）。
    """"
    save_path = os.path.join(save_folder, mode)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, save_name + 'landmarks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in lms_list])

    with open(os.path.join(save_path, save_name + 'images.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in imgs_list])

    with open(os.path.join(save_path, save_name + 'masks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in msks_list])   

# check if the path is valid
def check_list(rlms_list, rimgs_list, rmsks_list):
    """接受三个列表，分别是 rlms_list（原始关键点文件路径列表）、
    rimgs_list（原始图像文件路径列表）和 
    rmsks_list（原始掩码文件路径列表）。
    调用 check_list 函数，可以筛选出有效的数据样本，并将这些有效样本的路径写入训练列表文件，方便在训练深度学习模型时使用。

    对每个数据样本的路径进行检查，确保关键点、图像和掩码文件都存在。
    如果存在，则将路径添加到新的列表 lms_list、imgs_list 和 msks_list 中，并输出一个标志 'true'；否则输出 'false'。
    返回筛选后的列表。
    """
    lms_list, imgs_list, msks_list = [], [], []
    for i in range(len(rlms_list)):
        flag = 'false'
        lm_path = rlms_list[i]
        im_path = rimgs_list[i]
        msk_path = rmsks_list[i]
        if os.path.isfile(lm_path) and os.path.isfile(im_path) and os.path.isfile(msk_path):
            flag = 'true'
            lms_list.append(rlms_list[i])
            imgs_list.append(rimgs_list[i])
            msks_list.append(rmsks_list[i])
        print(i, rlms_list[i], flag)
    return lms_list, imgs_list, msks_list
