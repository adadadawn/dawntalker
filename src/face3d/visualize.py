# check the sync of 3dmm feature and the audio
# 检查3dmm功能和音频的同步性
"""
这段代码的主要目标是生成一个视频，将3D面部可塑模型（3DMM）的特征与音频结合起来。以下是代码的主要步骤：
"""
import cv2
import numpy as np
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.models.facerecon_model import FaceReconModel
import torch
import subprocess, platform
import scipy.io as scio
from tqdm import tqdm 

# draft
def gen_composed_video(args, device, first_frame_coeff, coeff_path, audio_path, save_path, exp_dim=64):
    ##从 first_frame_coeff 和 coeff_path 中加载3DMM系数。
    coeff_first = scio.loadmat(first_frame_coeff)['full_3dmm']

    coeff_pred = scio.loadmat(coeff_path)['coeff_3dmm']

    coeff_full = np.repeat(coeff_first, coeff_pred.shape[0], axis=0) # 257
#   构建完整的系数矩阵 coeff_full。
    coeff_full[:, 80:144] = coeff_pred[:, 0:64]
    coeff_full[:, 224:227]  = coeff_pred[:, 64:67] # 3 dim translation
    coeff_full[:, 254:]  = coeff_pred[:, 67:] # 3 dim translation
#   初始化一个临时视频文件路径 tmp_video_path。
    tmp_video_path = '/tmp/face3dtmp.mp4'
#   创建一个 FaceReconModel 类的实例。
    facemodel = FaceReconModel(args)
    
    video = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))
#  对于每一帧的系数，使用 FaceReconModel 进行前向传播，获取预测的面部特征点和渲染图像。
    for k in tqdm(range(coeff_pred.shape[0]), 'face3d rendering:'):
        cur_coeff_full = torch.tensor(coeff_full[k:k+1], device=device)

        facemodel.forward(cur_coeff_full, device)

        predicted_landmark = facemodel.pred_lm # TODO.
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        rendered_img = facemodel.pred_face
        rendered_img = 255. * rendered_img.cpu().numpy().squeeze().transpose(1,2,0)
        out_img = rendered_img[:, :, :3].astype(np.uint8)
#  将渲染后的图像写入视频。
        video.write(np.uint8(out_img[:,:,::-1]))

    video.release()
# 使用 FFmpeg 将音频与渲染后的视频结合在一起，生成最终的视频文件。
    command = 'ffmpeg -v quiet -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video_path, save_path)
    subprocess.call(command, shell=platform.system() != 'Windows')

