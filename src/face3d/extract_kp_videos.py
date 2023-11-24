"""这个脚本的主要目的是使用Python中的FaceAlignment库来处理视频，提取人脸关键点。"""

import os
import cv2
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle

from torch.multiprocessing import Pool, Process, set_start_method

# 这个类是一个抽象，用于使用face_alignment库从图像中提取人脸关键点。
class KeypointExtractor():
    def __init__(self, device):
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                                     device=device)   

    # extract_keypoint 方法是一个核心函数，它处理单个图像或图像列表，提取人脸关键点，并将关键点保存在文本文件中。
    def extract_keypoint(self, images, name=None, info=True):
        if isinstance(images, list):
            keypoints = []
            if info:
                i_range = tqdm(images,desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image)
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    keypoints = self.detector.get_landmarks_from_image(np.array(images))[0]
                    break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break    
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)                    
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #BGR（OpenCV 默认的颜色通道顺序）转换为 RGB。
            frame = Image.fromarray(frame)  #使用 PIL 库的 Image.fromarray 函数将 NumPy 数组表示的图像转换为 PIL 图像对象。
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

# 该函数以元组data为输入，其中data包含视频的文件名、命令行选项(opt)和GPU设备ID(device)。
def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置GPU设备。
    kp_extractor = KeypointExtractor()  # 初始化一个KeypointExtractor(kp_extractor)对象。
    images = read_video(filename)  #  使用read_video读取视频帧。 
    name = filename.split('/')[-2:]
    #  如果输出目录不存在，则创建输出目录。
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    #  调用extract_keypoint提取并保存人脸关键点。
    kp_extractor.extract_keypoint(
        images, 
        name=os.path.join(opt.output_dir, name[-2], name[-1])
    )

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    
    for ext in extensions:
        os.listdir(f'{opt.input_dir}')
        print(f'{opt.input_dir}/*.{ext}')
        filenames = sorted(glob.glob(f'{opt.input_dir}/*.{ext}'))
    print('Total number of videos:', len(filenames))
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None
