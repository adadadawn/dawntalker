import numpy as np
import cv2, os, sys, torch
from tqdm import tqdm
from PIL import Image 

# 3dmm extraction
import safetensors
import safetensors.torch 
from src.face3d.util.preprocess import align_img
from src.face3d.util.load_mats import load_lm3d
from src.face3d.models import networks

from scipy.io import loadmat, savemat
from src.utils.croper import Preprocesser


import warnings

from src.utils.safetensor_helper import load_x_from_safetensor 
warnings.filterwarnings("ignore")

def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations # 3
        }


class CropAndExtract():#comments from dawn
    # return coeff_path, png_path, crop_info 返回的是 参数路径 图片路径 图片剪裁后的参数
    def __init__(self, sadtalker_path, device):
        """
        用于图像裁剪和3DMM（3D可变模型）参数提取
        :param sadtalker_path:sadtalker_path包含一些路径信息的字典
        :param device:指定的设备（GPU或CPU）。
        """
        # 初始化了一个Preprocesser类的实例，该实例用于预处理。
        self.propress = Preprocesser(device)
        # 定义了一个ResNet50网络模型(net_recon)，并将其加载到指定的设备上。
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
        # 根据sadetalker_path中的信息加载了预训练的网络权重。
        if sadtalker_path['use_safetensor']:
            checkpoint = safetensors.torch.load_file(sadtalker_path['checkpoint'])    
            self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        else:
            checkpoint = torch.load(sadtalker_path['path_of_net_recon_model'], map_location=torch.device(device))    
            self.net_recon.load_state_dict(checkpoint['net_recon'])

        self.net_recon.eval()
        self.lm3d_std = load_lm3d(sadtalker_path['dir_of_BFM_fitting'])
        self.device = device
    
    def generate(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):
        """
        生成的3DMM参数
        :param input_path:接受输入图像的路径 (input_path)，
        :param save_dir:保存目录 (save_dir)，
        :param crop_or_resize:裁剪或调整大小模式
        :param source_image_flag:是否是源图像
        :param pic_size:图片大小
        :return:
        """
        # 提取图像的文件名，并构建保存提取信息的文件路径。
        # 从输入路径中提取图像文件名，去掉扩展名
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]  
        # 存储提取的关键点信息的文件路径
        landmarks_path =  os.path.join(save_dir, pic_name+'_landmarks.txt')
        # 存储生成的3DMM参数的文件路径。
        coeff_path =  os.path.join(save_dir, pic_name+'.mat')
        #  存储处理后的图像的文件路径。
        png_path =  os.path.join(save_dir, pic_name+'.png')  

        # load input
        # 加载输入图像，支持图像和视频两种类型。
        if not os.path.isfile(input_path):
            raise ValueError('input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [cv2.imread(input_path)]
            fps = 25
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = [] 
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break 
                full_frames.append(frame) 
                if source_image_flag:
                    break

        # 这行代码使用了列表推导式（list comprehension）将视频的每一帧从BGR颜色空间转换为RGB颜色空间，
        # 并将转换后的帧添加到新的列表 x_full_frames 中。
        x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames] 

        #### crop images as the
        # 根据提供的裁剪或调整大小模式，调用self.propress.crop方法进行图像处理，获取裁剪信息 crop_info。
        # 根据crop_or_resize参数的值执行不同的裁剪或调整大小操作，并生成相应的裁剪信息 crop_info。
        if 'crop' in crop_or_resize.lower(): # default crop 如果 crop_or_resize 包含子字符串 'crop'：
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            # 计算裁剪后的图像在原图中的位置范围，并将这些坐标信息存储在 crop_info 中
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif 'full' in crop_or_resize.lower(): # 如果 crop_or_resize 包含子字符串 'full'：
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            # 计算裁剪后的图像在原图中的位置范围，并将这些坐标信息存储在 crop_info 中
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else: # resize mode
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1]
            # crop_info 存储了整个图像的大小。
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        # 将每一帧的图像转换为PIL格式，并保存处理后的图像到 png_path。
        frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        # save crop info
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # 2. get the landmark according to the detected face.
        # 如果 landmarks_path 文件不存在，使用模型提取图像中的关键点信息，并保存到 landmarks_path 中。
        if not os.path.isfile(landmarks_path): 
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        # 如果有landmarks_path
        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # coeff_path 文件不存在，循环处理每一帧图像：
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch
            # video_coeffs: 是一个列表，每个元素对应于输入视频中的一帧。每个元素都是一个包含3DMM参数的NumPy数组，
            # full_coeffs: 是一个列表，每个元素对应于输入视频中的一帧。每个元素都是一个包含完整3DMM参数的NumPy数组，表示对应帧的所有参数，
            video_coeffs, full_coeffs = [],  []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video:'):

                frame = frames_pil[idx]
                W,H = frame.size
                lm1 = lm[idx].reshape([-1, 2])
            
                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2]+1)/2.
                    lm1 = np.concatenate(
                        [lm1[:, :1]*W, lm1[:, 1:2]*H], 1
                    )
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]
                #  lm1是（68,3）
                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)
 
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_t = torch.tensor(np.array(im1)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
                
                with torch.no_grad():
                    # 得到一个所有参数都拼接在一起的数组
                    full_coeff = self.net_recon(im_t)  # (1, 256)
                    # 返回各个3dmm参数的字典
                    coeffs = split_coeff(full_coeff)

                pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
 
                pred_coeff = np.concatenate([
                    pred_coeff['exp'],  # 64
                    pred_coeff['angle'], # 3
                    pred_coeff['trans'], # 3 tx, ty，tz
                    trans_params[2:][None], # scale, tx, ty
                    ], 1)   #（1,length）
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.cpu().numpy())

            semantic_npy = np.array(video_coeffs)[:,0] 

            savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]})

        return coeff_path, png_path, crop_info
