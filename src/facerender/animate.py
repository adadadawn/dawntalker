"""
这份代码是一个用于生成面部动画的Python脚本
"""

import os
import cv2
import yaml
import numpy as np
import warnings
from skimage.util import img_as_ubyte
import safetensors
import safetensors.torch 
warnings.filterwarnings('ignore')


import imageio
import torch
import torchvision


from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from src.facerender.modules.make_animation import make_animation 

from pydub import AudioSegment 
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark

try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

class AnimateFromCoeff():#

    def __init__(self, sadtalker_path, device):
        """

        :param sadtalker_path: 一个包含模型和配置文件路径的字典
        :param device: 指定模型在哪个设备上运行（例如，'cuda'或'cpu'）。
        """
        # 使用yaml模块打开并加载facerender_yaml文件，该文件包含了模型的配置参数。
        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)
        # 加载网络
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])

        generator.to(device)
        kp_extractor.to(device)
        he_estimator.to(device)
        mapping.to(device)
        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False 
        for param in he_estimator.parameters():
            param.requires_grad = False
        for param in mapping.parameters():
            param.requires_grad = False

        if sadtalker_path is not None:#根据不同的sadtalker_path加载不同的关键点参数
            if 'checkpoint' in sadtalker_path: # use safe tensor
                self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=None)
            else:
                self.load_cpk_facevid2vid(sadtalker_path['free_view_checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        if  sadtalker_path['mappingnet_checkpoint'] is not None:
            self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=mapping)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.") 

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()
         
        self.device = device


    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None, 
                        kp_detector=None, he_estimator=None,  
                        device="cpu"):
        """
        用于从安全张量（safetensor）格式的checkpoint文件中加载权重到模型中。
        :param checkpoint_path: 表示模型的checkpoint文件的路径。
        :param generator:表示生成器模型的实例
        :param kp_detector:表示关键点检测器模型的实例
        :param he_estimator:表示头部姿势估计器模型的实例
        :param device:表示模型加载到的设备类型
        :return:
        """
        checkpoint = safetensors.torch.load_file(checkpoint_path)

        if generator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'generator' in k:
                    x_generator[k.replace('generator.', '')] = v
            generator.load_state_dict(x_generator)
        if kp_detector is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'kp_extractor' in k:
                    x_generator[k.replace('kp_extractor.', '')] = v
            kp_detector.load_state_dict(x_generator)
        if he_estimator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'he_estimator' in k:
                    x_generator[k.replace('he_estimator.', '')] = v
            he_estimator.load_state_dict(x_generator)
        
        return None

    def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None,
                        kp_detector=None, he_estimator=None, optimizer_generator=None, 
                        optimizer_discriminator=None, optimizer_kp_detector=None, 
                        optimizer_he_estimator=None, device="cpu"):
        """
        从指定的检查点文件中加载模型状态和优化器状态，以便在恢复模型训练或使用预训练模型时使用。
        :param checkpoint_path:
        :param generator:
        :param discriminator:
        :param kp_detector:
        :param he_estimator:
        :param optimizer_generator:
        :param optimizer_discriminator:
        :param optimizer_kp_detector:
        :param optimizer_he_estimator:
        :param device:
        :return: checkpoint['epoch'] 返回从检查点文件中加载的训练时的 epoch数。
                这个信息可能对于继续训练模型或记录训练进度很有用。
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])
        # 返回从检查点文件中加载的训练时的 epoch 数。这个信息可能对于继续训练模型或记录训练进度很有用。
        return checkpoint['epoch']
    
    def load_cpk_mapping(self, checkpoint_path, mapping=None, discriminator=None,
                 optimizer_mapping=None, optimizer_discriminator=None, device='cpu'):
        """
        主要用于从指定的检查点文件中加载映射器（mapping）模型的状态和优化器状态，
        以便在恢复模型训练或使用预训练模型时使用。
        :param checkpoint_path:表示模型检查点文件的路径。该文件包含了训练模型时保存的权重和其他相关信息
        :param mapping:表示映射器（mapping）模型的实例
        :param discriminator: 可选参数，表示鉴别器模型的实例。
        :param optimizer_mapping:表示映射器优化器的实例
        :param optimizer_discriminator:表示鉴别器优化器的实例
        :param device: 字符串参数，表示模型加载到的设备类型。
        :return:
        """
        checkpoint = torch.load(checkpoint_path,  map_location=torch.device(device))
        if mapping is not None:
            mapping.load_state_dict(checkpoint['mapping'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch']

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):
        """
        主要用于生成动画视频
        :param x:生成动画所需的输入数据x
        :param video_save_dir:视频保存目录
        :param pic_path:图片路径
        :param crop_info:裁剪信息
        :param enhancer:增强器
        :param background_enhancer:背景增强器
        :param preprocess:预处理方式
        :param img_size:图像大小
        :return:
        """
        # 从输入数据x中提取源图像、源语义和目标语义信息。
        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor)
        # 将提取的图像和语义信息移动到指定的设备（由self.device指定）上。
        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        # 检查输入数据中是否包含头部姿态参数（yaw），如果有则提取并移动到指定设备上，否则设为None。
        if 'yaw_c_seq' in x:
            yaw_c_seq = x['yaw_c_seq'].type(torch.FloatTensor)
            yaw_c_seq = x['yaw_c_seq'].to(self.device)
        else:
            yaw_c_seq = None
        # 同理，对于pitch_c_seq和roll_c_seq也进行了相同的处理。
        if 'pitch_c_seq' in x:
            pitch_c_seq = x['pitch_c_seq'].type(torch.FloatTensor)
            pitch_c_seq = x['pitch_c_seq'].to(self.device)
        else:
            pitch_c_seq = None
        if 'roll_c_seq' in x:
            roll_c_seq = x['roll_c_seq'].type(torch.FloatTensor) 
            roll_c_seq = x['roll_c_seq'].to(self.device)
        else:
            roll_c_seq = None
        # 从输入数据中提取视频的帧数。
        frame_num = x['frame_num']
        # 调用make_animation函数生成预测视频，传入了源图像、源语义、目标语义以及相关的模型实例和姿态参数
        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                        yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True)
        # 对生成的视频进行形状调整，确保帧数与输入要求一致。
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]
        # 将生成的视频帧从PyTorch Tensor格式转换为NumPy数组，并进行一些形状调整和数据类型转换。
        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)
        # 如果提供了裁剪信息，保持生成视频的宽高比，并调整为指定的图像大小。
        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0]
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
        # 构建视频文件的保存路径，使用imageio库保存生成的视频帧为动画视频。
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        
        imageio.mimsave(path, result,  fps=float(25))
        # 构建最终的视频文件路径，并将其赋值给return_path。
        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        # 从输入数据中提取音频文件路径，并构建新的音频文件路径。
        audio_path =  x['audio_path'] 
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        # 从音频文件中提取并保存与视频帧数对应的音频片段。
        start_time = 0
        # cog will not keep the .mp3 filename
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num 
        end_time = start_time + frames*1/25*1000
        word1=sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")
        # 调用save_video_with_watermark函数，将生成的视频和新的音频合并，生成最终的视频文件。
        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        # 打印生成的视频文件的名称。
        print(f'The generated video is named {video_save_dir}/{video_name}')
        # 如果预处理方式中包含"full"，则进行全图处理，生成带有水印的完整视频，并更新return_path为完整视频的路径
        if 'full' in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path

        # 如果提供了增强器（enhancer），则使用增强器对视频进行处理，并生成增强后的视频
        # 最终的视频路径更新为增强后的视频路径。
        #### paste back then enhancers
        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer

            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            except:
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            
            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            os.remove(enhanced_path)
        # 删除临时文件，即生成视频的中间文件和新生成的音频文件。
        os.remove(path)
        os.remove(new_audio_path)
        # 返回生成的视频文件路径，可以是原始视频、全图处理后的视频或增强后的视频，具体取决于预处理方式和是否使用增强器。
        return return_path

