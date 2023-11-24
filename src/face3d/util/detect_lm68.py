# 这段代码实现了一个人脸关键点检测的流程，
import os
import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from util.preprocess import align_for_lm
from shutil import move
# 通过np.loadtxt加载均值人脸数据，该数据保存了68个面部关键点的坐标，用于后续的坐标转换。
mean_face = np.loadtxt('util/test_mean_face.txt')
mean_face = mean_face.reshape([68, 2])

# save_label 函数用于将人脸关键点坐标保存到文本文件中。
def save_label(labels, save_path):
    np.savetxt(save_path, labels)

#  该函数接受原始图像、检测到的关键点坐标和保存路径，然后在图像上绘制关键点，最终将结果保存到指定路径。
def draw_landmarks(img, landmark, save_name):
    landmark = landmark
    lm_img = np.zeros([img.shape[0], img.shape[1], 3])
    lm_img[:] = img.astype(np.float32)
    landmark = np.round(landmark).astype(np.int32)

    for i in range(len(landmark)):
        for j in range(-1, 1):
            for k in range(-1, 1):
                if img.shape[0] - 1 - landmark[i, 1]+j > 0 and \
                        img.shape[0] - 1 - landmark[i, 1]+j < img.shape[0] and \
                        landmark[i, 0]+k > 0 and \
                        landmark[i, 0]+k < img.shape[1]:
                    lm_img[img.shape[0] - 1 - landmark[i, 1]+j, landmark[i, 0]+k,
                           :] = np.array([0, 0, 255])
    lm_img = lm_img.astype(np.uint8)

    cv2.imwrite(save_name, lm_img)

# 该函数用于加载图像和关键点坐标，其中包括从文件名中获取的图像路径和关键点坐标文件路径。
def load_data(img_name, txt_name):
    return cv2.imread(img_name), np.loadtxt(txt_name)

# create tensorflow graph for landmark detector
# 用于加载已保存的 TensorFlow 图，该图包含了人脸关键点检测模型的结构和权重。
def load_lm_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='net')
        img_224 = graph.get_tensor_by_name('net/input_imgs:0')
        output_lm = graph.get_tensor_by_name('net/lm:0')
        lm_sess = tf.Session(graph=graph)

    return lm_sess,img_224,output_lm

# landmark detection
# 整个流程的主函数。
"""
它遍历输入目录中的图像文件，依次加载图像和关键点坐标。
对每个图像进行预处理，包括人脸对齐操作。
使用 TensorFlow 图进行人脸关键点检测。
将检测到的关键点坐标转换回原始图像坐标系。
如果检测失败，将相关图像从数据集中移除。
绘制部分图像的关键点并保存，保存所有关键点坐标到文件中。
"""
def detect_68p(img_path,sess,input_op,output_op):
    print('detecting landmarks......')
    # 获取指定目录（img_path）下的所有图片文件的文件名，
    names = [i for i in sorted(os.listdir(
        img_path)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]
    
    # 创建了三个子目录，分别用于可视化结果（vis_path）、
    # 移除无法检测到5个面部关键点的图片（remove_path），
    # 以及保存检测到的68个关键点的文本文件（save_path）。
    vis_path = os.path.join(img_path, 'vis')
    remove_path = os.path.join(img_path, 'remove')
    save_path = os.path.join(img_path, 'landmarks')
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    if not os.path.isdir(remove_path):
        os.makedirs(remove_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
#  对于每张图片，加载图片文件（full_image_name）和相应的包含5个面部关键点的文本文件（full_txt_name）。
    for i in range(0, len(names)):
        name = names[i]
        print('%05d' % (i), ' ', name)
        full_image_name = os.path.join(img_path, name)
        txt_name = '.'.join(name.split('.')[:-1]) + '.txt'
        full_txt_name = os.path.join(img_path, 'detections', txt_name) # 5 facial landmark path for each image

        # if an image does not have detected 5 facial landmarks, remove it from the training list
        # 如果文本文件不存在，说明无法检测到5个面部关键点，将这张图片移动到移除目录，并跳过后续处理。
        if not os.path.isfile(full_txt_name):
            move(full_image_name, os.path.join(remove_path, name))
            continue 

        # load data
        # load_data 函数加载了当前图像文件 full_image_name 和与之对应的关键点文本文件 full_txt_name。
        img, five_points = load_data(full_image_name, full_txt_name)
        # align_for_lm 函数接收原始图像 img 和检测到的五个关键点坐标 five_points。
        # 该函数执行了一些对齐操作，以便为后续的68个关键点检测做准备。
        # 返回值 input_img 包含了经过对齐操作后的图像数据。
        #scale 是一个用于缩放的比例因子，可能用于将对齐后的图像尺寸与模型期望的尺寸匹配。
         #bbox 是一个边界框，可能表示了对齐后的人脸区域。
        input_img, scale, bbox = align_for_lm(img, five_points) # align for 68 landmark detection 

        # if the alignment fails, remove corresponding image from the training list
        # 如果对齐失败（scale == 0），说明该图片无法进行有效的对齐，将该图片移动到移除目录，并跳过后续处理。
        if scale == 0:
            move(full_txt_name, os.path.join(
                remove_path, txt_name))
            move(full_image_name, os.path.join(remove_path, name))
            continue

        # detect landmarks
        # 将处理后的图像输入到人脸关键点检测模型中，得到检测到的68个关键点的坐标。
        input_img = np.reshape(
            input_img, [1, 224, 224, 3]).astype(np.float32)
        landmark = sess.run(
            output_op, feed_dict={input_op: input_img})

        # transform back to original image coordinate
        # 将关键点坐标转换回原始图像坐标系，并保存在相应的文本文件中。
        # landmark 是通过模型检测得到的关键点坐标，通过 reshape 将其变形成 (68, 2) 的形状。
        # 然后，mean_face 被加到每个关键点的坐标上，以进行坐标的变换。这样的操作通常用于将检测到的关键点坐标映射到一个更通用或标准的坐标系上。
        # bbox 是一个包含了人脸边界框信息的数组。这个边界框用于限定关键点检测的区域。具体地，bbox[0] 表示边界框的左上角 x 坐标。
        landmark = landmark.reshape([68, 2]) + mean_face
        landmark[:, 1] = 223 - landmark[:, 1]
        landmark = landmark / scale
        landmark[:, 0] = landmark[:, 0] + bbox[0]
        landmark[:, 1] = landmark[:, 1] + bbox[1]
        landmark[:, 1] = img.shape[0] - 1 - landmark[:, 1]
        # 如果当前处理的图片序号是100的倍数，可视化绘制当前图像上的关键点。
        if i % 100 == 0:
            draw_landmarks(img, landmark, os.path.join(vis_path, name))
        save_label(landmark, os.path.join(save_path, txt_name))
