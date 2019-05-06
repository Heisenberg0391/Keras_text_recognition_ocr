"""
2019年3月13日14:34:13
功能：对crop_img.py过滤后的公开数据集加噪并resize
输入：图像
输出：加噪resize图像
"""
import os
import glob
from PIL import Image
import numpy as np
import codecs
import matplotlib.pyplot as plt
import mycrnn_pc.config.cfg as cfg
import cv2
from skimage import data, exposure


def add_noise(raw_img):
    """
    :param img_path: 图像路径
    :return: 加噪后的图像ndarray
    """
    h = raw_img.shape[0]
    w = raw_img.shape[1]
    factor = np.random.randint(100, 3000)* 0.1
    scale = np.random.randint(1, 50) * 0.1
    noise = np.random.rayleigh(scale, (h, w))
    noise = noise / noise.max() * factor  # 控制分布的灰度范围
    noisy = raw_img.copy() + noise
    noisy = noisy / noisy.max() * 255
    return noisy

if __name__ == '__main__':
    img_dir = 'F:\deeplearning\datasets\文本识别数据集\icdar2017rctw_train_v1.2\\cropped'
    save_dir = 'F:\deeplearning\datasets\文本识别数据集\icdar2017rctw_train_v1.2\\resize_cropped'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_paths = os.path.join(img_dir, "*.jpg")

    # 获取全部图像路径，存成list
    images = glob.glob(img_paths)

    # 遍历图像
    for each in images:
        file_name = os.path.split(each)[1]  # 文件名
        image = np.array(Image.open(each))  # 读取图片
        # noisy = add_noise(image)
        normalizedImg = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # 灰度归一化
        resized_noisy = cv2.resize(normalizedImg, (128, 32), interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(save_dir, file_name)
        Image.fromarray(resized_noisy).convert('L').save(save_path)