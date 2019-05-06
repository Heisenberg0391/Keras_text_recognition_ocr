# coding = utf-8
"""
2019年2月28日09:05:05
功能：过滤样本标注区并保存图片和对应标注文件，
去除(1)文字超纲的(2)难度为1的(3)内角方差太大的(4)歪的(5)纵向文本
输入：RCTW数据集原始图片及其标注，
输出：截取区域图和标注txt
"""
import os
import glob
from PIL import Image
import numpy as np
import codecs
import matplotlib.pyplot as plt
import mycrnn_pc.config.cfg as cfg
import cv2
from skimage import data, exposure, measure
import progressbar


def add_noise(raw_img):
    """
    :param img_path: 图像路径
    :return: 加噪后的图像ndarray
    """
    # mode = np.random.randint(0, 2)
    mode = 0
    if mode == 0:  # 瑞利噪声
        h = raw_img.shape[0]
        w = raw_img.shape[1]
        lim = 430
        scale = np.random.randint(1, 30) * 0.1
        noise = np.random.rayleigh(scale, (h, w))
        factor = np.random.randint(0, lim*10)*0.1
        noise = noise / noise.max() * factor # 控制分布的灰度范围
        noisy_image = np.zeros(raw_img.shape, np.float32)
        # 分通道添加噪声
        noisy_image[:, :, 0] = raw_img[:, :, 0] + noise
        noisy_image[:, :, 1] = raw_img[:, :, 1] + noise
        noisy_image[:, :, 2] = raw_img[:, :, 2] + noise
        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)  # 归一化
        noisy = noisy_image.astype(np.uint8)
    else:  # 高斯噪声
        row, col, ch = raw_img.shape
        mean = 0
        lim = 4500
        var = np.random.randint(0, lim)  # 关键参数
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (row, col))  # 高斯噪声
        noisy_image = np.zeros(raw_img.shape, np.float32)
        # 分通道添加噪声
        noisy_image[:, :, 0] = raw_img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = raw_img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = raw_img[:, :, 2] + gaussian
        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)  # 归一化
        noisy = noisy_image.astype(np.uint8)
    return noisy


def rgb_2_gray(input_img):
    """
    # 转灰度图
    :param input_img: ndarray-like img with RGB channels
    :return: ndarray-like grayscale img
    """
    max_gray = input_img.copy()[:,:,0]  # 模板
    for i in range(input_img.shape[0]):  # h
        for j in range(input_img.shape[1]):  # w
            pixel = input_img[i,j,:]
            gray = max(pixel)  # value
            # gray = 0.5*(max(pixel)+min(pixel))  # luster
            max_gray[i, j] = gray
    # Image.fromarray(max_gray).show()
    return max_gray


def save_mapping_file(label_id_seq, mapping_path, img_name, max_label_len):
    '''
    # 将图像标注映射表保存
    :param map_list: 映射表map_list
    :return: 保存txt
    '''
    # 写图像文件名和类别序列的对照表
    label_sequence = np.ones([1, max_label_len]) * -1  # 类别ID序列
    label_sequence[0, 0:label_len] = label_id_seq
    label_sequence = label_sequence[0].astype('int')
    with codecs.open(mapping_path, mode='a', encoding='utf-8') as f:
        label_sequence = label_sequence.tolist()
        f.write("{}.jpg,{}\n".format(
            img_name, ' '.join(str(e) for e in label_sequence)))


def inner_angle_var(points):
    """
    计算四边形内角标准差
    :param points: 4个顶点list
    :return: 标准差
    """
    angles = []
    for i in range(len(points)):
        p1 = points[i]
        ref = points[i - 1]
        p2 = points[i - 2]
        x1, y1 = p1[0] - ref[0], p1[1] - ref[1]
        x2, y2 = p2[0] - ref[0], p2[1] - ref[1]

        # Use dot product to find angle between vectors
        # This always returns an angle between 0, 180
        numer = (x1 * x2 + y1 * y2)
        denom = np.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2))
        angle = np.arccos(numer / denom) / np.pi * 180  # 不区分内外角
        # 只计算内角
        if not x1 * y2 < x2 * y1:
            angle = 360 - angle
        angles.append(angle)
        # 内角方差
    return np.var(angles)


if __name__ == '__main__':
    # 路径
    issue_img = 'F:\deeplearning\datasets\文本识别数据集\CASIA-10k\\train\PAL03705.jpg'
    img_dir = 'F:\deeplearning\datasets\文本识别数据集\CASIA-10k\\train'
    save_dir = 'F:\deeplearning\datasets\文本识别数据集\CASIA-10k\\noisy_color_crop_train'
    dict_path = 'E:\Python\WorkSpace\mycrnn_pc\config\dict4200.txt'
    mapping_path = os.path.join(save_dir, 'map_list.txt')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    psnr = []
    ssim = []

    with codecs.open(mapping_path, mode='w', encoding='utf-8') as f:
        # 清除遗留文件
        pass

    img_paths = os.path.join(img_dir, "*.jpg")
    # 获取全部图像路径，存成list
    images = glob.glob(img_paths)

    # 打开字典
    dict = []
    with codecs.open(dict_path, mode='r', encoding='utf-8') as f:
        # 按行读取字典
        for line in f:
            # 当前行单词去除结尾，为了正常读取空格，第一行两个空格
            word = line.strip('\r\n')
            # 只要没超出上限就继续添加单词
            dict.append(word)

    # 一些参数
    invalid = 0
    valid = 0
    total = 0
    max_label_len = cfg.max_label_len

    # 初始化进度条
    widgets = ["数据集创建中: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(images),
                                   widgets=widgets).start()

    for per, each in enumerate(images):
        # if valid > 640:
        #     break
        # each = issue_img  # fixme 有问题时用这行
        dir = os.path.split(each)[0]  # 图片文件夹路径
        raw_file_name = os.path.split(each)[1].split('.')[0]  # 不包含后缀的文件名
        annotation = os.path.join(dir, raw_file_name) + '.txt'  # 标注文件
        image = np.array(Image.open(each).convert('RGB'))  # 读取图片
        h = image.shape[0]
        w = image.shape[1]
        # 打开标注文件
        with codecs.open(annotation, mode='r', encoding='ANSI') as f:
            for i, row in enumerate(f):
                flag = True
                total += 1
                row = row.strip('\r\n')
                contents = row.split(',')
                # 构建类别ID序列
                label_seq = contents[-1].strip('\n').strip('"')
                label_id_seq = []
                label_len = len(label_seq)
                if label_len > max_label_len or label_len==0:
                    # 确保CTC的字符数小于序列长度
                    flag = False
                    continue
                for each in label_seq:
                    # 过滤无效文字
                    if each in dict:
                        index = dict.index(each)
                        label_id_seq.append(index)
                    else:
                        flag = False
                        invalid += 1
                        break  # 只要有无效文字，就舍弃当前样本区域
                if not flag:
                    continue
                # 文本区坐标
                up_left_x = int(contents[0])
                up_left_y = int(contents[1])
                up_right_x = int(contents[2])
                up_right_y = int(contents[3])
                low_right_x = int(contents[4])
                low_right_y = int(contents[5])
                low_left_x = int(contents[6])
                low_left_y = int(contents[7])

                if up_left_x >= up_right_x or up_left_y >= low_left_y or low_left_x >= low_right_x or up_right_y >= low_right_y:
                    # 去除不存在的标注区域
                    flag = False
                    continue
                # 去除倾斜度太大
                alpha = np.arctan(np.floor(up_right_y - up_left_y) / np.floor(up_right_x - up_left_x)) / np.pi * 180
                if alpha > 10 or alpha < -10:
                    flag = False
                    continue

                # 计算内角方差判断是否接近矩形，去除太不规则的
                # 坐标归一化防止爆炸
                points = np.array([
                    [up_left_x/w, -up_left_y/h],
                    [up_right_x/w, -up_right_y/h],
                    [low_right_x/w, -low_right_y/h],
                    [low_left_x/w, -low_left_y/h]]).astype('float64')
                std = np.sqrt(inner_angle_var(points))
                if std > 20:  # 内角标准差小于20认为标准矩形
                    flag = False
                    continue

                # 取最大外接矩形
                max_left_x = min(up_left_x, low_left_x)
                max_right_x = max(up_right_x, low_right_x)
                max_up_y = min(up_left_y, up_right_y)
                max_down_y = max(low_left_y, low_right_y)
                h = float(max_down_y - max_up_y)
                w = float(max_right_x - max_left_x)
                if h < 20:
                    # 去除太小的
                    flag = False
                    continue
                if w < 1.5*h:
                    # 去除竖着的
                    flag = False
                    continue
                if w/h > 8:
                    # 去除太长的, 8以内占90%
                    flag = False
                    continue
                # slice image
                cropped = image[max_up_y:max_down_y,max_left_x:max_right_x,:]
                # 避免当前图像中一个有效区都没有
                if cropped.size < 1:
                    flag = False
                    continue
                # save image and annotation file
                img_name = raw_file_name + '_' + str(i+1)
                img_path = os.path.join(save_dir, (img_name+'.jpg'))
                # 保存截取区的灰度图
                # gray_cropped = cropped[:,:,0]  # 初始化灰度图输出矩阵
                # cv2.decolor(cropped, gray_cropped.copy())  # opencv以指针方式直接修改

                if flag:
                    noisy = add_noise(cropped)
                    # 统计
                    # psnr.append(measure.compare_psnr(cropped, noisy))
                    # ssim.append(measure.compare_ssim(cropped, noisy, multichannel=True))
                    resized = cv2.resize(noisy, (128, 32), interpolation=cv2.INTER_CUBIC)
                    Image.fromarray(resized).save(img_path)
                    # Image.fromarray(cropped).convert('L').save(img_path)
                    valid += 1
                    # fixme 标注文件写在一个txt里
                    save_mapping_file(label_id_seq, mapping_path, img_name, max_label_len)
        pbar.update(per)
    pbar.finish()

    print("共{}个，保留{}个，英文{}个".format(total, valid, invalid))
    # print('\navg_psnr:{:.5f}'.format(np.mean(psnr)))
    # print('\navg_ssim:{:.5f}'.format(np.mean(ssim)))