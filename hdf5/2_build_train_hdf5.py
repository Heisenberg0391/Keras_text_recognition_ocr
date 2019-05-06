# import the necessary packages
# import config.cfg as cfg
import glob
from sklearn.model_selection import train_test_split
import mycrnn_pc.config.cfg as cfg
from mycrnn_pc.hdf5.tools.hdf5datasetwriter import HDF5DatasetWriter
import numpy as np
import progressbar
import os
import json
from PIL import Image
import codecs


train_imgs = []  # 样本图像
train_labels = []  # 样本类别序列
label_lengths = [] # 有效类别序列长度
train_label_lengths = []  # 最大类别序列长度
train_source_strs = []  # 用于可视化回调
feature_length = cfg.max_label_len

if not os.path.exists(cfg.HDF5_PATH):
    os.makedirs(cfg.HDF5_PATH)

# 读取mapping list并根据映射表构建数据集
with codecs.open(cfg.MAP_LIST, mode='r', encoding='utf-8') as f:
    print("正在读取映射表")
    for i,each in enumerate(f):
        line = each.strip('\r\n').split(',')  # 将当前行样本的item划分开
        train_imgs.append(os.path.join(cfg.DATASET_DIR, line[0]))  # 根据映射表中的文件名加载图像
        label_str = line[1].strip(' ').strip('[').strip(']')  # 得到类别ID序列str
        train_labels.append(np.fromstring(label_str, dtype=int, sep=' '))  # 类别序列转int并存list

# 划分出10%作为测试集
# split = train_test_split(train_imgs, train_labels,
#                          test_size=0.1, random_state=42)
# (train_imgs, test_imgs, train_labels, test_labels) = split

# 划分出25%作为验证集
split = train_test_split(train_imgs, train_labels,
                         test_size=0.3, random_state=42)

(train_imgs, val_imgs, train_labels, val_labels) = split

# 构建一个list用于将训练、验证、测试图像路径及其类别和HDF5输出文件配对
datasets = [
    ("train", train_imgs, train_labels, cfg.TRAIN_HDF5),
    ("val", val_imgs, val_labels, cfg.VAL_HDF5)]
    #("test", test_imgs, test_labels, cfg.TEST_HDF5)]

# 初始化图像均值
mean = []
(R, G, B) = ([], [], [])
# 图像shape
raw_img = Image.open(train_imgs[0])
image_shape = np.array(raw_img).shape
h = image_shape[0]
w = image_shape[1]
if len(image_shape) == 3:
    d = 3
else:
    d = 1

# 分别保存训练、验证、测试集
for (dType, imgs, labels, outputPath) in datasets:

    # 初始化进度条
    widgets = ["数据集创建中: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imgs),
                                   widgets=widgets).start()

    # 创建 HDF5 writer
    print("\n[INFO] 正在创建 {}...".format(outputPath))
    if d == 3:
        writer = HDF5DatasetWriter((len(imgs), h, w, 3), outputPath)
    else:
        writer = HDF5DatasetWriter((len(imgs), h, w), outputPath)

    # 遍历数据集中的图像路径, i是第i个样本
    for (i, (img, label)) in enumerate(zip(imgs, labels)):
        # 加载图像并预处理
        if d == 3:
            image = np.array(Image.open(img))[:,:,0:3]
        else:
            image = np.array(Image.open(img))
        # 如果当前是训练集，则计算RGB均值并更新
        if dType == "train":
            if d == 3:  # 彩色图做均值预处理
                b_mean = np.mean(image[:, :, 0])
                g_mean = np.mean(image[:, :, 1])
                r_mean = np.mean(image[:, :, 2])
                R.append(r_mean)
                G.append(g_mean)
                B.append(b_mean)
            else:
                r_mean = np.mean(image[:, :])
                R.append(r_mean)
                G.append(-1)
                B.append(-1)

        # 将图像及类别添加至HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    # 关闭HDF5 writer
    pbar.finish()
    writer.close()


# 构建均值字典，将均值保存为json文件
print("[INFO] 正在保存通道均值...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(cfg.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()


