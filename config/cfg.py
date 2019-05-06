# coding=utf-8
import os
# 路径参数
OUTPUT_DIR = 'saved_weights/'
DATASET_DIR = 'path'
MAP_LIST = os.path.join(DATASET_DIR, 'map_list.txt')

HDF5_PATH = 'path'


TRAIN_HDF5 = os.path.join(HDF5_PATH, 'train.hdf5')
VAL_HDF5 = os.path.join(HDF5_PATH, 'val.hdf5')
TEST_HDF5 = os.path.join(HDF5_PATH, 'test.hdf5')
DATASET_MEAN = os.path.join(HDF5_PATH, 'mean.json')

config_path = 'path'
corpus = os.path.join(config_path, 'splited_sentences.txt')  # 语料集
dict = os.path.join(config_path, 'dict4200.txt')
FONT_PATH = 'path'  # 字体文件路径

# 超参数
img_h = 32  # 图像高度固定
batch_size = 128  # 原为64，总样本数64*10000
downsample_factor = 2**2
# 每个epoch样本总数
n_samples = batch_size * 5000 # 保证划分数据集时*0.8*0.8不留小数
# 样本句子长度
max_row_len = 8
# RNN输入序列长度，即input_length=200/8+1-2
max_label_len = 33

# 划分数据集
val_proportion = 0.25
val_size = int(n_samples * val_proportion)  #  测试集占比25%
train_size = n_samples - val_size


