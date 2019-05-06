import os
import itertools
import editdistance
import numpy as np
from keras import backend as K
# from mycrnn_pc.model.vanilla_crnn import CRNN
# from mycrnn_pc.model.cnnctc import CRNN
# from mycrnn_pc.model.lpcrnn1 import CRNN
# from mycrnn_pc.model.lpcrnn import CRNN
# from mycrnn_pc.model.rnet import CRNN
from mycrnn_pc.model.dense_lpcrnn import CRNN

import mycrnn_pc.config.cfg as cfg
import mycrnn_pc.utils.sample_gen as sg
import mycrnn_pc.utils.callback as cbk
from mycrnn_pc.hdf5.preprocessors.img2arypreprocessor import ImageToArrayPreprocessor
import time
from mycrnn_pc.hdf5.tools.hdf5datasetgenerator import HDF5DatasetGenerator
import progressbar
import os


def labels_to_text(labels):
    # 找到类别ID对应的字
    text = []
    for each in labels:
        each = int(each)
        if each == len(dict):  # CTC Blank
            text.append("")  # todo 把空白符画出来
        else:
            text.append(dict[each])
    return "".join(text)


def decode_predict_ctc(out, top_paths=1):
    # 用beam search代替viz callback里的最优路径
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    start = time.time()
    for i in range(top_paths):
        labels = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
                                          greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        results.append(labels)
    end = time.time()
    print("解码一个样本用时:{:.3f}秒".format(end-start))
    return results


def decode_batch(test_func, word_batch):
    # 从batch中选择一个图像送入模型得到softmax的输出矩阵
    # 纵轴是序列元素, 横轴是类别:(33*4203)
    softmax_out = test_func([word_batch])[0]
    ret = []
    for j in range(softmax_out.shape[0]):  # 遍历验证批中的每一个样本
        rnn_out = softmax_out[j, 2:]  # 去除RNN输出的前两项, 这两项受语言模型影响
        # (时刻, 类别)矩阵，对于每个时刻沿着类别轴(axis=1)找到概率最大的类别
        # this should be beam search with a dictionary and language model
        best_path = np.argmax(rnn_out, 1)  # argmax返回的是索引，贪婪寻找最大概率路径
        out_best = list(best_path)
        out_best = [k for k, g in itertools.groupby(out_best)]  # 合并重复字符
        outstr = labels_to_text(out_best)  # 解码类别序列得到字符
        ret.append(outstr)
    return ret  # 当前验证批的所有类别序列字符


def show_stats(num, val_batch_gen, test_func):
    # 显示输出和真实值的编辑距离
    num_left = num
    mean_norm_ed = 0.0
    mean_ed = 0.0
    acc_count = 0.0
    i=0
    while num_left > 0:  # 循环取验证批样本，直到足够num个
        # 取一个验证批
        word_batch = next(val_batch_gen)[0]  # fixme the_input 是nan
        # 计算编辑距离
        num_proc = min(word_batch['the_input'].shape[0], num_left)  # 每次最多取一个batch的样本计算
        decoded_res = decode_batch(test_func, word_batch['the_input'][0:num_proc])  # 模型预测输出
        for j in range(num_proc):
            # 累计样本的编辑距离
            label_length = int(word_batch['label_length'][j])
            label_ids = word_batch['the_labels'][j][0:label_length]
            truth = labels_to_text(label_ids)
            predict = decoded_res[j]
            if predict == truth:
                acc_count += 1
                edit_dist = 0
            else:
                edit_dist = editdistance.eval(predict, truth)
            mean_ed += float(edit_dist)
            mean_norm_ed += float(edit_dist) / len(truth)  # 归一化
            i += 1
            pbar.update(i)
        num_left -= num_proc

    # 全对率
    acc = acc_count/num
    # 平均编辑距离
    mean_norm_ed = mean_norm_ed / num
    mean_ed = mean_ed / num
    print("{}samples, 平均编辑距离:{:.5f}, 平均归一化编辑距离:{:.5f}, 全对率:{:.5f}\n".format(
            num, mean_ed, mean_norm_ed, acc))
    # 保存编辑距离
    with open(os.path.join(output_dir, "dense_fine加噪casia.txt"), 'a', encoding='utf-8') as f:
        f.write("{}samples, 平均编辑距离:{:.5f}, 平均归一化编辑距离:{:.5f}, 全对率:{:.5f}\n".format(
            num, mean_ed, mean_norm_ed, acc))


if __name__ == '__main__':
    img_w = None
    TEST_HDF5 = "C:\mycrnn_dataset\CASIA低噪高斯/test.hdf5"  # 测试集hdf5
    num = 6152  # 测试样本数
    weight_file = "E:\Python\WorkSpace\mycrnn_pc/test_files/dense_casia.h5"  # 模型权重
    output_dir = 'E:\Python\WorkSpace\mycrnn_pc/test_files'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 通道顺序
    if K.image_data_format() == 'channels_first':
        input_shape = (3, cfg.img_h, img_w)
    else:
        input_shape = (cfg.img_h, img_w, 3)

    # 实例化图像生成器
    img_gen = sg.TextGenerator(batch_size=cfg.batch_size,
                               img_w=img_w,
                               img_h=cfg.img_h,
                               downsample_factor=cfg.downsample_factor,
                               train_size=cfg.train_size)

    n_classes = img_gen.get_output_size()
    model, y_pred, test_func = CRNN.build(input_shape, n_classes, train=False)
    model.load_weights(weight_file)
    dict = img_gen.dict  # 字典

    # 初始化预处理器
    iap = ImageToArrayPreprocessor()
    testGen = HDF5DatasetGenerator(TEST_HDF5, n_classes, cfg.batch_size, preprocessors=[iap])

    val_batch_gen = testGen.generator()  # 测试样本生成器

    # 初始化进度条
    widgets = ["正在解码中: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=num,
                                   widgets=widgets).start()
    # 显示编辑距离
    show_stats(num, val_batch_gen, test_func)
