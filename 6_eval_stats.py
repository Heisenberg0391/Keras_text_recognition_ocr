import os
import itertools
import editdistance
import numpy as np
from keras import backend as K
K.set_learning_phase(1)
# from model.vanilla_crnn import CRNN
from model.wavelet_crnn import CRNN
import config.cfg as cfg
import utils.sample_gen as sg
import os

def labels_to_text(labels):
    # 找到类别ID对应的字
    text = []
    for each in labels:
        if each == len(dict):  # CTC Blank
            text.append("")  # todo 把空白符画出来
        else:
            text.append(dict[each])
    return "".join(text)

def decode_batch(test_func, word_batch):
    # 从batch中选择一个图像送入模型得到softmax的输出矩阵
    # 纵轴是序列元素, 横轴是类别:(21*4001)
    softmax_out = test_func([word_batch])[0]
    ret = []
    for j in range(softmax_out.shape[0]):  # 遍历验证批中的每一个样本
        rnn_out = softmax_out[j, 2:]  # 去除RNN输出的前两项, 这两项受语言模型影响
        # (时刻, 类别)矩阵，对于每个时刻沿着类别轴(axis=1)找到概率最大的类别
        # this should be beam search with a dictionary and language model
        best_path = np.argmax(rnn_out, 1)  # argmax返回的是索引
        out_best = list(best_path)
        out_best = [k for k, g in itertools.groupby(out_best)]  # 合并重复字符
        outstr = labels_to_text(out_best)  # 解码类别序列得到字符
        ret.append(outstr)
    return ret  # 当前验证批的所有类别序列字符

def show_stats(num):
    # 显示输出和真实值的编辑距离
    num_left = num
    mean_norm_ed = 0.0
    mean_ed = 0.0
    acc_count = 0.0
    while num_left > 0:  # 循环取验证批样本，直到足够num个
        # 取一个验证批
        word_batch = next(val_batch_gen)[0]
        # 计算编辑距离
        num_proc = min(word_batch['the_input'].shape[0], num_left)  # 每次最多取一个batch的样本计算
        decoded_res = decode_batch(test_func, word_batch['the_input'][0:num_proc])  # 模型预测输出
        for j in range(num_proc):
            # 累计样本的编辑距离
            edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
            mean_ed += float(edit_dist)
            mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])  # 归一化
        num_left -= num_proc
        # 计算acc
        result = decode_batch(test_func, word_batch['the_input'][0:num_proc])  # 解码结果
        for i in range(len(result)):
            if result[i] == word_batch['source_str'][i]:
                acc_count += 1  # 解码正确的个数
    # 识别率
    acc = acc_count/num
    # 平均编辑距离
    mean_norm_ed = mean_norm_ed / num
    mean_ed = mean_ed / num
    print("{}samples, 平均编辑距离:{:.5f}, 平均归一化编辑距离:{:.5f}, acc:{}\n".format(
            num, mean_ed, mean_norm_ed, acc))
    # 保存编辑距离
    with open(os.path.join(output_dir, "test_stats.txt"), 'a', encoding='utf-8') as f:
        f.write("{}samples, 平均编辑距离:{:.5f}, 平均归一化编辑距离:{:.5f}, acc:{}\n".format(
            num, mean_ed, mean_norm_ed, acc))

if __name__ == '__main__':

    img_w = 128
    testset = "C:\mycrnn_dataset\加噪hdf5/test.hdf5"  # 测试集hdf5
    weight_file = "E:\Python\WorkSpace\mycrnn_pc/test_files/改版weights.h5"  # 模型权重
    output_dir = 'E:\Python\WorkSpace\mycrnn_pc/test_files'

    # 通道顺序
    if K.image_data_format() == 'channels_first':
        input_shape = (1, cfg.img_h, img_w)
    else:
        input_shape = (cfg.img_h, img_w, 1)

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
    val_batch_gen = img_gen.val_batch_gen()  # 测试样本生成器
    num = 1024  # 测试样本数

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 显示编辑距离
    print('正在解码...')
    show_stats(num)


