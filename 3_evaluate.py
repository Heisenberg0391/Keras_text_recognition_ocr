import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from model.crnn import CRNN
import config.cfg as cfg
import utils.sample_gen as sg
import time
from PIL import Image


def labels_to_text(img_gen, labels):
    # 把类别ID转成字符
    ret = []
    for c in labels:
        if c == len(img_gen.dict):  # CTC的空白符
            ret.append("")
        else:
            ret.append(img_gen.dict[c])
    return "".join(ret)


def decode_predict_ctc(out, top_paths=1):
    # 用beam search代替viz callback里的最优路径
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        labels = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
                                          greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        text = labels_to_text(img_gen, labels)
        results.append(text)
    return results


if __name__ == '__main__':
    weight_file = "./*.h5"
    img_w = 200
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

    # 网络结构
    n_classes = img_gen.get_output_size()
    model, _, _ = CRNN.build(input_shape, n_classes, train=False)
    model.load_weights(weight_file)

    grey_img_batch = img_gen.paint_text('端传媒：定于一尊的麻烦')
    grey_img = Image.fromarray(grey_img_batch[0] * 255).convert('LA')  # 去归一化并转成灰度图
    grey_img.show()

    # 扩展通道维度，shape=(num_in_batch, img_h, img_w, channels)
    color_img_batch = np.expand_dims(grey_img_batch, axis=-1)
    start_time = time.time()
    net_out_value = model.predict(color_img_batch)
    pred_texts = decode_predict_ctc(net_out_value, 1)
    end_time = time.time()
    print(pred_texts)
    print("所用时间：{:.2f}秒".format(end_time-start_time))
