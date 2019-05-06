import os
import itertools
import editdistance
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
import keras.callbacks
import json
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置matplotlib显示中文
from PIL import Image


class VizCallback(keras.callbacks.Callback):
    """
    显示训练情况反馈
    """
    def __init__(self, img_gen, OUTPUT_DIR, test_func, val_batch_gen, start_epoch, n_gpus=1, public=False):
        super(VizCallback, self).__init__()
        self.public = public
        self.dict = img_gen.dict
        self.test_func = test_func
        self.output_dir = OUTPUT_DIR
        # 训练历史
        self.jsonPath = os.path.join(self.output_dir, "history.json")
        self.val_batch_gen = val_batch_gen
        self.viz_samples = 4  # 图像宽度小于256时不能为奇数
        self.n_gpus = n_gpus
        self.start_epoch = start_epoch
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_train_begin(self, logs={}):
        # 初始化字典
        self.H = {}
        # 如果存在json历史路径，则加载训练历史
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                # 检查是否有开始位置
                if self.start_epoch > 0:
                    # 遍历历史记录中的每一项，去除开始位置后的所有项
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_epoch]

    def labels_to_text(self, labels):
        # 找到类别ID对应的字
        text = []
        for each in labels:
            if each == len(self.dict):  # CTC Blank
                text.append("")  # todo 把空白符画出来
            else:
                text.append(self.dict[each])
        return "".join(text)

    def decode_batch(self, test_func, word_batch):
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
            outstr = self.labels_to_text(out_best)  # 解码类别序列得到字符
            ret.append(outstr)
        return ret  # 当前验证批的所有类别序列字符

    def show_edit_distance(self, num):  # todo 看编辑距离
        # 显示输出和真实值的编辑距离
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:  # 循环取验证批样本，直到足够num个
            word_batch = next(self.val_batch_gen)[0]  # 取一个验证批
            num_proc = min(word_batch['the_input'].shape[0], num_left)  # 每次最多取一个batch的样本计算
            decoded_res = self.decode_batch(self.test_func, word_batch['the_input'][0:num_proc])  # 模型预测输出
            for j in range(num_proc):
                # 累计样本的编辑距离
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])  # 归一化
            num_left -= num_proc
        # 平均编辑距离
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('Out of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f\n'
              % (num, mean_ed, mean_norm_ed))
        # 保存编辑距离
        with open(os.path.join(self.output_dir, "edit_distance.txt"), 'a', encoding='utf-8') as f:
            f.write("共{}个测试样本, 平均编辑距离:{:.5f}, 平均归一化编辑距离:{:.5f}\n".format(num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        """ 保存checkpoint并显示编辑距离
            多GPU会将model拆开并行处理然后concatenate
            保存的时候需要找到对应的model
        """
        # print("正在保存模型...")

        if self.n_gpus > 1:
            try:
                self.model.get_layer('model_1').save_weights(
                    os.path.join(self.output_dir, 'weights{}.h5'.format(epoch)))
            except:
                pass
            try:
                self.model.get_layer('model_3').save_weights(
                    os.path.join(self.output_dir, 'weights{}.h5'.format(epoch)))
            except:
                pass
            try:
                self.model.get_layer('model_5').save_weights(
                    os.path.join(self.output_dir, 'weights{}.h5'.format(epoch)))
            except:
                pass
        else:
            self.model.save_weights(os.path.join(self.output_dir, 'weights{}.h5'.format(epoch)))

        if not self.public:  # 合成数据集再使用viz callback
            # 显示编辑距离
            print('正在解码...')
            self.show_edit_distance(256)
            ''' 生成一个验证批，形如：
            inputs = {'the_input': train_batch,  # 样本图像批
                      'the_labels': labels,  # 样本类别序列批
                      'input_length': input_length,  # RNN输入长度批
                      'label_length': label_length}  # 类别序列长度批
            '''
            val_batch = next(self.val_batch_gen)[0]
            # 解码CTC
            result = self.decode_batch(self.test_func, val_batch['the_input'][0:self.viz_samples])
            if val_batch['the_input'][0].shape[0] < 256:
                # 如果图像长度<256则subplot分两列, 否则画一列
                cols = 2
            else:
                cols = 1
            plt.figure(0)
            for i in range(self.viz_samples):
                plt.subplot(self.viz_samples // cols, cols, i + 1)
                the_input = val_batch['the_input'][i, :, :, 0]*255

                # 绘制回调图
                plt.imshow(the_input, cmap='gray')
                plt.xlabel('Truth = \'{}\'\nDecoded = \'{}\''.format(
                    val_batch['source_str'][i], result[i]))  # 标注真实值和预测值

            # 保存回调图
            plt.savefig(os.path.join(self.output_dir, 'e{}.png'.format(epoch)))
            plt.close()

        # 保存训练曲线
        # 遍历历史记录并更新损失、精度等
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # 检查是否保存训练历史
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dumps(self.H))
            f.close()

        # 确保画图之前至少经过两次训练（从第0次开始）
        if len(self.H['loss']) > 1:
            # 绘制训练损失和精度
            N = np.arange(0, len(self.H['loss']))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            plt.title("Loss[epoch{}]".format(len(self.H['loss'])))
            # 限制坐标轴范围
            # limits = [None, None, -5, 5]
            # plt.axis(limits)
            axes = plt.gca()
            axes.set_ylim([-1, 5])
            plt.xlabel('Epochs #')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "train_loss.jpg"), dpi=1080)
            plt.close()
            # 保存图像
            plt.figure()
            plt.plot(N, self.H['acc'], label='train_acc')
            plt.plot(N, self.H['val_acc'], label='val_acc')
            plt.title("Accuracy[epoch{}]".format(len(self.H['acc'])))
            plt.xlabel('Epochs #')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "train_acc.jpg"), dpi=1080)
            plt.close()