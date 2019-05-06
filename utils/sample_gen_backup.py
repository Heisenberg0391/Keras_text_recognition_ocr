# -*- coding:utf8 -*-
# 生成合成文本图像用于训练
import codecs
import numpy as np
import mycrnn_pc.config.cfg as cfg
import os
import keras
import keras.backend as k
from PIL import Image, ImageDraw, ImageFont
import glob


def add_noise(raw_img):
    """
    :param img_path: 图像路径
    :return: 加噪后的图像ndarray
    """
    h = raw_img.shape[0]
    w = raw_img.shape[1]
    factor = np.random.randint(500)
    # factor = 500

    noise = np.random.rayleigh(1, (h, w))
    noise = noise / noise.max() * factor  # 控制分布的灰度范围
    noisy = raw_img.copy() + noise
    noisy = noisy / noisy.max() * 255
    return noisy

def resize(nd_image, target_h, target_w):
    """
    :param nd_image: ndimage
    :param target_h: 目标高度
    :return: 保持长宽比的单通道resize图像
    """
    h = nd_image.size[1]
    w = nd_image.size[0]

    if not (h <= target_h):  # 如果超高则先按照高度resize
        new_w = target_h * w // h
        image = nd_image.resize((new_w, target_h), Image.ANTIALIAS)
        # 如果调整完高度后长度合适则返回图像
        if image.size[0] <= target_w:
            return image
    if not (w <= target_w):  # 宽度resize
        new_h = target_w * h // w
        image = nd_image.resize((target_w, new_h), Image.ANTIALIAS)
        return image

class TextGenerator(keras.callbacks.Callback):
    def __init__(self, batch_size, img_w, img_h, downsample_factor, train_size):
        # 初始化参数来自config文件
        super(TextGenerator, self).__init__()
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.train_size = train_size
        # 语料参数
        self.dictfile = cfg.dict
        self.corpus_file = cfg.corpus
        self.dict = []
        self.cur_val_index = self.train_size  # 验证集起始点
        self.cur_train_index = 0
        self.n_samples = cfg.n_samples
        self.max_row_len = cfg.max_row_len
        self.max_label_len = cfg.max_label_len
        # 加载字体文件
        self.load_fonts()
        # 加载语料集
        self.build_dict()
        self.build_sentence_list(self.n_samples, self.max_row_len)

    def load_fonts(self):
        """ 加载字体文件
        :return: self.fonts
        """
        self.fonts = {}  # 所有字体
        self.font_name = []  # 字体名，用于索引self.fonts
        # 字体完整路径
        font_path = os.path.join(cfg.FONT_PATH, "*.*")
        # 获取全部字体路径，存成list
        fonts = list(glob.glob(font_path))
        # 遍历字体文件
        for each in fonts:
            # 字体大小
            fonts_list = {}  # 每一种字体的不同大小
            font_name = each.split('\\')[-1].split('.')[0]  # 字体名
            self.font_name.append(font_name)
            font_size = 25
            for j in range(0, 10):  # 当前字体的不同大小
                # 调整字体大小
                cur_font = ImageFont.truetype(each, font_size, 0)
                fonts_list[str(j)] = cur_font
                font_size -= 1
            self.fonts[font_name] = fonts_list

    def build_dict(self):
        """ 打开字典，加载全部字符到list
            每行是一个字
        :return: self.dict
        """
        with codecs.open(self.dictfile, mode='r', encoding='utf-8') as f:
            # 按行读取语料
            for line in f:
                # 当前行单词去除结尾，为了正常读取空格，第一行两个空格
                word = line.strip('\r\n')
                # 只要没超出上限就继续添加单词
                self.dict.append(word)
        # 最后一位作为空白符(不是空格)
        self.blank_label = len(self.dict)

    def build_sentence_list(self, n_samples, max_row_len=None):
        """ 加载语料数据，编码整数ID
            # return: 直接修改self的属性
            句子数：self.n_samples
            最大句子长度：self.max_row_len
        """
        print('正在加载语料...')
        assert max_row_len <= self.max_label_len  # 最大类别序列长度
        assert n_samples % self.batch_size == 0
        assert (self.train_size * n_samples) % self.batch_size == 0
        self.n_samples = n_samples
        sentence_list = []
        self.train_list = []
        # n_samples行，每行为当前句子的类别序列
        self.label_sequence = np.ones([self.n_samples, self.max_label_len]) * -1
        # 16000长度的数组，每一元代表对应16元数组的长度
        self.label_len = [0] * self.n_samples

        with codecs.open(self.corpus_file, mode='r', encoding='utf-8') as f:
            # 按行读取语料
            for sentence in f:
                sentence = sentence.strip()  # 去除回车
                if sentence is not '':
                    if len(sentence) <= max_row_len and len(sentence_list) < n_samples:
                        # 只要长度和数量没超出上限就继续添加单词
                        sentence_list.append(sentence)
                    elif len(sentence) > max_row_len and len(sentence_list) < n_samples:
                        # 随机截断句子
                        sentence_list.append(sentence[0:np.random.random_integers(1, max_row_len)])

        np.random.shuffle(sentence_list)  # 打乱语料
        if len(sentence_list) < self.n_samples:
            raise IOError('Could not pull enough words corpus file.')

        # 遍历语料中的每一句(行)
        for i, sentence in enumerate(sentence_list):
            # 每个句子的长度
            label_len = len(sentence)
            filted_sentence = ''
            # 将单词分成字符，然后找到每个字符对应的整数ID list
            # n_samples个样本每个一行max_row_len元向量(单词最大长度)，每一元为该字符的整数ID
            label_sequence = []
            for j, word in enumerate(sentence):
                index = self.dict.index(word)
                label_sequence.append(index)
                filted_sentence += word

            self.label_len[i] = label_len
            # 扩展了批维度
            self.label_sequence[i, 0:self.label_len[i]] = label_sequence

        self.train_list = sentence_list  # 过滤后的训练集
        # 扩展维度
        self.label_len = np.expand_dims(np.array(self.label_len), 1)

    def paint_text(self, text):
        """ 使用PIL绘制文本图像，传入画布尺寸，返回文本图像
        :param h: 画布高度
        :param w: 画布宽度
        :return: img
        """
        # 创建画布
        canvas = Image.new('RGB', (self.img_w, self.img_h), (255, 255, 255))
        # canvas[0:] = np.random.randint(150, 250)
        # 转换图像模式，保证合成的两张图尺寸模式一致
        draw = ImageDraw.Draw(canvas)

        # todo rotate先考虑随机旋转再调整字体大小
        # 自动调整字体大小避免超出边界, 至少留白水平10%
        valid_fonts = {}
        np.random.shuffle(self.font_name)
        cur_fonts = self.fonts.get(self.font_name[0])

        # 文本区域上限
        limit = [self.img_w - 4, self.img_h - 4]
        try:
            for each in cur_fonts:
                text_size = cur_fonts[each].getsize(text)  # fixme 慢在这儿了
                if (text_size[0] < limit[0]) and (text_size[1] < limit[1]):
                    # 找到不超出边界的所有字体
                    valid_fonts[each] = cur_fonts.get(each)
        except:
            ValueError('字体太大')

        # print('寻找字体用时{}s'.format(end - start))
        # np.random.shuffle(valid_fonts)
        keys = list(valid_fonts.keys())
        np.random.shuffle(keys)
        font = valid_fonts.get(keys[0])
        # font = self.fonts[-1]  # 最小的字体
        text_size = font.getsize(text)
        assert text_size[0] < self.img_w - 4
        assert text_size[1] < self.img_h - 4

        # 随机平移
        horizontal_space = self.img_w - text_size[0]
        vertical_space = self.img_h - text_size[1]
        start_x = np.random.randint(2, horizontal_space - 2)
        start_y = np.random.randint(2, vertical_space - 2)

        # 绘制当前文本行
        draw.text((start_x, start_y), text, font=font, fill=(0, 0, 0, 255))
        img_array = np.array(canvas)

        # 取单通道
        grayscale = np.array(img_array)[:, :, 0]  # 灰度图[h, w, d]
        grey_img = add_noise(grayscale)
        # 数据增强
        # aug_img = aug(grey_img)
        # 画图看一下
        # img = Image.fromarray(grey_img).convert('L')
        # img.show()
        # 归一化，归一化之后不能直接画图看，都是0-1之间的，全黑
        grey_img = grey_img.astype(np.float32) / 255
        # 扩展表示数据批的维度, shape=(num_in_batch, img_h, img_w)
        grey_img = np.expand_dims(grey_img, 0)
        return grey_img

    def batch_generator(self, index, batch_size):
        """ 生成图像批
        :param index: 已经用过的样本数
        :param batch_size: 图像批大小
        :param train: 表明当前批为训练批还是验证批
        :return:    字典input{样本图像批，样本类别序列批，RNN输出长度批，类别序列长度批}
                    字典output{CTC批}，每一元为当前样本的CTC损失
        """
        # 每当train/val/test需要图像时，随机绘制文本图像
        # 通道顺序
        if k.image_data_format() == 'channels_first':
            train_batch = np.ones([batch_size, 1, self.img_h, self.img_w])
        else:
            train_batch = np.ones([batch_size, self.img_h, self.img_w, 1])

        # 每一行是一个数组，对于输入序列的每一元(每一个anchor)，对应标签数组的一元
        # 一个样本为一行
        labels = np.ones([batch_size, self.max_label_len])
        # 每行代表一个样本的长度(anchor个数)
        input_length = np.zeros([batch_size, 1])
        # 每行代表一个样本的类别序列长度(字符个数)
        label_length = np.zeros([batch_size, 1])
        source_str = []

        for i in range(batch_size):
            # 遍历批中每个样本(每一行文本)
            if k.image_data_format() == 'channels_last':
                # train_batch是数据批，shape=(batch_size, img_h, img_w, channels)
                # 每一元代表一个样本，每个样本为img_h*img_w的文本图像
                # train_batch的第i个样本的0:img_h, 0:img_w，通道1(单通道)
                # 赋值为文本图像, paint_text()返回的img的shape=(w, h)
                train_batch[i, :, 0:self.img_w, 0] = self.paint_text(
                    self.train_list[index + i])[0, :, :]

            if self.train_list[index + i] == '':  # 空白样本
                label_length[i] = 1
                labels[i, :] = self.blank_label
            else:  # 非空样本
                # 当前样本类别真值序列
                labels[i, :] = self.label_sequence[index + i]
                # 减2是舍弃掉RNN输出的前两个没用的时间段。
                label_length[i] = self.label_len[index + i]

            # CTC的输入序列长度
            # 即RNN的输出序列长度=图像宽度/总Pooling尺寸+补边-1步长卷积-RNN输出前两项=200/8+1-2=24
            input_length[i] = self.img_w // self.downsample_factor+1-2
            source_str.append(self.train_list[index + i])

        inputs = {'the_input': train_batch,  # 样本图像批
                  'the_labels': labels,  # 样本类别序列批
                  'input_length': input_length,  # RNN输出长度批
                  'label_length': label_length,  # 类别序列长度批
                  'source_str': source_str } # used for visualization only

        # CTC批，每一元为当前样本的CTC损失
        outputs = {'ctc': np.zeros([batch_size])}  # dummy data for dummy loss function
        return inputs, outputs

    def train_batch_gen(self):
        # 生成下一个训练批
        while 1:
            # 生成一个批
            batch = self.batch_generator(self.cur_train_index, self.batch_size)
            self.cur_train_index += self.batch_size  # 样本指针步进一个批
            if self.cur_train_index >= self.train_size:
                # 当训练样本达到上限，开始生成验证样本
                self.cur_train_index = self.cur_train_index % self.batch_size
            yield batch

    def val_batch_gen(self):
        while 1:
            batch = self.batch_generator(self.cur_val_index, self.batch_size)
            self.cur_val_index += self.batch_size
            if self.cur_val_index >= self.n_samples:
                self.cur_val_index = self.train_size + self.cur_val_index % self.batch_size
            yield batch

    def get_output_size(self):
        return len(self.dict)+1  # 字符类别数+空白符