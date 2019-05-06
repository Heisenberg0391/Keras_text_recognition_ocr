# -*- coding:utf8 -*-
"""
功能：将语料中的每行文本绘制成图像
输入：语料文件
输出：文本图像
参数：在config.cfg.py中: config_path, corpus, dict, FONT_PATH
"""
import codecs
import numpy as np
import mycrnn_pc.config.cfg as cfg
import os
from PIL import Image, ImageDraw, ImageFont
import progressbar
import glob


def add_noise(raw_img):
    """
    :param img_path: 图像路径
    :return: 加噪后的图像ndarray
    """
    h = raw_img.shape[0]
    w = raw_img.shape[1]
    factor = np.random.randint(100, 6500)* 0.1
    scale = np.random.randint(1, 50) * 0.1
    noise = np.random.rayleigh(scale, (h, w))
    noise = noise / noise.max() * factor  # 控制分布的灰度范围
    noisy = raw_img.copy() + noise
    noisy = noisy / noisy.max() * 255
    return noisy


class TextGenerator():
    def __init__(self, input_shape, save_path):
        """初始化参数来自config文件
        """
        self.img_h = input_shape[0]
        self.img_w = input_shape[1]
        self.depth = input_shape[2]
        # 语料参数
        self.max_row_len = cfg.max_row_len
        self.max_label_len = cfg.max_label_len  # CTC最大输入长度
        self.n_samples = cfg.n_samples
        self.dictfile = cfg.dict  # 字典
        self.dict = []
        self.corpus_file = cfg.corpus  # 语料集
        self.save_path = save_path
        # 加载字体文件
        self.font_factor = 1  # 控制字体大小
        # 加载字体文件
        self.load_fonts()
        # 加载语料集
        self.build_dict()
        self.build_train_list(self.n_samples, self.max_row_len)


    def load_fonts(self):
        """ 加载字体文件并设定字体大小
            TODO： 无需设定字体大小，交给pillow
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
                # 当前行单词去除结尾
                word = line.strip('\r\n')
                # 只要没超出上限就继续添加单词
                self.dict.append(word)
        # 最后一类作为空白占位符
        self.blank_label = len(self.dict)

    def mapping_list(self):
        # 写图像文件名和类别序列的对照表
        file_path = os.path.join(cfg.DATASET_DIR, 'map_list.txt')
        with codecs.open(file_path, mode='w', encoding='utf-8') as f:
            for i in range(len(self.train_list)):
                # 文件名, 类别ID序列, 类别长度, 类别字符序列
                label_sequence = self.label_sequence[i].tolist()
                f.write("{}.png,{}\n".format(
                    i, ' '.join(str(e) for e in label_sequence)))

    def build_train_list(self, n_samples, max_row_len=None):
        # 过滤语料，留下适合的内容组成训练list
        print('正在加载语料...')
        assert max_row_len <= self.max_label_len  # 最大类别序列长度
        self.n_samples = n_samples  # 语料总行数
        sentence_list = []  # 存放每行句子
        self.train_list = []
        self.label_len = [0] * self.n_samples  # 类别序列长度
        self.label_sequence = np.ones([self.n_samples, self.max_label_len]) * -1  # 类别ID序列

        with codecs.open(self.corpus_file, mode='r', encoding='utf-8') as f:
            # 按行读取语料
            for sentence in f:
                sentence = sentence.strip()  # 去除行末回车
                if len(sentence_list) < n_samples:
                    # 只要长度和数量没超出上限就继续添加单词
                    sentence_list.append(sentence)

        np.random.shuffle(sentence_list)  # 打乱语料
        if len(sentence_list) < self.n_samples:
            raise IOError('语料不足')

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

            if filted_sentence is not '':
                # 当前样本的类别序列及其长度
                self.label_len[i] = label_len
                self.label_sequence[i, 0:self.label_len[i]] = label_sequence
            else:  # 单独处理空白样本
                self.label_len[i] = 1
                self.label_sequence[i, 0:self.label_len[i]] = self.blank_label  # 空白符

        self.label_sequence = self.label_sequence.astype('int')
        self.train_list = sentence_list  # 过滤后的训练集
        self.mapping_list()  # 保存图片名和类别序列的 map list

    def paint_text(self, text, i):
        """ 使用PIL绘制文本图像，传入画布尺寸，返回文本图像
        :param h: 画布高度
        :param w: 画布宽度
        :return: img
        """
        # 创建画布
        canvas = Image.new('RGB', (self.img_w, self.img_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)


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
        text_size = font.getsize(text)
        assert text_size[0] < self.img_w - 4
        assert text_size[1] < self.img_h - 4

        # 随机平移
        horizontal_space = self.img_w - text_size[0]
        vertical_space = self.img_h - text_size[1]
        start_x = np.random.randint(2, horizontal_space-2)
        start_y = np.random.randint(2, vertical_space-2)

        # 绘制当前文本行
        draw.text((start_x, start_y), text, font=font, fill=(0, 0, 0, 255))
        img_array = np.array(canvas)

        if self.depth == 1:
            # 取单通道
            grayscale = img_array[:, :, 0]  # [32, 256, 4]
            # grayscale = add_noise(grayscale)
            ndimg = Image.fromarray(grayscale).convert('L')
            # ndimg.show()
            # 保存
            save_path = os.path.join(self.save_path, '{}.png'.format(i))  # 类别序列即文件名
            ndimg.save(save_path)
        else:
            img = img_array
            # todo 数据增强
            # 画图看一下
            ndimg = Image.fromarray(img).convert('RGB')
            # ndimg.show()
            # 保存
            save_path = os.path.join(self.save_path, '{}.png'.format(i))  # 类别序列即文件名
            ndimg.save(save_path)

    def generator(self):
        n_samples = len(self.train_list)
        # 进度条
        widgets = ["数据集创建中: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=n_samples, widgets=widgets).start()

        for i in range(n_samples):
            # 绘制当前文本
            self.paint_text(self.train_list[i], i)
            pbar.update(i)

        pbar.finish()


if __name__ == '__main__':
    np.random.seed(0)  # 决定训练集的打乱情况
    # 输出路径
    if not os.path.exists(cfg.DATASET_DIR):
        os.makedirs(cfg.DATASET_DIR)
    img_h = 32
    img_w = 128
    depth = 1
    # 通道顺序, channel_last
    input_shape = (img_h, img_w, depth)

    # 实例化图像生成器
    if not os.path.exists(cfg.DATASET_DIR):
        os.makedirs(cfg.DATASET_DIR)
    img_gen = TextGenerator(input_shape=input_shape, save_path=cfg.DATASET_DIR)
    img_gen.generator()
