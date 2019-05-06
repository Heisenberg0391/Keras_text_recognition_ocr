"""
功能：读取hdf5文件, 遍历数据组成数据批
      以generator的形式yield给model.fit_generator
输入： HDF5文件
输出： yield inputs, outputs
"""
import numpy as np
import h5py
import mycrnn_pc.config.cfg as cfg
import codecs
import threading


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def get_label_len(label_seq, nums, blank_label):  # todo
    label_length = np.zeros([nums, 1])
    for i, labels in enumerate(label_seq):
        length = 0
        if labels[0] == blank_label:  # 空白样本
            label_length[i] = 1
            continue
        for label in labels:  # 非空样本
            if label != -1:  length += 1
        label_length[i] = length
    return label_length


class HDF5DatasetGenerator:
    def __init__(self, dbPath, n_classes, batchSize, preprocessors=None,
        aug=None):
        # 批大小，预处理器，数据增强，是否one-hot编码，类别总数
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.max_label_len = cfg.max_label_len
        # 打开HDF5 database确定样本总数
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
        self.blank_label = n_classes

    def generator(self, passes=np.inf):
        # 当前训练次数
        epochs = 0

        # 跟随训练不断循环生成数据集，直到模型满足条件停止训练
        while epochs < passes:
            # 遍历 HDF5 dataset，从0到numImages，每次步进一个批
            for i in np.arange(0, self.numImages, self.batchSize):
                # todo 在这里实现batch generator
                # 提取所需数据，从i开始取一个batchSize这么多，假设batchSize是32
                # 这里的批就是ndarray格式
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]
                input_length = np.ones([len(images), 1]) * self.max_label_len
                # 每行代表一个样本的类别序列长度(字符个数)
                label_length = get_label_len(labels, len(images), self.blank_label)

                # 检查是否有预处理器
                if self.preprocessors is not None:
                    # 存放已处理图像
                    procImages = []

                    # 遍历图像
                    for image in images:
                        # 依次应用预处理器
                        # cv2.imshow('image', image)
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        # 更新已处理图像
                        procImages.append(image)
                    # 将已处理图像转为numpy阵列
                    images = np.array(procImages)
                # 检查是否需要数据增强
                if self.aug is not None:
                    # aug.flow是一个生成器，每次循环回来会接着上次yield的位置继续计算下一个
                    # 这样不用把所有数据增强的结果存在内存里，节省空间
                    images = next(self.aug.flow(images, batch_size=self.batchSize))

                # 输出4元组
                inputs = {'the_input': images,  # 样本图像批
                          'the_labels': labels,  # 样本类别序列批
                          'input_length': input_length,  # RNN输出长度批
                          'label_length': label_length}  # 类别序列长度批

                # CTC批，每一元为当前样本的CTC损失
                outputs = {'ctc': np.zeros([self.batchSize])}  # dummy data for dummy loss function
                yield inputs, outputs

            # 累加训练次数
            epochs += 1

    def close(self):
        # 关闭 database
        self.db.close()
