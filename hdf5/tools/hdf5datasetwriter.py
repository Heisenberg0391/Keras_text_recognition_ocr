import h5py
import mycrnn_pc.config.cfg as cfg

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=2048):
        # 确保输出路径全新
        # if os.path.exists(outputPath):
        #     raise ValueError("路径冲突", outputPath)

        # 打开HDF5文件句柄，创建数据集：
        # 样本图像, RNN输入长度批, 类别序列, 类别序列长度, dims(N, h, w, d)
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(dataKey, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0], cfg.max_label_len), dtype='float')

        # 存储缓冲区大小，初始化缓冲区以及数据集索引
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # 向缓冲区中添加数据行（把二维图像拉伸成一维向量）和类别
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # 检查是否需要将缓冲区写入磁盘
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # 将缓冲区写入磁盘并重置缓冲区
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self):
        # 检查缓冲区中是否有需要写入硬盘的项
        if len(self.buffer['data']) > 0:
            self.flush()
        # 关闭文件句柄
        self.db.close()

