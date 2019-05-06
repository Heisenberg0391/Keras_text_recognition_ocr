# import the necessary packages

class MeanPreprocessor:
    def __init__(self, rMean, gMean=None, bMean=None):
        # 整个数据集的通道均值
        # 如果只传入一个则认为是灰度图
        self.rMean = rMean
        if gMean and bMean:
            self.gMean = gMean
            self.bMean = bMean

    def preprocess(self, raw_image):
        if self.gMean != -1 :
            # 将图像分成3个独立通道, channel_last
            image = raw_image.copy()
            B = image[:, :, 0]
            G = image[:, :, 1]
            R = image[:, :, 2]
            # 减去均值
            R -= self.rMean
            G -= self.gMean
            B -= self.bMean
            # 合并
            image[:, :, 0] = B
            image[:, :, 1] = G
            image[:, :, 2] = R
            return image
        else:
            # 如果是灰度图则只进行普通归一化
            image = raw_image.copy()
            return image.astype("float") / 255.0
