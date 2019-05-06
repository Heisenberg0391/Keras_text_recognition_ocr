from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # 存储图像格式
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # 使用Keras的工具函数重整图像维度
        return img_to_array(image, data_format=self.dataFormat)