import os
import datetime
import numpy as np
from keras import backend as K
K.set_learning_phase(1)
# from model.vanilla_crnn import CRNN
# from model.wavelet_crnn import CRNN
# from model.cnnctc import CRNN
from model.rnet import CRNN

import config.cfg as cfg
import utils.sample_gen as sg
import utils.callback as cbk
from keras.utils import multi_gpu_model
import time
from keras.utils import plot_model
from mycrnn_pc.hdf5.preprocessors.img2arypreprocessor import ImageToArrayPreprocessor
from mycrnn_pc.hdf5.preprocessors.meanpreprocessor import MeanPreprocessor
from mycrnn_pc.hdf5.tools.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.optimizers import Adam, SGD
import json
import os



def train(OUTPUT_DIR, start_epoch, stop_epoch, lr, n_gpus, img_w=None, public=True):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
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
                               train_size = cfg.train_size)

    # 网络结构
    n_classes = img_gen.get_output_size()
    model, y_pred, test_func = CRNN.build(input_shape, n_classes, train=True)
    model.summary()
    
    # 保存模型图
    plot_model(model, to_file=os.path.join(OUTPUT_DIR, 'model.png'), show_shapes=True)

    # 可视化回调，负责保存模型并画图显示训练进度
    viz_cb = cbk.VizCallback(img_gen, OUTPUT_DIR, test_func, img_gen.val_batch_gen(),
                             start_epoch, n_gpus, public=public)

    # 优化器，clipnorm限制梯度上限，避免遇到梯度悬崖
    # sgd = SGD(lr=lr, momentum=0.9, nesterov=True, clipnorm=10)
    sgd = Adam(lr=lr)

    # 加载模型权重checkpoint继续训练
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, 'weights{}.h5'.format(start_epoch - 1))
        model.load_weights(weight_file)
    if n_gpus > 1:
        # 多GPU
        model = multi_gpu_model(model, gpus=n_gpus)

    # 加载通道均值
    # means = json.loads(open(cfg.DATASET_MEAN).read())

    # 初始化预处理器
    # mp = MeanPreprocessor(means["R"], means["G"], means["B"])
    iap = ImageToArrayPreprocessor()


    # 初始化训练/验证集生成器
    # 参数：HDF5数据集路径，批大小，数据增强， 预处理器，类别数
    trainGen = HDF5DatasetGenerator(cfg.TRAIN_HDF5, n_classes, cfg.batch_size, preprocessors=[iap])
    valGen = HDF5DatasetGenerator(cfg.VAL_HDF5, n_classes, cfg.batch_size, preprocessors=[iap])

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=["accuracy"])
    model.fit_generator(
        generator=trainGen.generator(),
        steps_per_epoch=trainGen.numImages // cfg.batch_size,
        epochs=stop_epoch,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // cfg.batch_size,
        callbacks=[viz_cb, img_gen],
        initial_epoch=start_epoch,
        max_queue_size = cfg.batch_size*4)


if __name__ == '__main__':
    # GPU参数
    n_gpus = 1
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    # sess = tf.Session(cfg=tf.ConfigProto(gpu_options=gpu_options))

    # 输出路径
    np.random.seed(55)
    run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    start_time = time.time()
    OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, run_name)

    # 自定义训练图像宽度，训练stop_epoch-start_epoch次
    train(OUTPUT_DIR, start_epoch=0, stop_epoch=100, lr=1e-4, n_gpus=n_gpus, img_w=128, public=True)
    # train(OUTPUT_DIR, start_epoch=50, stop_epoch=100, lr=1e-4, n_gpus=n_gpus, img_w=128)

    # 保持文本不变，增加宽度，依靠留白实现平移不变性，抑制RNN的语言模型在文本首尾的多余输出
    end_time = time.time()
    print("training time : {:.2f} hours".format((end_time - start_time)/3600))
