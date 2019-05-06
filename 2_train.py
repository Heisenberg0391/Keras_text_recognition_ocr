import datetime
import numpy as np
from keras import backend as K
K.set_learning_phase(1)
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from model.vanilla_crnn import CRNN
import config.cfg as cfg
import utils.sample_gen as sg
import utils.callback as cbk
from keras.utils import multi_gpu_model
import time
from keras.utils import plot_model
import tensorflow as tf
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def train(OUTPUT_DIR, start_epoch, stop_epoch, img_w, lr, n_gpus):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
                               train_size = cfg.train_size)

    # 网络结构
    n_classes = img_gen.get_output_size()
    model, y_pred, test_func = CRNN.build(input_shape, n_classes, train=True)
    model.summary()

    # 保存模型图
    plot_model(model, to_file=os.path.join(OUTPUT_DIR, 'model.png'), show_shapes=True)

    # 可视化回调，负责保存模型并画图显示训练进度
    viz_cb = cbk.VizCallback(img_gen, OUTPUT_DIR, test_func, img_gen.val_batch_gen(), start_epoch, n_gpus)

    # 优化器，clipnorm限制梯度上限，避免遇到梯度悬崖
    # sgd = SGD(lr=lr, decay=lr/(stop_epoch-start_epoch), momentum=0.9, nesterov=True, clipnorm=10)
    sgd = Adam(lr=lr)

    # 加载模型权重checkpoint继续训练
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, 'weights{}.h5'.format(start_epoch - 1))
        model.load_weights(weight_file)
    if n_gpus > 1:
        # 多GPU
        model = multi_gpu_model(model, gpus=n_gpus)


    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=["accuracy"])
    model.fit_generator(
        generator=img_gen.train_batch_gen(),
        steps_per_epoch=cfg.train_size // cfg.batch_size,
        epochs=stop_epoch,
        validation_data=img_gen.val_batch_gen(),
        validation_steps=cfg.val_size // cfg.batch_size,
        callbacks=[viz_cb, img_gen],
        initial_epoch=start_epoch,
        max_queue_size=cfg.batch_size*2)


if __name__ == '__main__':
    # 输出路径
    np.random.seed(55)
    run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    start_time = time.time()
    OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, run_name)

    # 自定义训练图像宽度，训练stop_epoch-start_epoch次
    train(OUTPUT_DIR, start_epoch=0, stop_epoch=20, img_w=128, lr=1e-4, n_gpus=2)
    # 保持文本不变，增加宽度，依靠留白实现平移不变性，抑制RNN的语言模型在文本首尾的多余输出
    # train(OUTPUT_DIR, start_epoch=20, stop_epoch=30, img_w=260, lr=1e-5, n_gpus=2)
    # train(OUTPUT_DIR, start_epoch=40, stop_epoch=50, img_w=512, lr=1e-7, n_gpus=2)
    end_time = time.time()
    print("training time : {:.2f} hours".format((end_time - start_time)/3600))
