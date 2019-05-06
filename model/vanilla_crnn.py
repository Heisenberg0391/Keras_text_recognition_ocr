'''
CRNN模型结构
'''
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Activation, ZeroPadding2D
from keras.layers import Reshape, Flatten, TimeDistributed, Permute, Bidirectional
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU, LSTM
from keras.models import Model
import keras.backend as K
from keras.layers import Lambda
import mycrnn_pc.config.cfg as cfg
from keras.regularizers import l2


def ctc_lambda_func(args):
    '''
    # the actual loss calc occurs here despite it not being
    # an internal Keras loss function
    # Arguments
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.

    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    '''
    y_pred, labels, input_length, label_length = args
    # RNN前两项没用，y_pred是softmax的输出，要保证y_pred长度小于输入长度
    # fixme 20190301 原版crnn不去除前两项
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class CRNN:
    @ staticmethod
    def build(input_shape, n_classes, train=True, reg=0.01):
        '''
        input就是generator每次yield的
        inputs = {'the_input': train_batch,  # 样本图像批
                  'the_labels': labels,  # 样本类别序列批
                  'input_length': input_length,  # RNN输入长度批
                  'label_length': label_length}  # 类别序列长度批
        '''
        if K.image_data_format() == "channels_first":
            chanDim = 1
        else:
            chanDim = -1
        # input: (h, w, n_channels), kernel: (h, w)
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1_1',
                   kernel_regularizer=l2(reg))(input_data)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
        x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2_1',
                   kernel_regularizer=l2(reg))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
        x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3_1',
                   kernel_regularizer=l2(reg))(x)
        x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3_2',
                   kernel_regularizer=l2(reg))(x)
        x = ZeroPadding2D(padding=(0, 1), name='pad1')(x)  # 只补宽度，不补高度
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), name='pool3')(x)
        x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv4_1',
                   kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, name='bn1')(x)
        x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5_1',
                   kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, name='bn2')(x)
        x = ZeroPadding2D(padding=(0, 1), name='pad2')(x)  # 只补宽度，不补高度
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), name='pool4')(x)
        x = Conv2D(512, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='valid', name='conv7',
                   kernel_regularizer=l2(reg))(x)

        # 最后一层的尺寸：(高, 宽, 深), POOL和conv的补边、步长决定宽高，kernel数量决定深度
        shape = x.get_shape()
        # conv_to_rnn_dims = (int(shape[1]), int(shape[2]) * int(shape[3]))

        # CONV模块最后的特征图的每一列作为RNN输入序列的一元
        # cnn_feature = Reshape(target_shape=conv_to_rnn_dims, name='map2seq')(x)
        x = Permute((2, 1, 3))(x)
        x = TimeDistributed(Flatten(), name='timedistrib')(x)
        # 2层双向RNN
        # x = Bidirectional(GRU(256, return_sequences=True, implementation=2), name='bi-lstm1')(cnn_out)
        #x = Dense(int(shape[1]) * int(shape[3]), name='bi-lstm1_out')(x)
        # 第2层可能导致过拟合
        #rnn_out = Bidirectional(GRU(256, return_sequences=True, implementation=2), name='bi-lstm2')(x)
        rnn_f = LSTM(256, return_sequences=True, name='rnn1_f')(x)
        rnn_b = LSTM(256, return_sequences=True, go_backwards=True, name='rnn1_b')(x)
        x = concatenate([rnn_f, rnn_b])
        rnn_f = LSTM(256, return_sequences=True, name='rnn2_f')(x)
        rnn_b = LSTM(256, return_sequences=True, go_backwards=True, name='rnn2_b')(x)
        rnn_out = concatenate([rnn_f, rnn_b])

        # 全连接神经元数量=字符类别数+1(+1 for blank token)
        x = Dense(n_classes, name='dense')(rnn_out)
        # softmax层
        y_pred = Activation('softmax', name='softmax')(x)
        # CTC的输入序列长度和及其对应的类别序列长度
        # 原始序列长度必须小于等于CTC的输出序列长度，保证每个输入时刻最多对应一个类别
        # input_length是y_pred的长度，即送入ctc的长度也就是卷积层最后的宽度
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        labels = Input(name='the_labels', shape=[cfg.max_label_len], dtype='float32')

        # ctc层
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        if train == True:
            # 训练时需要labels, input_length, label_length计算ctc损失
            model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        else:
            # 测试时只需要输入数据和预测输出
            model = Model(inputs=input_data, outputs=y_pred)

        """
        # 获取softmax层的输出，在可视化过程中用于解码验证，代替model.predict()
        # inputs: List of placeholder tensors.
        # outputs: List of output tensors.
        """
        test_func = K.function([input_data], [y_pred])  # [input_data]是tensor input_data的list
        return model, y_pred, test_func
