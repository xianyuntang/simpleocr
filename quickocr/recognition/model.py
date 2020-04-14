import tensorflow as tf
from tensorflow.python.keras.layers import Dense, MaxPooling2D, Conv2D, \
    Reshape, BatchNormalization, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import cv2
import numpy as np
import os


class SelfAttention(Layer):
    def __init__(self, units=512, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feature_dim = input_shape[2]
        self.Wq = self.add_weight(name=f'{self.name}_q', shape=(feature_dim, self.units)
                                  )
        self.Wk = self.add_weight(name=f'{self.name}_k', shape=(feature_dim, self.units)
                                  )
        self.Wv = self.add_weight(name=f'{self.name}_v', shape=(feature_dim, 1))

        self.bh = self.add_weight(name=f'{self.name}_bh', shape=(self.units,))

        self.ba = self.add_weight(name=f'{self.name}_ba', shape=(1,))

    def call(self, inputs, **kwargs):
        batch_size, input_len, _ = inputs.shape
        q = K.expand_dims(K.dot(inputs, self.Wq), 2)
        k = K.expand_dims(K.dot(inputs, self.Wk), 1)
        h = tf.tanh(q + k + self.bh)

        e = K.dot(h, self.Wv) + self.ba
        # e = K.reshape(e, shape=(batch_size, input_len, input_len))
        e = tf.reshape(e, shape=(batch_size, input_len, input_len))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())
        v = K.batch_dot(a, inputs)
        return v


class VGGNetModel(Model):
    def __init__(self, output_size):
        super(VGGNetModel, self).__init__()
        # block 1
        self.block1_conv1 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2())

        self.block1_conv2 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        self.block1_batch_norm = BatchNormalization(name='block1_batch_norm')

        # block 2
        self.block2_conv1 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block2_conv2 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        self.block2_batch_norm = BatchNormalization(name='block2_batch_norm')

        # Block 3
        self.block3_conv1 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block3_conv2 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block3_conv3 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block3_pool = MaxPooling2D((1, 2), strides=(1, 2), name='block3_pool')
        self.block3_batch_norm = BatchNormalization(name='block3_batch_norm')

        # Block 4
        self.block4_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block4_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block4_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block4_pool = MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool')
        self.block4_batch_norm = BatchNormalization(name='block4_batch_norm')

        # Block 5
        self.blcok5_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block5_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2())
        self.block5_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2())

        self.block5_pool = MaxPooling2D((1, 2), strides=(1, 2), name='block5_pool')
        self.block5_batch_norm = BatchNormalization(name='block5_batch_norm')

        # Block 6
        self.block6_reshape = Reshape(target_shape=(-1, 512))

        self.self_attention = SelfAttention(units=512, dynamic=True, name='attention')
        #
        self.prediction = Dense(units=output_size, kernel_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(), name='prediction')

    def call(self, inputs, training=None, mask=None, preference=False):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)
        x = self.block1_batch_norm(x)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)
        x = self.block2_batch_norm(x)
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)
        x = self.block3_batch_norm(x)
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)
        x = self.block4_batch_norm(x)
        x = self.blcok5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)
        x = self.block5_batch_norm(x)
        x = self.block6_reshape(x)
        x = self.self_attention(x)
        x = self.prediction(x)

        return x


def labels_to_text(label, charset: list):
    ret = []
    for batch_label in label:

        tmp = []
        for l in batch_label:
            l = int(l)
            if l == -1:  # CTC Blank
                tmp.append('')
            else:
                tmp.append(charset[l])

        ret.append(''.join(tmp))
    return ret


def decode_predict_ctc(out, charset):
    out = tf.cast(out, dtype=tf.float32)
    out = tf.transpose(out, (1, 0, 2))
    decoded = tf.nn.ctc_greedy_decoder(out, sequence_length=tf.cast(tf.ones(out.shape[1]) * out.shape[0],
                                                                    dtype=tf.int32))[0]
    labels = [
        tf.sparse.to_dense(
            st, default_value=-1)
        for st in decoded
    ][0]
    text = labels_to_text(labels, charset)

    return text


def load_charset(charset_file):
    charset = u''
    # print("####### loading character ######")
    with open(charset_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            charset += line.replace('\n', '')
    return charset


def get_text(image):
    K.clear_session()
    K.set_learning_phase(0)
    charset = load_charset(os.path.join('.', 'quickocr', 'weights', 'label.txt'))
    ocr_model = VGGNetModel(output_size=len(charset) + 1)
    ocr_model(np.random.random(size=(1, 80, 32, 1)).astype(np.float32))
    ocr_model.load_weights(os.path.join('.', 'quickocr', 'weights', 'recognition_weights.h5'), by_name=True)
    height, width, depth = image.shape
    # 直的
    if height > width:
        new_height = int(32 * (height / width))
        image = cv2.resize(image, (32, new_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
        image = image.astype(np.float32)
        image = image / 255
    else:
        new_width = int(width / height * 32)
        image = cv2.resize(image, (new_width, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
        image = image.astype(np.float32)
        image = image / 255

        image = image.T
    image = np.expand_dims(image, 0)

    image = np.expand_dims(image, -1)

    predictions = ocr_model(image)
    return decode_predict_ctc(predictions, charset)[0]
