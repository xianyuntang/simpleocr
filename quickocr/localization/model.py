import tensorflow as tf
from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, \
    BatchNormalization, UpSampling2D, concatenate, Lambda, ZeroPadding2D
from tensorflow.python.keras.models import Model
import math


class VGGNetModel(Model):
    def __init__(self):
        super(VGGNetModel, self).__init__()
        # block 1
        self.block1_conv1 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())

        self.block1_conv2 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        self.block1_batch_norm = BatchNormalization(name='block1_batch_norm')

        # block 2
        self.block2_conv1 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block2_conv2 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        self.block2_batch_norm = BatchNormalization(name='block2_batch_norm')

        # Block 3
        self.block3_conv1 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block3_conv2 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block3_conv3 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
        self.block3_batch_norm = BatchNormalization(name='block3_batch_norm')

        # Block 4
        self.block4_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block4_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block4_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
        self.block4_batch_norm = BatchNormalization(name='block4_batch_norm')

        # Block 5
        self.blcok5_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block5_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block5_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())

        self.block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
        self.block5_batch_norm = BatchNormalization(name='block5_batch_norm')

        # Block 6
        self.block6_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block6_conv1',
                                   dilation_rate=6,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block6_conv2 = Conv2D(512, (1, 1),
                                   activation='relu',
                                   padding='same',
                                   name='block6_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())

        self.block6_batch_norm = BatchNormalization(name='block6_batch_norm')

        # block 7
        self.block7_up_conv1 = Conv2D(512, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block7_up_conv1',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block7_batch_norm1 = BatchNormalization(name='block7_batch_norm1')
        self.block7_up_conv2 = Conv2D(256, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block7_up_conv2',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block7_batch_norm2 = BatchNormalization(name='block7_batch_norm2')
        self.block7_up_sampling = UpSampling2D(name='block7_up_sampling')
        # block 8
        self.block8_up_conv1 = Conv2D(256, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block8_up_conv1',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block8_batch_norm1 = BatchNormalization(name='block8_batch_norm1')
        self.block8_up_conv2 = Conv2D(128, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block8_up_conv2',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block8_batch_norm2 = BatchNormalization(name='block8_batch_norm2')
        self.block8_up_sampling = UpSampling2D(name='block8_up_sampling')

        # block 9
        self.block9_up_conv1 = Conv2D(128, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block9_up_conv1',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block9_batch_norm1 = BatchNormalization(name='block9_batch_norm1')
        self.block9_up_conv2 = Conv2D(64, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block9_up_conv2',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block9_batch_norm2 = BatchNormalization(name='block9_batch_norm2')
        self.block9_up_sampling = UpSampling2D(name='block9_up_sampling')

        # block 10
        self.block10_up_conv1 = Conv2D(64, (3, 3),
                                       activation='relu',
                                       padding='same',
                                       name='block10_up_conv1',
                                       kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                       kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block10_batch_norm1 = BatchNormalization(name='block10_batch_norm1')
        self.block10_up_conv2 = Conv2D(32, (3, 3),
                                       activation='relu',
                                       padding='same',
                                       name='block10_up_conv2',
                                       kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                       kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block10_batch_norm2 = BatchNormalization(name='block10_batch_norm2')
        self.block10_up_sampling = UpSampling2D(name='block10_up_sampling')

        # block 11
        self.block11_conv1 = Conv2D(32, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv1',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv2 = Conv2D(32, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv2',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv3 = Conv2D(16, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv3',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv4 = Conv2D(16, (1, 1),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv4',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv5 = Conv2D(2, (1, 1),
                                    activation='sigmoid',
                                    padding='same',
                                    name='block11_conv5',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())

        self.padding = ZeroPadding2D(padding=((1, 0), (0, 0)), data_format='channels_last')

    def call(self, inputs, training=None, mask=None, preference=False):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)
        x = self.block1_batch_norm(x)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)
        x_2 = self.block2_batch_norm(x)
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)
        x_3 = self.block3_batch_norm(x)
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)
        x_4 = self.block4_batch_norm(x)
        x = self.blcok5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)
        x_5 = self.block5_batch_norm(x)
        x = self.block6_conv1(x)
        x = self.block6_conv2(x)
        x = self.block6_batch_norm(x)
        x = concatenate([x_5, x])
        x = self.block7_up_conv1(x)
        x = self.block7_batch_norm1(x)
        x = self.block7_up_conv2(x)
        x = self.block7_batch_norm2(x)
        x = self.block7_up_sampling(x)
        if x_4.shape[1] == 67:
            x = self.padding(x)

        x = concatenate([x_4, x])
        x = self.block8_up_conv1(x)
        x = self.block8_batch_norm1(x)
        x = self.block8_up_conv2(x)
        x = self.block8_batch_norm2(x)
        x = self.block8_up_sampling(x)
        if x_3.shape[1] == 135:
            x = self.padding(x)
        x = concatenate([x_3, x])
        x = self.block9_up_conv1(x)
        x = self.block9_batch_norm1(x)
        x = self.block9_up_conv2(x)
        x = self.block9_batch_norm2(x)
        x = self.block9_up_sampling(x)

        x = concatenate([x_2, x])
        x = self.block10_up_conv1(x)
        x = self.block10_batch_norm1(x)
        x = self.block10_up_conv2(x)
        x = self.block10_batch_norm2(x)
        x = self.block10_up_sampling(x)

        x = self.block11_conv1(x)
        x = self.block11_conv2(x)
        x = self.block11_conv3(x)
        x = self.block11_conv4(x)
        x = self.block11_conv5(x)

        region_score = Lambda(lambda layer: layer[..., 0])(x)
        affinity_score = Lambda(lambda layer: layer[..., 1])(x)
        return region_score, affinity_score


def get_det_boxes_core(text_map, link_map, link_threshold, low_text):
    link_map = np.squeeze(link_map)
    text_map = np.squeeze(text_map)
    link_map = link_map.copy()
    text_map = text_map.copy()
    img_h, img_w = text_map.shape

    ret, text_score = cv2.threshold(text_map, low_text, 1, 0)
    ret, link_score = cv2.threshold(link_map, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                         connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):

        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        segmap = np.zeros(text_map.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def get_det_boxes(text_map, link_map, link_threshold, low_text):
    boxes, labels, mapper = get_det_boxes_core(text_map, link_map, link_threshold, low_text)

    return boxes


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def get_box(region_score, affinity_map, min_thresh=25, max_thresh=100000):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    region_score = np.squeeze((region_score * 255).astype(np.uint8))
    affinity_map = np.squeeze((affinity_map * 255).astype(np.uint8))
    _, region_score = cv2.threshold(region_score, 100, 255, 0)

    _, affinity_map = cv2.threshold(affinity_map, 100, 255, 0)
    text_map = region_score + affinity_map
    text_map = np.clip(text_map, 0, 255)

    mix = cv2.morphologyEx(text_map, cv2.MORPH_CLOSE, kernel)

    output, _, cord, _ = cv2.connectedComponentsWithStats(mix, 4, cv2.CV_32S)
    boxes = []
    for i in range(output):
        if min_thresh <= cord[i][4] <= max_thresh:
            boxes.append([cord[i][0] * 3, cord[i][1] * 2.8125, (cord[i][0] + cord[i][2]) * 3,
                          (cord[i][1] + cord[i][3]) * 2.8125])
    return boxes


if __name__ == '__main__':
    import numpy as np
    import cv2

    model = VGGNetModel()
    data = np.random.random((1, 768, 768, 3)).astype(np.float32)
    a, b = model(data)
