import numpy as np
from quickocr.localization.model import VGGNetModel, get_det_boxes, adjust_result_coordinates
import cv2
from tensorflow.python.keras import backend as K


def text_localization(images: list):
    w = 1280
    h = 768
    K.clear_session()
    K.set_learning_phase(0)
    model = VGGNetModel()
    model(np.random.random((1, h, w, 1)).astype(np.float32))
    model.load_weights(os.path.join('.', 'quickocr', 'weights', 'text_localization'), by_name=True)
    for image_p in images:
        original_image = cv2.imread(image_p)

        target_ratio_h = original_image.shape[0] / h
        target_ratio_w = original_image.shape[1] / w
        inference_image = original_image.copy()
        inference_image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2GRAY)
        inference_image = inference_image / 255
        inference_image = cv2.resize(inference_image, dsize=(w, h))
        inference_image = np.expand_dims(inference_image, -1)
        inference_image = np.expand_dims(inference_image, 0)
        inference_image = inference_image.astype(np.float32)

        score_text, score_link = model(inference_image)

        bboxes = get_det_boxes(score_text.numpy(), score_link.numpy(), 0.35, 0.35)
        bboxes = adjust_result_coordinates(bboxes, target_ratio_w, target_ratio_h)
        for bbox in bboxes:
            bbox = np.reshape(bbox, newshape=(-1, 8)).astype(np.int)[0]
            print(bbox)
            clipped_image = original_image[bbox[1]:bbox[5], bbox[0]:bbox[4]]
            original_image = cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[4], bbox[5]), (255, 0, 0), 2, 1)

        cv2.imshow('t', original_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    predict()
