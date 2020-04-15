from configparser import ConfigParser
import numpy as np
import requests
import os
import cv2
from quickocr.localization import model as localization_model
from quickocr.recognition import model as recognition_model


def get_text(images: list, text_only=False):
    cfg = ConfigParser()
    base_path = os.path.dirname(__file__)
    cfg.read_string(open(os.path.join(base_path, 'data', 'config.txt')).read())

    if not os.path.exists(os.path.join(base_path, 'data', 'recognition_weights.h5')):
        with open(os.path.join(base_path, 'data', 'recognition_weights.h5'), 'wb') as f:
            weights = requests.get(cfg.get('recognition', 'url'))
            f.write(weights.content)

    if not os.path.exists(os.path.join(base_path, 'data', 'localization_weights.h5')):
        with open(os.path.join(base_path, 'data', 'localization_weights.h5'), 'wb') as f:
            weights = requests.get(cfg.get('localization', 'url'))
            f.write(weights.content)
    for image_path in images:
        original_image = cv2.imread(image_path)
        bboxes = localization_model.get_text_localization(original_image, weights=os.path.join(base_path, 'data',
                                                                                               'localization_weights.h5'))
        for bbox in bboxes:
            bbox = np.reshape(bbox, newshape=(-1, 8)).astype(np.int)[0]
            print(bbox)
            clipped_image = original_image[bbox[1]:bbox[5], bbox[0]:bbox[4]]
            if text_only:
                original_image = cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[4], bbox[5]), (255, 0, 0), 2,
                                               1)
            text = recognition_model.get_text(clipped_image,
                                              weights=os.path.join(base_path, 'data', 'recognition_weights.h5'),
                                              label=os.path.join(base_path, 'data', 'label.txt'))
            print(text)


if __name__ == '__main__':
    get_text([r'E:\cdc\Project\Snapnews\opensource\test\1586743108.4626567.jpg'])