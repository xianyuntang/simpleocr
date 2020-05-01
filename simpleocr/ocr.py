from configparser import ConfigParser
import numpy as np
import requests
import os
import cv2
import tqdm
from simpleocr.localization import model as localization_model
from simpleocr.recognition import model as recognition_model


def get_text(images: list, text_only=False, dist_dir='./output'):
    cfg = ConfigParser()
    base_path = os.path.dirname(__file__)
    cfg.read_string(open(os.path.join(base_path, 'data', 'config.txt')).read())

    if not os.path.exists(os.path.join(base_path, 'data', 'recognition_weights.h5')):
        print('download recognition_weights.h5')
        weights = requests.get(cfg.get('recognition', 'url'))
        with open(os.path.join(base_path, 'data', 'recognition_weights.h5'), 'wb') as f:
            for content in tqdm.tqdm(weights.iter_content()):
                f.write(content)

    if not os.path.exists(os.path.join(base_path, 'data', 'localization_weights.h5')):
        print('download localization_weights.h5')
        weights = requests.get(cfg.get('localization', 'url'))
        with open(os.path.join(base_path, 'data', 'localization_weights.h5'), 'wb') as f:
            for content in tqdm.tqdm(weights.iter_content()):
                f.write(content)

    for image_path in images:
        original_image = cv2.imread(image_path)
        bboxes = localization_model.get_text_localization(original_image, weights=os.path.join(base_path, 'data',
                                                                                               'localization_weights.h5'))
        for index, bbox in enumerate(bboxes):
            filename = os.path.split(image_path)
            bbox = np.reshape(bbox, newshape=(-1, 8)).astype(np.int)[0]
            clipped_image = original_image[bbox[1]:bbox[5], bbox[0]:bbox[4]]
            cv2.imwrite(os.path.join(dist_dir, f'{filename}_{index}.jpg'), clipped_image)
            if not text_only:
                original_image = cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[4], bbox[5]), (255, 0, 0), 2,
                                               1)
                cv2.imwrite(os.path.join(dist_dir, f'{filename}.jpg'), original_image)
            text = recognition_model.get_text(clipped_image,
                                              weights=os.path.join(base_path, 'data', 'recognition_weights.h5'),
                                              label=os.path.join(base_path, 'data', 'label.txt'))
            with open(os.path.join(dist_dir, f'{filename}_{index}.txt'), 'w', encoding='utf-8') as f:
                f.write(text)


if __name__ == '__main__':
    get_text([r'E:\cdc\Project\Snapnews\opensource\test\3192342.jpg'])
