# Simpleocr library
simpleocr is a python traditional chinese OCR package based on deep learning method.

The library consists of text localization and text recognition.

## Text localization
The model is the reimplementation of CRAFT(Character-Region Awareness For Text detection) by tensorflow.

[paper](https://arxiv.org/abs/1904.01941) | [github](https://github.com/clovaai/CRAFT-pytorch)
 
## Text recognition
The reimplementation is based on CRNN model that RNN layer is replaced with self-attention layer.

##### CRNN
[paper](https://arxiv.org/abs/1707.03985)

##### Self attention

[paper](https://arxiv.org/abs/1706.03762)

# Installation
```
$ pip install simpleocr
```
or 
```
$ git clone https://github.com/xianyuntang/simpleocr
$ cd simpleocr
$ python setup.py install
```
# Usage
```
from simpleocr import ocr
ocr.get_text(['image.jpg'])
```



# TODO
1. English support
2. GPU support