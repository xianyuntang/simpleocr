# simple-ocr library
simple-ocr is a OCR tool for traditional chinese based on deep learning method.

The library consist of text localization and text recognition.

## text localization
The model is the reimplementation of CRAFT(Character-Region Awareness For Text detection) by tensorflow.

[paper](https://arxiv.org/abs/1904.01941) | [github](https://github.com/clovaai/CRAFT-pytorch)
 
## text recognition
The reimplementation is based on CRNN model which RNN layer is replaced with self-attention layer.

##### CRNN
[paper](https://arxiv.org/abs/1707.03985)

##### self attention

[paper](https://arxiv.org/abs/1706.03762)

# installation
```
$ pip install quickocr
```

# usage
```
from quickocr import quickocr
quickocr.get_text(['image.jpg'])
```



# TODO
1. English support