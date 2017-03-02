# FSRCNN-TensorFlow
TensorFlow implementation of the Fast Super-Resolution Convolutional Neural Network (FSRCNN). This implements two models: FSRCNN which is more accurate but slower and FSRCNN-s which is faster but less accurate. Based on this [project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).

## Prerequisites
 * Python 2.7
 * TensorFlow
 * Scipy version > 0.18
 * h5py
 * PIL

## Usage
For training: `python main.py`
<br>
Can specify epochs, learning rate, data directory, etc:
<br>
`python main.py --epochs 10 --learning_rate 0.0001 --data_dir Train`
<br>
For testing: `python main.py --is_train False`

To use FSCRNN-s instead of FSCRNN: `python main.py --fast True`

Includes script `expand_data.py` which scales and rotates all the images in your training set to expand it:
<br>
`python expand_data.py Train`

## Result

<br><br>
Original butterfly image:
![orig](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/orig.png)<br>
Bicubic interpolated image:
![bicubic](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/bicubic.png)<br>
Super-resolved image:
![srcnn](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/srcnn.png)

## References
* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)
<br>
* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 
<br>
* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
