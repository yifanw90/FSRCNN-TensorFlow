# FSRCNN-Tensorflow
TensorFlow implementation of the Fast Super-Resolution Convolutional Neural Network (FSRCNN). This implements two models, FSRCNN which is more accurate and FSRCNN-s which is faster (approaches real-time performance). Based on this [project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).

## Prerequisites
 * TensorFlow
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * h5py
 * PIL

## Usage
For training , `python main.py`
Can specify epochs, learning rate,  `python main.py --epochs 10`
<br>
For testing, `python main.py --is_train False`

To use FSCRNN-s over FSCRNN , `python main.py --fast True`

Includes script expand_data.py which scales and rotates all the images in your training set to expand your dataset just like in the paper
`python expand_data.py Train`

## Result
After training 15,000 epochs, I got similar super-resolved image to reference paper. Training time takes 12 hours 16 minutes and 1.41 seconds. My desktop performance is Intel I7-6700 CPU, GTX970, and 16GB RAM. Result images are shown below.<br><br>
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
