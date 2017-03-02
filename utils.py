"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
from math import floor
import struct

import tensorflow as tf
from PIL import Image  
from scipy.misc import imread
import numpy as np
from multiprocessing import Pool, Lock, active_children

import pdb

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Downsampled by scale factor
  """

  image = Image.open(path).convert('L')
  (width, height) = image.size
  label_ = np.array(list(image.getdata())).astype(np.float).reshape((height, width)) / 255
  image.close()

  cropped_image = Image.fromarray(modcrop(label_, scale))
  
  (width, height) = cropped_image.size
  new_width, new_height = int(width / scale), int(height / scale)
  scaled_image = cropped_image.resize((new_width, new_height), Image.ANTIALIAS)
  cropped_image.close()

  (width, height) = scaled_image.size
  input_ = np.array(list(scaled_image.getdata())).astype(np.float).reshape((height, width))

  return input_, label_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
  data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))

  return data

def make_data(sess, checkpoint_dir, data, label):
  """
  Make input data as h5 file format
  Depending on 'train' (flag value), savepath would be changed.
  """
  if FLAGS.train:
    savepath = os.path.join(os.getcwd(), '{}/train.h5'.format(checkpoint_dir))
  else:
    savepath = os.path.join(os.getcwd(), '{}/test.h5'.format(checkpoint_dir))

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def image_read(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def train_input_worker(args):
  image_data, config = args
  image_size, label_size, stride, scale, save_image = config

  single_input_sequence, single_label_sequence = [], []
  padding = abs(image_size - label_size) / 2 # (21 - 11) / 2 = 5
  label_padding = label_size / scale # 21 / 3 = 7

  input_, label_ = preprocess(image_data, scale)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  for x in range(0, h - image_size - padding + 1, stride):
    for y in range(0, w - image_size - padding + 1, stride):
      sub_input = input_[x + padding : x + padding + image_size, y + padding : y + padding + image_size]
      sub_label = label_[(x + label_padding) * scale : (x + label_padding) * scale + label_size, (y + label_padding) * scale : (y + label_padding) * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      single_input_sequence.append(sub_input)
      single_label_sequence.append(sub_label)

  return [single_input_sequence, single_label_sequence]

def thread_train_setup(config):
  sess = config.sess

  # Load data path
  data = prepare_data(sess, dataset=config.data_dir)

  # Initialize multiprocessing pool with # of processes = config.threads
  pool = Pool(config.threads)

  # Distribute images_per_thread images across each worker
  config_values = [config.image_size, config.label_size, config.stride, config.scale, config.save_image]
  images_per_thread = len(data) / config.threads
  workers = []
  for thread in range(config.threads):
    args_list = [(data[i], config_values) for i in range(thread * images_per_thread, (thread + 1) * images_per_thread)]
    worker = pool.map_async(train_input_worker, args_list)
    workers.append(worker)
  print("{} worker processes created".format(config.threads))

  pool.close()

  results = []
  for i in range(len(workers)):
    print("Waiting for worker process {}".format(i))
    results.extend(workers[i].get(timeout=240))
    print("Worker process {} done".format(i))

  print("All worker processes done!")

  sub_input_sequence, sub_label_sequence = [], []

  for image in range(len(results)):
    single_input_sequence, single_label_sequence = results[image]
    sub_input_sequence.extend(single_input_sequence)
    sub_label_sequence.extend(single_label_sequence)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)


def train_input_setup(config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale = config.image_size, config.label_size, config.stride, config.scale

  # Load data path
  data = prepare_data(sess, dataset=config.data_dir)

  sub_input_sequence, sub_label_sequence = [], []
  padding = abs(image_size - label_size) / 2 # (21 - 11) / 2 = 5
  label_padding = label_size / scale # 21 / 3 = 7

  for i in xrange(len(data)):
    input_, label_ = preprocess(data[i], scale)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    for x in range(0, h - image_size - padding + 1, stride):
      for y in range(0, w - image_size - padding + 1, stride):
        sub_input = input_[x + padding : x + padding + image_size, y + padding : y + padding + image_size]
        sub_label = label_[(x + label_padding) * scale : (x + label_padding) * scale + label_size, (y + label_padding) * scale : (y + label_padding) * scale + label_size]

        sub_input = sub_input.reshape([image_size, image_size, 1])
        sub_label = sub_label.reshape([label_size, label_size, 1])
        
        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)


def test_input_setup(config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale = config.image_size, config.label_size, config.stride, config.scale

  # Load data path
  data = prepare_data(sess, dataset="Test")

  sub_input_sequence, sub_label_sequence = [], []
  padding = abs(image_size - label_size) / 2 # (21 - 11) / 2 = 5
  label_padding = label_size / scale # 21 / 3 = 7

  pic_index = 2 # Index of image based on lexicographical order in data folder
  input_, label_ = preprocess(data[pic_index], config.scale)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  nx, ny = 0, 0
  for x in range(0, h - image_size - padding + 1, stride):
    nx += 1
    ny = 0
    for y in range(0, w - image_size - padding + 1, stride):
      ny += 1
      sub_input = input_[x + padding : x + padding + image_size, y + padding : y + padding + image_size]
      sub_label = label_[(x + label_padding) * scale : (x + label_padding) * scale + label_size, (y + label_padding) * scale : (y + label_padding) * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      sub_input_sequence.append(sub_input)
      sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  make_data(sess, config.checkpoint_dir, arrdata, arrlabel)

  return nx, ny

def save_params(sess, weights, biases):
  param_dir = "params/"

  if not os.path.exists(param_dir):
    os.makedirs(param_dir)

  weight_file = open(param_dir + "weights", 'wb')
  for layer in weights:
    layer_weights = sess.run(weights[layer])

    for filter_x in range(len(layer_weights)):
      for filter_y in range(len(layer_weights[filter_x])):
        filter_weights = layer_weights[filter_x][filter_y]
        for input_channel in range(len(filter_weights)):
          for output_channel in range(len(filter_weights[input_channel])):
            weight_value = filter_weights[input_channel][output_channel]
            weight_file.write(struct.pack("f", weight_value))
          weight_file.write(struct.pack("x"))

    weight_file.write("\n\n")
  weight_file.close()

  bias_file = open(param_dir + "biases.txt", 'w')
  for layer in biases:
    bias_file.write("Layer {}\n".format(layer))
    layer_biases = sess.run(biases[layer])
    for bias in layer_biases:
      bias_file.write("{}, ".format(bias))
    bias_file.write("\n\n")

  bias_file.close()

# Merges sub-images back into original image size
def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

# Converts array to image and saves it
def array_image_save(array, image_path):
  image = Image.fromarray(array)
  if image.mode != 'RGB':
    image = image.convert('RGB')
  image.save(image_path)
  print("Saved image: {}".format(image_path))
