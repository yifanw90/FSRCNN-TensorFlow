import os
import sys
import glob
import numpy as np
from PIL import Image
import pdb

# Artifically expands the dataset by a factor of 19 by scaling and then rotating every image
def main():
  if len(sys.argv) == 2:
    data = prepare_data(sys.argv[1])
  else:
    print("Missing argument: You must specify a folder with images to expand")
    return

  for i in xrange(len(data)):
    scale(data[i])
    rotate(data[i])

def prepare_data(dataset):
  filenames = os.listdir(dataset)
  data_dir = os.path.join(os.getcwd(), dataset)
  data = glob.glob(os.path.join(data_dir, "*.bmp"))

  return data

def scale(file):
  image = Image.open(file)
  width, height = image.size

  scales = [0.9, 0.8, 0.7, 0.6]
  for scale in scales:
    new_width, new_height = int(width * scale), int(height * scale)
    new_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    new_path = '{}-{}.bmp'.format(file[:-4], scale)
    new_image.save(new_path)

def rotate(file):
  image = Image.open(file)

  rotations = [90, 180, 270]
  for rotation in rotations:
    new_image = image.rotate(rotation, expand=True)
    new_path = '{}-{}.bmp'.format(file[:-4], rotation)
    new_image.save(new_path)

if __name__ == '__main__':
  main()