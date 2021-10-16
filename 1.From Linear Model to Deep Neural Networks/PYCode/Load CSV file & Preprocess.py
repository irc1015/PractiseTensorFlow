import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pylab as plt
import csv

print(tf.__version__)

#download 5-flowers dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

#count photos of dataset
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

print(data_dir)

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

reshape_dims = [IMG_HEIGHT,IMG_WIDTH]

#read and decode dataset
def read_and_decode(filename, reshape_dims):
  # 1.Read the file
  img = tf.io.read_file(filename)

  # 2.Convert the compressed string to a 3D uint8 tensor.
  img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)

  # 3.Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)

  # 4.Resize the image to the desired size.
  return tf.image.resize(img, reshape_dims)

CLASS_NAMES = [item.numpy().decode("utf-8") for item in
               tf.strings.regex_replace(
                 tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/*"),
                 "/Users/zhuzhirui/.keras/datasets/flower_photos/", "")]
print(CLASS_NAMES)

#save all of elements which have not '.' , ignore all of elements which have '.'
CLASS_NAMES = [item for item in CLASS_NAMES if item.find(".") == -1]
print(CLASS_NAMES)

def show_image(filepath):
    img = read_and_decode(filepath,[IMG_HEIGHT,IMG_WIDTH])
    #image --> TensorFlow tensor

    plt.imshow(img.numpy())

roses = list(data_dir.glob('roses/*'))
print(roses[5])
show_image(str(roses[5]))

#display 5 images in a row
tulips = tf.io.gfile.glob(str(data_dir)+'/tulips/*.jpg')
f,ax = plt.subplots(1,5,figsize=(15,15))
for idx, filepath in enumerate(tulips[:5]):
    print(filepath)
    img = read_and_decode(filepath, [IMG_HEIGHT, IMG_WIDTH])
    ax[idx].imshow(img.numpy())
    ax[idx].axis('off')
    #display each size of images

#display 5 images in a raw with labels
dandelion = tf.io.gfile.glob(str(data_dir)+'/dandelion/*.jpg')
f,ax = plt.subplots(1,5,figsize=(15,15))
for idx,filepath in enumerate(dandelion[10:15]):
    print(filepath)
    img = read_and_decode(filepath, [IMG_HEIGHT, IMG_WIDTH])
    ax[idx].imshow(img.numpy())
    ax[idx].set_title('dandelionNO:{}'.format(idx))
    ax[idx].axis('off')

#extract the label
basename = tf.strings.regex_replace(
    tf.io.gfile.glob(str(data_dir) + '/*'),str(data_dir),''
)
print(basename)
label = tf.strings.split(basename,'/')
print(label)

#decode each line of CSV file : img->tensor by path, and extract the label
def decode_csv(csv_row):
  record_defaults = ["path", "flower"]
  rowpath, label_string = tf.io.decode_csv(csv_row,record_defaults)
  img = read_and_decode(rowpath,reshape_dims)
  #label = tf.math.equal(CLASS_NAMES, label_string)
  return img, label_string

#read each line of CSV file by decode function
#return a tuple
dataset = (tf.data.TextLineDataset(
    str(data_dir) + '/train_set.csv').map(decode_csv))

#show first 3 samples of dataset
for img, label in dataset.take(3):
  avg = tf.math.reduce_mean(img, axis=[0,1]) # average pixel in the image
  print(label, avg)
