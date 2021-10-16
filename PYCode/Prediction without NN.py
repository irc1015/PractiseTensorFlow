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

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
reshape_dims = [IMG_HEIGHT,IMG_WIDTH]

CLASS_NAMES = ['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips']

def read_and_decode(filename, reshape_dims):
  # 1.Read the file
  img = tf.io.read_file(filename)

  # 2.Convert the compressed string to a 3D uint8 tensor.
  img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)

  # 3.Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)

  # 4.Resize the image to the desired size.
  return tf.image.resize(img, reshape_dims)

def decode_csv(csv_row):
  record_defaults = ["path", "flower"]
  rowpath, label_string = tf.io.decode_csv(csv_row,record_defaults)
  img = read_and_decode(rowpath,reshape_dims)
  #label = tf.math.equal(CLASS_NAMES, label_string)
  return img, label_string

class Centroid:
    def __init__(self, label):
        self.label = label
        self.sum_so_far = tf.constant(0., dtype=tf.float32)
        self.count_so_far = 0

    def update(self, value):
        self.sum_so_far = self.sum_so_far + value
        self.count_so_far = self.count_so_far + 1
        if self.count_so_far % 100 == 0:
            print(self.label, self.count_so_far)

    def centroid(self):
        return self.sum_so_far / self.count_so_far

    def __str__(self):
        return '{} {}'.format(self.label, self.centroid().numpy())


class CentroidRule:
    def __init__(self):
        self.centroids = {
            f: Centroid(f) for f in CLASS_NAMES}

    def fit(self, dataset):
        for img, label in dataset:
            label = label.numpy().decode("utf-8")
            avg = tf.reduce_mean(img, axis=[0, 1])  # average pixel in the image
            self.centroids[label].update(avg)

    def predict(self, img):
        avg = tf.reduce_mean(img, axis=[0, 1])  # average pixel in the image
        best_label = ""
        best_diff = 999
        for key, val in self.centroids.items():
            diff = tf.reduce_sum(tf.abs(avg - val.centroid()))
            if diff < best_diff:
                best_diff = best_diff
                best_label = key
        return best_label

    def evaluate(self, dataset):
        num_correct, total_images = 0, 0
        for img, label in dataset:
            correct = label.numpy().decode('utf-8')
            predicted = self.predict(img)
            if correct == predicted:
                num_correct = num_correct + 1
            total_images = total_images + 1
        accuracy = num_correct / total_images
        return (accuracy)

rule = CentroidRule()

train_dataset = (tf.data.TextLineDataset('/Users/zhuzhirui/.keras/datasets/flower_photos/train_set.csv')
                 .map(decode_csv))

eval_dataset = (tf.data.TextLineDataset("/Users/zhuzhirui/.keras/datasets/flower_photos/eval_set.csv")
                .map(decode_csv))

rule.fit(train_dataset)

print(rule.centroids['daisy'])
print(rule.centroids['roses'])
print(rule.centroids['sunflowers'])
print(rule.centroids['dandelion'])
print(rule.centroids['tulips'])

print(rule.evaluate(eval_dataset))

#How to predict a sample
#/Users/zhuzhirui/.keras/datasets/flower_photos/roses/3145692843_d46ba4703c.jpg,roses
filepath = '/Users/zhuzhirui/.keras/datasets/flower_photos/roses/3145692843_d46ba4703c.jpg'
img = read_and_decode(filepath,[IMG_HEIGHT,IMG_WIDTH])
pred = rule.predict(img)
print(pred)
