import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
import numpy as np
import IPython

image_feature_description = {
    'image_width':tf.io.FixedLenFeature([],tf.int64),
    'image_height':tf.io.FixedLenFeature([],tf.int64),
    'image_depth':tf.io.FixedLenFeature([],tf.int64),
    'label':tf.io.FixedLenFeature([],tf.int64),
    'image':tf.io.FixedLenFeature([],tf.string)
}

def parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

Training_Set = 'Training_Set.tfrecord'
Evaluation_Set = 'Evaluation_Set.tfrecord'

raw_image_dataset = tf.data.TFRecordDataset(Training_Set)
train_dataset = raw_image_dataset.map(parse_image_function)

raw_image_dataset = tf.data.TFRecordDataset(Evaluation_Set)
eval_dataset = raw_image_dataset.map(parse_image_function)

for image_features in train_dataset.take(3):
  image = image_features['image'].numpy()
  label = image_features['label']
  image_width = image_features['image_width']
  image_height = image_features['image_height']
  print(label)
  print('{} * {}'.format(image_width,image_height))
  IPython.display.display(IPython.display.Image(data=image))
