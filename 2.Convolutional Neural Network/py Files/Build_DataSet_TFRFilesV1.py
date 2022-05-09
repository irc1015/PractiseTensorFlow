#build training set and evaluation set TFRecord files

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
import numpy as np
import pathlib

#depth of per image
IMG_CHANNELS = 3

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
#data_dir = '/Users/zhuzhirui/.keras/datasets/flower_photos'

CLASS_NAMES = [item.numpy().decode("utf-8") for item in
               tf.strings.regex_replace(
                 tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/*"),
                 "/Users/zhuzhirui/.keras/datasets/flower_photos/", "")]
print(CLASS_NAMES)

#save all of elements which have not '.' , ignore all of elements which have '.'
CLASS_NAMES = [item for item in CLASS_NAMES if item.find(".") == -1]
print(CLASS_NAMES)
#CLASS_NAMES = ['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips']

#build shuffled image list and shuffled & corresponding label list
def file_convert_numpylist(is_random=True):
    label_list = []
    roses = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/roses/*")]
    for item in roses:
        label_list.append('roses')
    sunflowers = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/sunflowers/*")]
    for item in sunflowers:
        label_list.append('sunflowers')
    daisy = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/daisy/*")]
    for item in daisy:
        label_list.append('daisy')
    dandelion = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/dandelion/*")]
    for item in dandelion:
        label_list.append('dandelion')
    tulips = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/tulips/*")]
    for item in tulips:
        label_list.append('tulips')
    image_list = roses + sunflowers + daisy + dandelion + tulips

    label_list = np.asarray(label_list)
    image_list = np.asarray(image_list)

    if is_random:
        list_index = np.arange(len(image_list))
        np.random.shuffle(list_index)
        label_list = label_list[list_index]
        image_list = image_list[list_index]

    return image_list,label_list

image_list,label_list = file_convert_numpylist(is_random=True)

#divided into training set and evaluation set
train_image_list = image_list[:int(0.9*len(image_list))]
train_label_list = label_list[:int(0.9*len(label_list))]

eval_image_list = image_list[int(0.9*len(image_list)):]
eval_label_list = label_list[int(0.9*len(label_list)):]

#encode int64, bytes, float
def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))

def bytes_feature(values):
    #if isinstance(values,type(tf.constant(0))):
        #values = values.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))

def image_to_tfexample(image, label):
    image_shape = image.shape
    feature = {
        'image_width': int64_feature(image_shape[0]),
        'image_height': int64_feature(image_shape[1]),
        'image_depth':int64_feature(image_shape[2]),
        'label':bytes_feature(label.tobytes()),
        'image':bytes_feature(image.tobytes())
    }
    return tf.train.Features(feature=feature)

def convert_TFRFiles(image_list, label_list, TFRecord_filename):
    with tf.io.TFRecordWriter(TFRecord_filename) as writer:
        for i in range(len(image_list)):
            image = tf.io.read_file(image_list[i])
            image = tf.image.decode_jpeg(image, channels=IMG_CHANNELS)
            image = image.numpy()
            label = label_list[i]
            example = tf.train.Example(features = image_to_tfexample(image, label))
            writer.write(example.SerializeToString())
            print('writing {} of {}'.format(i+1,len(image_list)))
        print('saved in {}'.format(TFRecord_filename))

#training TFRecord
convert_TFRFiles(train_image_list, train_label_list, TFRecord_filename = 'Training_Set.tfrecord')
#evaluation TFRecord
convert_TFRFiles(eval_image_list, eval_label_list, TFRecord_filename = 'Evaluation_Set.tfrecord')

