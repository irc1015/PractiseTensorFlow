import csv
import tensorflow as tf
import numpy as np
import pathlib
import random

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

CLASS_NAMES = [item.numpy().decode("utf-8") for item in
               tf.strings.regex_replace(
                 tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/*"),
                 "/Users/zhuzhirui/.keras/datasets/flower_photos/", "")]
print(CLASS_NAMES)

#save all of elements which have not '.' , ignore all of elements which have '.'
CLASS_NAMES = [item for item in CLASS_NAMES if item.find(".") == -1]
print(CLASS_NAMES)


#label
cols = ["path", "flower"]

#all of samples path
roses = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/roses/*")]
sunflowers = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/sunflowers/*")]
daisy = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/daisy/*")]
dandelion = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/dandelion/*")]
tulips = [item for item in tf.io.gfile.glob("/Users/zhuzhirui/.keras/datasets/flower_photos/tulips/*")]

#joint path and label into a str
for i in range(0,len(roses)):
    roses[i] =str(roses[i]+'@roses')
for i in range(0,len(sunflowers)):
    sunflowers[i] =str(sunflowers[i]+'@sunflowers')
for i in range(0,len(daisy)):
    daisy[i] =str(daisy[i]+'@daisy')
for i in range(0,len(dandelion)):
    dandelion[i] =str(dandelion[i]+'@dandelion')
for i in range(0,len(tulips)):
    tulips[i] =str(tulips[i]+'@tulips')

#split that kind of str to a list[path,label]
roseslist =[item.split('@') for item in roses]
sunflowerslist = [item.split('@') for item in sunflowers]
daisylist = [item.split('@') for item in daisy]
dandelionlist = [item.split('@') for item in dandelion]
tulipslist = [item.split('@') for item in tulips]

#split trainset and evaluationset
train_roses = roseslist[:int(0.9*len(roseslist))]
eval_roses = roseslist[int(0.9*len(roseslist)):]

train_sunflowers = sunflowerslist[:int(0.9*len(sunflowerslist))]
eval_sunflowers = sunflowerslist[int(0.9*len(sunflowerslist)):]

train_daisy = daisylist[:int(0.9*len(daisylist))]
eval_daisy = daisylist[int(0.9*len(daisylist)):]

train_dandelion = dandelionlist[:int(0.9*len(dandelionlist))]
eval_dandelion = dandelionlist[int(0.9*len(dandelionlist)):]

train_tulips = tulipslist[:int(0.9*len(tulipslist))]
eval_tulips = tulipslist[int(0.9*len(tulipslist)):]

train_set = train_roses + train_sunflowers +train_daisy +train_dandelion + train_tulips
eval_set = eval_roses + eval_sunflowers + eval_daisy + eval_dandelion + eval_tulips

print('TrainSet length:'+str(len(train_set)))
print('EvaluationSet length:'+str(len(eval_set)))

random.shuffle(train_set)
random.shuffle(eval_set)
#shuffle the Training Dataset and Evaluation Dataset CSV file

with open(str(data_dir)+'/train_set.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(train_set)

with open(str(data_dir)+'/eval_set.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(eval_set)


