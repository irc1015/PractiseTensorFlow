import tensorflow as tf
import numpy as np
import os,math,re,sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #CPU Only

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'#GPU Running

import matplotlib.pylab as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
AUTO = tf.data.experimental.AUTOTUNE
print("Tensorflow version " + tf.__version__)

strategy = tf.distribute.MirroredStrategy(devices=None) #Use all available GPUs or CPU
print("REPLICAS:", strategy.num_replicas_in_sync)

IMG_SIZE = [224, 224]
BATCH_SIZE = 16
EPOCHS = 13

LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8


def scheduler_epoch(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        learning_rate = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        learning_rate = LR_MAX
    else:
        learning_rate = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return learning_rate

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_epoch, verbose=True)

DataSet_Path = '/Users/zhuzhirui/.keras/datasets/104_Flowers'
Train_Size_Choice = {
    192: DataSet_Path + '/jpeg-192x192_train.tfrecord',
    224: DataSet_Path + '/jpeg-224x224_train.tfrecord',
    311: DataSet_Path + '/jpeg-311x311_train.tfrecord',
    512: DataSet_Path + '/jpeg-512x512_train.tfrecord'
}

Val_Size_Choice = {
    192: DataSet_Path + '/jpeg-192x192_val.tfrecord',
    224: DataSet_Path + '/jpeg-224x224_val.tfrecord',
    311: DataSet_Path + '/jpeg-311x311_val.tfrecord',
    512: DataSet_Path + '/jpeg-512x512_val.tfrecord'
}

TrainSet_Path = Train_Size_Choice[IMG_SIZE[0]]
ValSet_Path = Val_Size_Choice[IMG_SIZE[0]]

CLASS = ['toad lily', 'love in the mist', 'monkshood', 'azalea', 'fritillary',
         'silverbush', 'canterbury bells', 'stemless gentian', 'pink primrose', 'buttercup',
         'poinsettia', 'desert-rose', 'bird of paradise', 'columbine', 'frangipani',
         'sweet pea', 'siam tulip', 'great masterwort', 'hard-leaved pocket orchid', 'marigold',
         'foxglove', 'wild pansy', 'windflower', 'daisy', 'tiger lily',
         'purple coneflower', 'orange dahlia', 'globe-flower', 'lilac hibiscus', 'fire lily',
         'balloon flower', 'iris', 'bishop of llandaff', 'yellow iris', 'garden phlox',
         'alpine sea holly', 'geranium', 'pink quill', 'tree poppy', 'spear thistle',
         'bromelia', 'common dandelion', 'sword lily', 'peruvian lily', 'carnation',
         'cosmos', 'spring crocus', 'lotus', 'bolero deep blue', 'anthurium',
         'rose', 'water lily', 'primula', 'blackberry lily', 'gaura',
         'trumpet creeper', 'globe thistle', 'sweet william', 'snapdragon', 'mexican petunia',
         'cyclamen ', 'petunia', 'gazania', 'king protea', 'blanket flower',
         'common tulip', 'giant white arum lily', 'wild rose', 'morning glory', 'thorn apple',
         'pincushion flower', 'tree mallow', 'canna lily', 'camellia', 'pink-yellow dahlia',
         'bee balm', 'wild geranium', 'artichoke', 'black-eyed susan', 'ruby-lipped cattleya',
         'clematis', 'prince of wales feathers', 'hibiscus', 'cautleya spicata', 'lenten rose',
         'red ginger', "colt's foot", 'hippeastrum ', 'mallow', 'californian poppy',
         'corn poppy', 'moon orchid', 'passion flower', 'grape hyacinth', 'japanese anemone',
         'watercress', 'cape flower', 'osteospermum', 'barberton daisy', 'bougainvillea',
         'magnolia', 'sunflower', 'daffodil', 'wallflower']

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3) #JPEG --> Tensor
    image = tf.image.convert_image_dtype(image, tf.float32)
    #image = tf.reshape(image, [*IMG_SIZE, 3])
    image = tf.image.resize(image, IMG_SIZE) #image must be 3-D, so resize it
    return image

def read_tfrecord(example):
    image_feature_dict = {
        'image': tf.io.FixedLenFeature([],tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_id': tf.io.FixedLenFeature([],tf.string)
    }
    example = tf.io.parse_single_example(example, image_feature_dict) #return a dict
    image = decode_image(example['image'])
    label = tf.cast(example['label'], tf.int64) #104 classes in[0,103]
    image_id = example['image_id']
    return image,label   #return a tuple

def load_dataset(filenames, ordered = False):
    ignore_order = tf.data.Options()

    if not ordered:
        ignore_order.deterministic = False # experimental_deterministic has been baned

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def get_training_dataset():
    dataset = load_dataset(TrainSet_Path) #ordered default is False
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) #Prepare later elements
    return dataset

def get_validation_dataset(ordered = False):
    dataset = load_dataset(ValSet_Path, ordered = ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache() #cache files in memory
    dataset = dataset.prefetch(AUTO)
    return dataset

with strategy.scope():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=[*IMG_SIZE, 3]),  #224*224*3
            tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'), #[11,11,3,96]
            #[224, 224, 3] --> [54, 54, 96]
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu'), #[5, 5, 96, 256]
            #[54, 54, 96] --> [50, 50, 256]
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2), #[50, 50, 256] --> [25, 25, 256]
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'), #[3, 3, 256, 384]
            #[25, 25, 256] --> [23, 23, 384]
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2), #[23, 23, 384] --> [11, 11, 384]
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'), #[3, 3, 384, 384]
            #[11, 11, 384] --> [9, 9, 384]
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'), #[3, 3, 384, 256]
            #[9, 9, 384] --> [7, 7, 256]
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2), #[7, 7, 256] --> [3, 3, 256]
            tf.keras.layers.Flatten(), #3 * 3 * 256 = 2304
            tf.keras.layers.Dense(4096, activation='relu'), #64 * 64
            tf.keras.layers.Dense(4096, activation='relu'), #64 * 64
            tf.keras.layers.Dense(len(CLASS), activation='softmax')
        ]
    )

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
    steps_per_execution=1
)

model.summary()

NUM_TRAINING_IMG = 12753
STEPS_PER_EPOCH = NUM_TRAINING_IMG // BATCH_SIZE
NUM_VALIDATION_IMG = 3712
VALIDATION_STEPS = -(-NUM_VALIDATION_IMG // BATCH_SIZE) #The "-(- // )" trick rounds up instead of down

history = model.fit(
    get_training_dataset(), step_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
    validation_data=get_validation_dataset(), validation_steps=VALIDATION_STEPS,
    callbacks=[lr_callback]
)

