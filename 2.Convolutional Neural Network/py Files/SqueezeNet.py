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

IMG_SIZE = [512, 512]
BATCH_SIZE = 16
EPOCHS = 13

LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .95


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

NUM_TRAINING_IMG = 12753
STEPS_PER_EPOCH = NUM_TRAINING_IMG // BATCH_SIZE
NUM_VALIDATION_IMG = 3712
VALIDATION_STEPS = -(-NUM_VALIDATION_IMG // BATCH_SIZE) #The "-(- // )" trick rounds up instead of down

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

    def fire(x, squeeze, expand):
        y = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
        y = tf.keras.layers.BatchNormalization()(y)
        #contraction stage
        y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)
        y1 = tf.keras.layers.BatchNormalization()(y1)
        #expansion stage 1*1 filter
        y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)
        y3 = tf.keras.layers.BatchNormalization()(y3)
        # expansion stage 3*3 filter
        return tf.keras.layers.concatenate([y1, y3])
        #combine y1 and y3 into a tensor which has channels of (y1 + y3) channels

    def fire_module(squeeze, expand):
        return lambda x: fire(x, squeeze, expand)

    x = tf.keras.layers.Input(shape=[*IMG_SIZE, 3])
    y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(x)#[3,3,3,32]
    #[512,512,3] --> [512,512,32]
    y = tf.keras.layers.BatchNormalization()(y)
    # add this input layer
    y = fire_module(16, 32)(y)
    ''' contraction stage [1,1,32,16]    [512,512,32] --> [512,512,16]
        expansion stage   [1,1,16,16]    [512,512,16] --> [512,512,16]
                          [3,3,16,16]    [512,512,16] --> [512,512,16]
        concatenate [512,512,32]
    '''
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #[512,512,32] --> [256,256,32]
    y = fire_module(48, 96)(y)
    ''' contraction stage [1,1,32,48]    [256,256,32] --> [256,256,48]
        expansion stage   [1,1,48,48]    [256,256,48] --> [256,256,48]
                          [3,3,48,48]    [256,256,48] --> [256,256,48]
        concatenate [256,256,96]
        '''
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #[256,256,96] --> [128,128,96]
    y = fire_module(64, 128)(y)
    ''' contraction stage [1,1,96,64]    [128,128,96] --> [128,128,64]
        expansion stage   [1,1,64,64]    [128,128,64] --> [128,128,64]
                          [3,3,64,64]    [128,128,64] --> [128,128,64]
        concatenate [128,128,128]
        '''
    y = fire_module(80, 160)(y)
    ''' contraction stage [1,1,128,80]    [128,128,128] --> [128,128,80]
        expansion stage   [1,1,80,80]     [128,128,80] --> [128,128,80]
                          [3,3,80,80]     [128,128,80] --> [128,128,80]
        concatenate [128,128,160]
        '''
    y = fire_module(96, 192)(y)
    ''' contraction stage [1,1,160,96]    [128,128,160] --> [128,128,96]
        expansion stage   [1,1,96,96]     [128,128,96] --> [128,128,96]
                          [3,3,96,96]     [128,128,96] --> [128,128,96]
        concatenate [128,128,192]
        '''
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #[128,128,192] --> [64,64,192]
    y = fire_module(112, 224)(y)
    ''' contraction stage [1,1,192,112]    [64,64,192] --> [64,64,112]
        expansion stage   [1,1,112,112]    [64,64,112] --> [64,64,112]
                          [3,3,112,112]    [64,64,112] --> [64,64,112]
        concatenate [64,64,224]
        '''
    y = fire_module(128, 256)(y)
    ''' contraction stage [1,1,224,128]    [64,64,224] --> [64,64,128]
        expansion stage   [1,1,128,128]    [64,64,128] --> [64,64,128]
                          [3,3,128,128]    [64,64,128] --> [64,64,128]
        concatenate [64,64,256]
        '''
    y = fire_module(160, 320)(y)
    ''' contraction stage [1,1,256,160]    [64,64,256] --> [64,64,160]
        expansion stage   [1,1,160,160]    [64,64,160] --> [64,64,160]
                          [3,3,160,160]    [64,64,160] --> [64,64,160]
        concatenate [64,64,320]
        '''
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #[64,64,320] --> [32,32,320]
    y = fire_module(192, 384)(y)
    ''' contraction stage [1,1,320,192]    [32,32,320] --> [32,32,192]
        expansion stage   [1,1,192,192]    [32,32,192] --> [32,32,192]
                          [3,3,192,192]    [32,32,192] --> [32,32,192]
        concatenate [32,32,384]
        '''
    y = fire_module(224, 448)(y)
    ''' contraction stage [1,1,384,224]    [32,32,384] --> [32,32,224]
        expansion stage   [1,1,224,224]    [32,32,224] --> [32,32,224]
                          [3,3,224,224]    [32,32,224] --> [32,32,224]
        concatenate [32,32,448]
        '''
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #[32,32,448] --> [16,16,448]
    y = fire_module(256, 512)(y)
    ''' contraction stage [1,1,448,256]    [16,16,448] --> [16,16,256]
        expansion stage   [1,1,256,256]    [16,16,256] --> [16,16,256]
                          [3,3,256,256]    [16,16,256] --> [16,16,256]
        concatenate [16,16,512]
        '''
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    #Average 512 channels
    y = tf.keras.layers.Dense(len(CLASS), activation='softmax', name='flower_prob')(y)

    model = tf.keras.Model(x, y)

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
    steps_per_execution=1
)
model.summary()

history = model.fit(
    get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
    validation_data=get_validation_dataset(), validation_steps=VALIDATION_STEPS,
    callbacks=[lr_callback]
)