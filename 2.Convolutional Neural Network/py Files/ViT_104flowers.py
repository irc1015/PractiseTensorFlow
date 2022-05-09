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
BATCH_SIZE = 12
EPOCHS = 40

LR_START = 0.0002
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 10
LR_SUSTAIN_EPOCHS = 15
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
    #dataset = dataset.cache() #cache files in memory  Might lead to session die
    dataset = dataset.prefetch(AUTO)
    return dataset

print("Training data shapes:")
for image, label in get_training_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training label examples:", label.numpy())
labelarray = label.numpy()
print(labelarray[0],type(labelarray[0]))
imagearray = image.numpy()
print(imagearray[0], type(imagearray[0]))

PATCH_SIZE = 16
num_patches = (IMG_SIZE[0] // PATCH_SIZE) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [ projection_dim * 2, projection_dim ] #Size of transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] #Size of the dense layers of final classifier

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self,images):
        batch_size = tf.shape(images)[0] #[batch, height, width, channel]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units = projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim = num_patches, output_dim = projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units: # hidden_units[0], hidden_units[1]...hidden_units[n]
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier():
    inputs = tf.keras.layers.Input(shape=[*IMG_SIZE, 3]) #[224, 224, 3] --> input
    patches = Patches(PATCH_SIZE)(inputs) #return the whole patches of a batch image
    encodeed_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers): #Don not care about the value, just care about the loop times
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encodeed_patches)

        #multi-head attention layer：to learn which parts of the input to focus on
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                              key_dim=projection_dim,
                                                              dropout=0.1)(x1, x1)
        #skip connection
        x2 = tf.keras.layers.Add()([attention_output, encodeed_patches])

        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        #skip connection
        encodeed_patches = tf.keras.layers.Add()([x3, x2])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encodeed_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = tf.keras.layers.Dense(len(CLASS))(features)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


with strategy.scope():
    model = create_vit_classifier()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
    steps_per_execution=8
)
model.summary()